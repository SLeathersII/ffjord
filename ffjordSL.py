import matplotlib
import matplotlib.pyplot as plt
#plt.switch_backend('TkAgg')
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint
torch.backends.cudnn.benchmark = True
SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams']
import time
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as tforms
from .lib import layers as layers
import ffjord.lib.odenvp as odenvp
import ffjord.lib.multiscale_parallel as multiscale_parallel
from .train_misc import standard_normal_logprob
from .train_misc import set_cnf_options, count_nfe, count_parameters, count_total_time
from .train_misc import add_spectral_norm, spectral_norm_power_iteration
from .train_misc import create_regularization_fns, get_regularization, append_regularization_to_log




class ffjord_args:
    def __init__(self,
                data: str="mnist", # choices=["mnist", "svhn", "cifar10", 'lsun_church'],
                dims: str = "64,64,64",
                strides: str = "1,1,1,1",
                num_blocks: int= 2, # help='Number of stacked CNFs.')
                conv:bool = True, # choices=[True, False])
                layer_type: str = "concat", # choices=["ignore", "concat", "concat_v2", "squash", "concatsquash", "concatcoord", "hyper", "blend"]
                divergence_fn: str = "approximate", # choices=["brute_force", "approximate"])
                nonlinearity: str = "softplus", # choices=["tanh", "relu", "softplus", "elu", "swish"]
                solver: str = 'dopri5', # choices=SOLVERS)
                atol: float= 1e-5,
                rtol: float= 1e-5,
                step_size: float= None, #help="Optional fixed step size.",

                test_solver: str= None, #choices=SOLVERS + [None])
                test_atol: float= None,
                test_rtol: float= None,

                imagesize: int= None,
                alpha: float= 1e-6,
                time_length: float= 1.0,
                train_T:bool= True,

                num_epochs: int= 1000,
                batch_size: int= 200,
                batch_size_schedule: str= "", # help="Increases the batchsize at every given epoch, dash separated."
                test_batch_size: int= 200,
                lr: float= 1e-3,
                warmup_iters: float= 1000,
                weight_decay: float= 0.0,
                spectral_norm_niter: int= 10,

                add_noise:bool= True, #choices=[True, False],
                batch_norm:bool= False, #choices=[True, False],
                residual:bool= False, #choices=[True, False],
                autoencode:bool= False, #choices=[True, False],
                rademacher:bool= True, #choices=[True, False],
                spectral_norm:bool= False, #choices=[True, False],
                multiscale:bool= True, #choices=[True, False],
                parallel:bool= False, #choices=[True, False],

                # Regularizations
                l1int: float= None, #help="int_t ||f||_1",
                l2int: float= None, #help="int_t ||f||_2",
                dl2int: float= None, #help="int_t ||f^T df/dt||_2",
                JFrobint: float= None, #help="int_t ||df/dx||_F",
                JdiagFrobint: float= None, #help="int_t ||df_i/dx_i||_F",
                JoffdiagFrobint: float= None, #help="int_t ||df/dx - df_i/dx_i||_F",

                time_penalty: float= 0, #help="Regularization on the end_time.",
                max_grad_norm: float= 1e10, # help="Max norm of graidents (default is just stupidly high to avoid any clipping)"
                begin_epoch: int= 1,
                resume: str= None,
                save: str= "experiments/",
                val_freq: int= 1,
                log_freq: int= 10,):
        
        self.data = data
        self.dims = dims
        self.strides = strides
        self.num_blocks = num_blocks
        self.conv = conv
        self.layer_type = layer_type
        self.divergence_fn = divergence_fn
        self.nonlinearity = nonlinearity
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.step_size = step_size
        self.test_solver = test_solver
        self.test_atol = test_atol
        self.test_rtol = test_rtol
        self.imagesize = imagesize
        self.alpha = alpha
        self.time_length = time_length
        self.train_T = train_T
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.batch_size_schedule = batch_size_schedule
        self.test_batch_size = test_batch_size
        self.lr = lr
        self.warmup_iters = warmup_iters
        self.weight_decay = weight_decay
        self.spectral_norm_niter = spectral_norm_niter
        
        self.add_noise = add_noise
        self.batch_norm = batch_norm
        self.residual = residual
        self.autoencode = autoencode
        self.rademacher = rademacher
        self.spectral_norm = spectral_norm
        self.multiscale = multiscale
        self.parallel = parallel
        self.l1int = l1int
        self.l2int = l2int
        self.dl2int = dl2int
        self.JFrobint = JFrobint
        self.JdiagFrobint = JdiagFrobint
        self.JoffdiagFrobint = JoffdiagFrobint
        self.time_penalty = time_penalty
        self.max_grad_norm = max_grad_norm
        self.begin_epoch = begin_epoch
        self.resume = resume
        self.save = save
        self.val_freq = val_freq
        self.log_freq = log_freq

def add_noise(x):
    """
    [0, 1] -> [0, 255] -> add noise -> [0, 1]
    """
    
    noise = x.new().resize_as_(x).uniform_()
    x = x * 255 + noise
    x = x / 256
    return x


def update_lr(optimizer, itr, args:ffjord_args):
    iter_frac = min(float(itr + 1) / max(args.warmup_iters, 1), 1.0)
    lr = args.lr * iter_frac
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_train_loader(train_set, epoch, args:ffjord_args):
    if args.batch_size_schedule != "":
        epochs = [0] + list(map(int, args.batch_size_schedule.split("-")))
        n_passed = sum(np.array(epochs) <= epoch)
        current_batch_size = int(args.batch_size * n_passed)
    else:
        current_batch_size = args.batch_size
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=current_batch_size, shuffle=True, drop_last=True, pin_memory=True
    )
    
    return train_loader


def get_dataset(args:ffjord_args):
    if args.add_noise:
        trans = lambda im_size: tforms.Compose([tforms.Resize(im_size), tforms.ToTensor(), add_noise])
    else:
        trans = lambda im_size: tforms.Compose([tforms.Resize(im_size), tforms.ToTensor()])
    if args.data == "mnist":
        im_dim = 1
        im_size = 28 if args.imagesize is None else args.imagesize
        train_set = dset.MNIST(root="./data", train=True, transform=trans(im_size), download=True)
        test_set = dset.MNIST(root="./data", train=False, transform=trans(im_size), download=True)
    elif args.data == "svhn":
        im_dim = 3
        im_size = 32 if args.imagesize is None else args.imagesize
        train_set = dset.SVHN(root="./data", split="train", transform=trans(im_size), download=True)
        test_set = dset.SVHN(root="./data", split="test", transform=trans(im_size), download=True)
    elif args.data == "cifar10":
        im_dim = 3
        im_size = 32 if args.imagesize is None else args.imagesize
        if args.add_noise:
            train_set = dset.CIFAR10(
                root="./data", train=True, transform=tforms.Compose([
                    tforms.Resize(im_size),
                    tforms.RandomHorizontalFlip(),
                    tforms.ToTensor(),
                    add_noise,
                ]), download=True
            )
        else:
            train_set = dset.CIFAR10(
                root="./data", train=True, transform=tforms.Compose([
                    tforms.Resize(im_size),
                    tforms.RandomHorizontalFlip(),
                    tforms.ToTensor(),
                    add_noise,
                ]), download=True
            )
        test_set = dset.CIFAR10(root="./data", train=False, transform=trans(im_size), download=True)
    elif args.data == 'celeba':
        im_dim = 3
        im_size = 64 if args.imagesize is None else args.imagesize
        train_set = dset.CelebA(
            train=True, transform=tforms.Compose([
                tforms.ToPILImage(),
                tforms.Resize(im_size),
                tforms.RandomHorizontalFlip(),
                tforms.ToTensor(),
                add_noise,
            ])
        )
        test_set = dset.CelebA(
            train=False, transform=tforms.Compose([
                tforms.ToPILImage(),
                tforms.Resize(im_size),
                tforms.ToTensor(),
                add_noise,
            ])
        )
    elif args.data == 'lsun_church':
        im_dim = 3
        im_size = 64 if args.imagesize is None else args.imagesize
        train_set = dset.LSUN(
            'data', ['church_outdoor_train'], transform=tforms.Compose([
                tforms.Resize(96),
                tforms.RandomCrop(64),
                tforms.Resize(im_size),
                tforms.ToTensor(),
                add_noise,
            ])
        )
        test_set = dset.LSUN(
            'data', ['church_outdoor_val'], transform=tforms.Compose([
                tforms.Resize(96),
                tforms.RandomCrop(64),
                tforms.Resize(im_size),
                tforms.ToTensor(),
                add_noise,
            ])
        )
    data_shape = (im_dim, im_size, im_size)
    if not args.conv:
        data_shape = (im_dim * im_size * im_size,)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=args.test_batch_size, shuffle=False, drop_last=True
    )
    return train_set, test_loader, data_shape


def compute_bits_per_dim(x, model):
    zero = torch.zeros(x.shape[0], 1).to(x)

    # Don't use data parallelize if batch size is small.
    # if x.shape[0] < 200:
    #     model = model.module

    z, delta_logp = model(x, zero)  # run model forward

    logpz = standard_normal_logprob(z).view(z.shape[0], -1).sum(1, keepdim=True)  # logp(z)
    logpx = logpz - delta_logp

    logpx_per_dim = torch.sum(logpx) / x.nelement()  # averaged over batches
    bits_per_dim = -(logpx_per_dim - np.log(256)) / np.log(2)

    return bits_per_dim, z, logpx


def create_model(args:ffjord_args, data_shape, regularization_fns):
    hidden_dims = tuple(map(int, args.dims.split(",")))
    strides = tuple(map(int, args.strides.split(",")))

    if args.multiscale:
        model = odenvp.ODENVP(
            (args.batch_size, *data_shape),
            n_blocks=args.num_blocks,
            intermediate_dims=hidden_dims,
            nonlinearity=args.nonlinearity,
            alpha=args.alpha,
            cnf_kwargs={"T": args.time_length, "train_T": args.train_T, "regularization_fns": regularization_fns},
        )
    elif args.parallel:
        model = multiscale_parallel.MultiscaleParallelCNF(
            (args.batch_size, *data_shape),
            n_blocks=args.num_blocks,
            intermediate_dims=hidden_dims,
            alpha=args.alpha,
            time_length=args.time_length,
        )
    else:
        if args.autoencode:

            def build_cnf():
                autoencoder_diffeq = layers.AutoencoderDiffEqNet(
                    hidden_dims=hidden_dims,
                    input_shape=data_shape,
                    strides=strides,
                    conv=args.conv,
                    layer_type=args.layer_type,
                    nonlinearity=args.nonlinearity,
                )
                odefunc = layers.AutoencoderODEfunc(
                    autoencoder_diffeq=autoencoder_diffeq,
                    divergence_fn=args.divergence_fn,
                    residual=args.residual,
                    rademacher=args.rademacher,
                )
                cnf = layers.CNF(
                    odefunc=odefunc,
                    T=args.time_length,
                    regularization_fns=regularization_fns,
                    solver=args.solver,
                )
                return cnf
        else:

            def build_cnf():
                diffeq = layers.ODEnet(
                    hidden_dims=hidden_dims,
                    input_shape=data_shape,
                    strides=strides,
                    conv=args.conv,
                    layer_type=args.layer_type,
                    nonlinearity=args.nonlinearity,
                )
                odefunc = layers.ODEfunc(
                    diffeq=diffeq,
                    divergence_fn=args.divergence_fn,
                    residual=args.residual,
                    rademacher=args.rademacher,
                )
                cnf = layers.CNF(
                    odefunc=odefunc,
                    T=args.time_length,
                    train_T=args.train_T,
                    regularization_fns=regularization_fns,
                    solver=args.solver,
                )
                return cnf

        chain = [layers.LogitTransform(alpha=args.alpha)] if args.alpha > 0 else [layers.ZeroMeanTransform()]
        chain = chain + [build_cnf() for _ in range(args.num_blocks)]
        if args.batch_norm:
            chain.append(layers.MovingBatchNorm2d(data_shape[0]))
        model = layers.SequentialFlow(chain)
    return model



def ffjord_train(model, 
                 args: ffjord_args,
                 train_set,
                 test_loader,
                 device = None,
                 device_ids = None,
                 exp_name: str = 'tmp',
                 ):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    regularization_fns, regularization_coeffs = create_regularization_fns(args)
    cvt = lambda x: x.type(torch.float32).to(device, non_blocking=True)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_hist = []
    best_loss = float("inf")
    itr = 0
    for epoch in range(args.begin_epoch, args.num_epochs + 1):
        model.train()
        train_loader = get_train_loader(train_set, epoch, args)
        for _, (x, y) in enumerate(train_loader):
            start = time.time()
            update_lr(optimizer, itr, args)
            optimizer.zero_grad()

            if not args.conv:
                x = x.view(x.shape[0], -1)

            # cast data and move to device
            x = cvt(x)
            # compute loss
            loss,z,logpz = compute_bits_per_dim(x, model)
            if regularization_coeffs:
                reg_states = get_regularization(model, regularization_coeffs)
                reg_loss = sum(
                    reg_state * coeff for reg_state, coeff in zip(reg_states, regularization_coeffs) if coeff != 0
                )
                loss = loss + reg_loss
            total_time = count_total_time(model)
            loss = loss + total_time * args.time_penalty
            loss_hist.append(loss)
            loss.backward()
            

            optimizer.step()

            if args.spectral_norm: spectral_norm_power_iteration(model, args.spectral_norm_niter)

            print(f'epoch: {epoch} -- itr: {itr} -- loss {loss}')
            itr += 1
            

            

        # compute test loss
        model.eval()
        if epoch % args.val_freq == 0:
            with torch.no_grad():
                start = time.time()
                print('validating-------------------------------------------------')
                losses = []
                for (x, y) in test_loader:
                    if not args.conv:
                        x = x.view(x.shape[0], -1)
                    x = cvt(x)
                    loss,z,logpz = compute_bits_per_dim(x, model)
                    losses.append(loss)

                loss = torch.mean(torch.tensor(losses))
                print(f"Epoch {epoch:04d} | Time {time.time()-start:.4f}, Bit/dim {loss:.4f}")
                if loss < best_loss:
                    best_loss = loss
                    print('best model saved--------------------------------')
                    model_return = deepcopy(model)
                    torch.save((model,args), f'{args.save}{exp_name}_epoch{epoch}.pth')
    return model_return, loss_hist