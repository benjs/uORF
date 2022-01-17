import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.optim import lr_scheduler


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    elif opt.lr_policy == 'warmup':
        scheduler = get_exp_decay_schedule_with_warmup(optimizer, num_warmup_steps=opt.warmup_steps, num_decay_steps=opt.attn_decay_steps)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    if init_type is not None:
        init_weights(net, init_type, init_gain=init_gain)
    return net




# Wrap lambda into class to make it pickle
class lr_exp_decay_schedule_with_warmup:
    def __init__(self, num_warmup_steps, num_decay_steps=2e5, decay_base=0.5, n_start_decay=0):
        self.num_warmup_steps = num_warmup_steps
        self.num_decay_steps = num_decay_steps
        self.decay_base = decay_base
        self.n_start_decay = n_start_decay
    
    def __call__(self, current_step: int):
        if current_step < self.num_warmup_steps:
            rate = float(current_step) / float(max(1, self.num_warmup_steps))
        else:
            if current_step < self.num_warmup_steps + self.n_start_decay:
                rate = 1
            else:
                rate = self.decay_base **(float(current_step - self.num_warmup_steps - self.n_start_decay) / float(self.num_decay_steps))

        return rate

def lr_lambda_exp_decay_schedule_with_warmup(current_step: int):
    if current_step < num_warmup_steps:
        rate = float(current_step) / float(max(1, num_warmup_steps))
    else:
        if current_step < num_warmup_steps + n_start_decay:
            rate = 1
        else:
            rate = decay_base **(float(current_step - num_warmup_steps - n_start_decay) / float(num_decay_steps))
    return rate

def get_exp_decay_schedule_with_warmup(optimizer, num_warmup_steps, num_decay_steps=2e5,
                                       decay_base=0.5, last_epoch=-1, n_start_decay=0):
    """
    Create a schedule with a learning rate that decays exponentialy from the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The totale number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    return lr_scheduler.LambdaLR(
        optimizer, 
        lr_exp_decay_schedule_with_warmup(num_warmup_steps, num_decay_steps, decay_base, n_start_decay), 
        last_epoch)


class lr_exp_decay_schedule:
    def __init__(self, num_decay_steps=2e5, decay_base=0.5):
        self.num_decay_steps = num_decay_steps
        self.decay_base = decay_base
    
    def __call__(self, current_step: int):
        return self.decay_base **(float(current_step) / float(self.num_decay_steps))

def lr_lambda_exp_decay_schedule(current_step: int):
    return decay_base **(float(current_step) / float(num_decay_steps))

def get_exp_decay_schedule(optimizer, num_decay_steps=1e5,
                                       decay_base=0.5, last_epoch=-1):

    return lr_scheduler.LambdaLR(optimizer, lr_exp_decay_schedule(num_decay_steps, decay_base), last_epoch)

if __name__ == '__main__':
    num_decay_steps = 200000
    n_start_decay = 100000
    decay_base = 0.5
    last_epoch = -1
    num_warmup_steps = 1000

    x = nn.Parameter(torch.tensor([0.,1]))
    opm = torch.optim.Adam([x,], lr=1.)
    lr_s = get_exp_decay_schedule_with_warmup(opm, num_warmup_steps, num_decay_steps=num_decay_steps, n_start_decay=n_start_decay)
    for i in range(5000000):
        lr_s.step()
        if i in [1000, 10000, 100000, 200000, 300000, 400000, 499999]:
            print('i={}, lr_factor={}'.format(i, opm.param_groups[0]['lr']))