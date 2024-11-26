import torch

import numpy as np
import contextlib
import math

def check_armijo_conditions(step_size, step_size_old, loss, grad_current,direction_current,
                      loss_next, c, beta_b,device):
    found = 0
    g_d=(compute_inner_product(grad_current,direction_current,device))
    break_condition = loss_next - \
        (loss - (step_size) * c * g_d)

    if (break_condition <= 0):
        found = 1


    else:
        # decrease the step-size by a multiplicative factor
        step_size = step_size * beta_b

    return found, step_size, step_size_old


def check_strong_wolfe_conditions(step_size, step_size_old, loss, grad_current,direction_current,
                      loss_next, c1,c2, beta_b,grad_next_t_grad_current,grad_norm):
    found = 0
    break_condition = loss_next - \
        (loss - (step_size) * c1 * grad_norm**2)
    
    if (break_condition <= 0):
        b=abs(grad_next_t_grad_current)-abs(c2*grad_norm**2)
        print('armijo')
        if (b<=0):
            print('strong_wolfe')
            found = 1
        else:
            step_size = step_size * beta_b

    else:
        step_size=step_size*beta_b
    return found, step_size, step_size_old

def check_wolfe_conditions(step_size, step_size_old, loss, grad_current,direction_current,
                      loss_next, c1,c2, beta_b,grad_next_t_grad_current,grad_norm):
    found = 0
    break_condition = loss_next - \
        (loss - (step_size) * c1 * grad_norm**2)

    if (break_condition <= 0):
        b=(grad_next_t_grad_current)-(grad_norm**2)

        print(int(b))
        if (b<=0):
            found = 1
        else:
            step_size = step_size * beta_b


    else:
        # decrease the step-size by a multiplicative factor
        step_size = step_size * beta_b

    return found, step_size, step_size_old



def reset_step(step_size,eta_max,n_batches_per_epoch=None, gamma=None, reset_option=1,
               init_step_size=None,step_size_list=None):
    if reset_option == 0:
        pass

    elif reset_option == 1:
        step_size = min(step_size * gamma**(1. / n_batches_per_epoch),eta_max)

    elif reset_option == 2:
        step_size = init_step_size
    
    elif reset_option == 3:
        if step_size<1e-5:
            step_size=1e-3

        else:
            step_size = step_size * gamma**(1. / n_batches_per_epoch)
    elif reset_option == 4:

        if len(step_size_list)<=n_batches_per_epoch:
            return init_step_size
        else:

            start=((len(step_size_list)//n_batches_per_epoch)-1)*n_batches_per_epoch
            end=start+n_batches_per_epoch
            step_size_avg = sum(step_size_list[start:end])/n_batches_per_epoch
            step_size=min(step_size_avg*2,eta_max)
        

    return step_size

def try_sgd_update(params, step_size, params_current, grad_current):
    zipped = zip(params, params_current, grad_current)

    for p_next, p_current, g_current in zipped:
        p_next.data = p_current - step_size * g_current

def compute_norm(grad_list,device):
    grad_norm=torch.tensor(0.,device=device)
    for g in grad_list:
        if g is None:
            continue
        grad_norm += torch.sum(torch.mul(g, g))
    grad_norm = torch.sqrt(grad_norm)
    return grad_norm

def mul_const_list(const, list_torch):
    multiplied_list = [l * const for l in list_torch]
    return multiplied_list

def momentum_direction(params,grad_current,momentum_buffer_list,weight_decay,momentum):
    direction=[]
    for i, (param,g_current) in enumerate(zip(params,grad_current)):
        if weight_decay != 0:
            g_current_tmp=g_current.add(param,alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(g_current_tmp).detach()
                momentum_buffer_list[i] = buf
            else:
                torch.add(g_current_tmp,buf,alpha=momentum,out=g_current_tmp)
            d_current = g_current_tmp.add(buf, alpha=momentum)
        direction.append(d_current)
    return direction

def compute_inner_product(grad_current,direction_current,device):
    g_d_inner_product=torch.tensor(0.,device=device)
    for g,d in zip(grad_current,direction_current):
        if g is None:
            continue
        g_t=torch.flatten(g)
        d_t=torch.flatten(d)
        g_d_inner_product+=torch.sum(torch.dot(g_t,d_t))
    return g_d_inner_product

def weight_norm(params,device):
    weight_sum=torch.tensor(0.,device=device)
    youso=0
    for p in params:
        p_f=torch.flatten(p)
        weight_sum+=torch.sum(torch.dot(p_f,p_f))
        youso+=torch.numel(p)
    return weight_sum



def get_grad_list(params):
    return [p.grad for p in params]

@contextlib.contextmanager
def random_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

@contextlib.contextmanager
def random_seed_torch(seed, device=0):
    cpu_rng_state = torch.get_rng_state()
    if torch.cuda.is_available():
        gpu_rng_state = torch.cuda.get_rng_state(0)

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        yield
    finally:
        torch.set_rng_state(cpu_rng_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(gpu_rng_state, device)
