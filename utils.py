import torch
import torch.nn as nn

import math, os, copy
import numpy as np
from scipy.optimize import bisect

import warnings

from torch.utils.data import DataLoader
import sys
from torch.optim import Optimizer

sys.path.append('./')
from myDataLoader import bucket_dataset

warnings.filterwarnings("ignore")


def bounded_cross_entropy(x, y, eps=-1):
    pred = F.log_softmax(x, dim=-1)
    pred = torch.log(torch.exp(pred) + math.exp(eps))
    return F.nll_loss(pred, y, reduce=False)


def evaluation(model, criterion, dataset, log, lr):
    model.eval()
    device = next(model.parameters()).device
    log.eval(len_dataset=len(dataset.test))
    losses, count = 0, 0
    with torch.no_grad():
        for batch in dataset.test:
            inputs, targets = (b.to(device) for b in batch)
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            correct = torch.argmax(predictions, 1) == targets
            log(None, loss.cpu(), correct.cpu(), lr)
            losses += loss.sum().item()
            count += inputs.shape[0]
    return losses / count


def noise_injection(model, p):
    k = 0
    device = next(model.parameters()).device
    noises, noises_scaled = [], []
    for i, param in enumerate(model.parameters()):
        if not param.requires_grad: continue
        t = len(param.view(-1))
        local_noise = torch.clip(torch.randn(param.data.size(), device=device), min=-2, max=2)
        noises.append(local_noise)
        scaled_local_noise = 0
        if torch.is_tensor(p) and p.dim() > 0:
            # print("p {}\tparam {}\tnoise {}".format(p.dim(),param.data.size(),local_noise.data.size()))
            scaled_local_noise = torch.mul(torch.reshape(torch.exp(p[k:(k + t)]).data, param.data.size()), local_noise)
            noises_scaled.append(scaled_local_noise)
        else:
            scaled_local_noise = local_noise * p
            noises_scaled.append(scaled_local_noise)
        # print(param.data[:5],scaled_local_noise[:5])
        param.data += scaled_local_noise
        k += t
    return noises, noises_scaled


def rm_injected_noises(model, noises_scaled):
    injected_noise = copy.deepcopy(noises_scaled)
    for i, param in enumerate(model.parameters()):
        if not param.requires_grad: continue
        param.data -= injected_noise[0]
        injected_noise = injected_noise[1:]

    return


def weight_decay(model, w0):
    k, weights = 0, 0
    for i, param in enumerate(model.parameters()):
        if not param.requires_grad: continue
        t = len(param.view(-1))
        # print(param.size(),len(w0))
        weights += torch.norm(param.view(-1) - w0[k:(k + t)]) ** 2
        k += t
    return weights


def get_kl_term(weight_decay, p, samples, we=None, layers=1, maxwe=1e5):
    denominator = weight_decay + torch.norm(torch.exp(p)) ** 2
    if we is None:
        we = torch.clip(len(p) / denominator, max=maxwe)
    kl = 0.5 * ((-2 * p).sum() - len(p) * torch.log(we) - len(p) + we * denominator)
    return (6 * (kl + 60 * layers) / samples) ** 0.5, kl, 1 / we ** 0.5


def get_kl_term_with_b(weight_decay, p, b):
    d = len(p)
    KL = (torch.exp(-2 * b.double()) * torch.exp(2 * (p).double()).sum() / d -
          (2 * (p).double().sum() / d - 2 * b.double() + 1))
    return (KL * d + weight_decay * torch.exp(-2 * b)) / 2


def kl_term_backward_mean(kl_loss, model, p, noises):
    grad_loss = []
    copy_noise = copy.deepcopy(noises)
    for i, (n, param) in enumerate(model.named_parameters()):  # gradient for p
        if not param.requires_grad: continue
        # print(n)
        grad_loss.append(torch.mul(copy_noise[0], param.grad).view(-1))
        copy_noise = copy_noise[1:]
    kl_loss.backward()
    # gradient for p
    k = 0
    # copy_noise = copy.deepcopy(noises)
    for i, param in enumerate(model.parameters()):
        if not param.requires_grad: continue
        t = len(param.grad.view(-1))
        g = torch.mul(grad_loss[0].view(-1), torch.exp(p.data[k:(k + t)]))
        # print('ggggggggggggg',g.shape)
        grad_loss = grad_loss[1:]
        p.grad[k:(k + t)] += g
        p.grad[k:(k + t)] = p.grad[k:(k + t)].mean() * (torch.ones(t, device=p.device))
        # print(float(p.grad[k:(k + t)].mean()))
        k += t
    return


def get_kl_term_layer_pb(model, wdecay_mulb, p, b):
    k, KL1, KL2, j = 0, 0, 0, 0
    for i, param in enumerate(model.parameters()):
        if not param.requires_grad: continue
        t = len(param.view(-1))
        # print('b\t',float(b[j]),'p\t',float(p[k:(k+t)].mean()))
        KL1 += torch.exp(-2 * b[j].double()) * torch.exp(2 * (p[k:(k + t)]).double()).sum()
        KL2 += 2 * b[j].double() * t
        k += t
        j += 1

    KL = KL1 - (2 * (p).double().sum() - KL2 + len(p))
    # print(float(KL1),float(KL))
    return (KL + wdecay_mulb) / 2


def weight_decay_mulb(model, b, w0):
    # noise injection
    k, weights, j = 0, 0, 0
    for i, param in enumerate(model.parameters()):
        if not param.requires_grad: continue
        t = len(param.view(-1))
        weights += torch.norm(param.view(-1) - w0[k:(k + t)]) ** 2 * torch.exp(-2 * b[j].double())
        k += t
        j += 1
    return weights


def kl_term_backward(kl_loss, model, p, noises):
    grad_loss = []
    copy_noise = copy.deepcopy(noises)
    for i, param in enumerate(model.parameters()):  # gradient for p
        if not param.requires_grad: continue

        grad_loss.append(torch.mul(copy_noise[0], param.grad).view(-1))
        copy_noise = copy_noise[1:]
    kl_loss.backward()
    # gradient for p
    k = 0
    # copy_noise = copy.deepcopy(noises)
    for i, param in enumerate(model.parameters()):
        if not param.requires_grad: continue
        t = len(param.grad.view(-1))
        g = torch.mul(grad_loss[0].view(-1), torch.exp(p.data[k:(k + t)]))
        grad_loss = grad_loss[1:]
        p.grad[k:(k + t)] += g
        k += t
    return


def initialization(args, model, w0decay=1.0):
    for param in model.parameters():
        if not param.requires_grad: continue
        param.data *= w0decay

    device = next(model.parameters()).device
    noises, noises_scaled, w0 = [], [], []
    pretrain_dim, clf_dim, num_layer = 0, 0, 0
    w0_pretrain, w0_clf = [], []
    initial_dims = 0
    paramdim2mean = {}
    model_type = args.model.split("-")[0].lower()
    for layer, (n, param) in enumerate(model.named_parameters()):
        if not param.requires_grad: continue

        if model_type in n.lower():
            pretrain_dim += len(param.data.view(-1))
            w0_pretrain.append(param.data.view(-1).detach().clone())
        else:
            clf_dim += len(param.data.view(-1))
            w0_clf.append(param.data.view(-1).detach().clone())
        initial_dims += len(param.data.view(-1))
        weight_mean = param.abs().mean()
        paramdim2mean[initial_dims] = weight_mean
        w0.append(param.data.view(-1).detach().clone())
        # print(param.data.size())
        num_layer += 1
    w0 = torch.cat(w0)
    w0_pretrain, w0_clf = torch.cat(w0_pretrain), torch.cat(w0_clf)
    # print(w0)

    # p = nn.Parameter(torch.ones(len(w0), device=device) * torch.log(w0.abs().mean()), requires_grad=True)
    # OR
    # p = nn.Parameter(torch.ones(len(w0), device=device) * -np.log(10), requires_grad=True)
    '''
        try different p dimensions
    '''
    sorted_paramdim2mean = sorted(paramdim2mean.items(), key=lambda x: x[0])
    # print(sorted_paramdim2mean)
    # p_pretrain = nn.Parameter(torch.ones(len(w0_pretrain), device=device) * 0.1 * torch.log(w0_pretrain.abs().mean(())), requires_grad=True)
    # p_clf = nn.Parameter(torch.ones(len(w0_clf), device=device) * 0.1 * torch.log(w0_clf.abs().mean(())), requires_grad=True)
    p = nn.Parameter(torch.zeros(len(w0), device=device), requires_grad=True)
    p.data[:pretrain_dim] += 2 * torch.log(w0_pretrain.abs().mean())
    # print(p.data[:5],w0_pretrain.abs().mean(),w0.abs().mean())
    p.data[pretrain_dim:] += 1.0 * torch.log(w0_clf.abs().mean())
    # p.data += 2.0*torch.log(w0.abs().mean())
    '''
    dim_init = 0
    p_mean = []
    idx = 0
    for (dim, mean_value) in sorted_paramdim2mean:
        p.data[dim_init:dim_init + dim] += 2.0 * torch.log(mean_value)
        p_mean.append(2.0 * torch.log(mean_value))
        dim_init += dim
        #print(idx,float(2.0 * torch.log(mean_value)))
        idx += 1'''
    return w0, p, num_layer, pretrain_dim, clf_dim, p.data[0], p.data[-1]


def save_model(model, w0, p, epoch, prior, opt1, opt2, sch1,
               file_name, others=None, folder='logs/'):
    if os.path.isdir(folder) == False:
        try:
            os.makedirs(folder)
        except:
            pass
    if sch1 is None:
        torch.save({
            'epoch': epoch, 'w0': w0,
            'model_state_dict': model.state_dict(),
            'p': p, 'prior': prior,
            'opt1': opt1.state_dict(),
            'opt2': opt2.state_dict(),
            'others': others,
        }, folder + '/' + file_name + '.pt')
    else:
        torch.save({
            'epoch': epoch, 'w0': w0,
            'model_state_dict': model.state_dict(),
            'p': p, 'prior': prior,
            'opt1': opt1.state_dict(),
            'opt2': opt2.state_dict(),
            'others': others,
            'sch1': sch1.state_dict()
        }, folder + '/' + file_name + '.pt')


######################################################################
######################################################################
######################################################################

def func_sum(x, gamma, error_list, error_mean_list):
    def func(err, err_mu):
        out = np.zeros((len(gamma), 1))
        for r in range(len(gamma)):
            out[r] = -(np.mean(np.exp(np.longdouble(gamma[r] * (err_mu - err))))
                       - np.exp(np.longdouble(3 * (gamma[r]) ** 2 * (x ** 2) / 2)))
        return out

    sum_output = 0
    for i in range(len(error_mean_list)):
        sum_output += func(error_list[i], np.mean(error_mean_list))
    return sum_output


'''
def gen_output(model, prior, dataset, n, criterion):
    error_list = []
    error_mean_list = []

    device = next(model.parameters()).to(device)
    train = torch.utils.data.DataLoader(dataset.train.dataset, batch_size=1000)
    # compute the output of the random model and store it in an array
    with torch.no_grad():
        for i in range(n):
            model1 = copy.deepcopy(model)
            # generating a random model/network from the prior distribtuion
            for param in model1.parameters():
                param.data += torch.randn(param.data.size(), device=device) * prior

            errors = []
            for batch in train:
                inputs, targets = (b.to(device) for b in batch)
                predictions = model1(inputs)
                err = criterion(predictions, targets)
                errors.extend(list(err.cpu().numpy()))

            error_list.append(errors)
            error_mean_list.append(np.mean(errors))
    return error_list, error_mean_list
'''


def gen_output_transformer(args, model, tokenizer, prior, dataset, n):
    error_list = []
    error_mean_list = []

    # device = next(model.parameters()).to(args.device)
    # train = torch.utils.data.DataLoader(dataset.train.dataset, batch_size=1000)
    # compute the output of the random model and store it in an array
    with torch.no_grad():
        for i in range(n):
            model1 = copy.deepcopy(model)
            # generating a random model/network from the prior distribtuion
            for param in model1.parameters():
                if not param.requires_grad: continue
                param.data += torch.randn(param.data.size(), device=args.device) * prior

            errors = []
            train_dataset = bucket_dataset(args, tokenizer, args.train_data)
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=lambda x: x)
            for batch in train_dataloader:
                # inputs, targets = (b.to(args.device) for b in batch)
                _, _, _, loss = model1(batch)
                # print(loss)
                # err = criterion(predictions, targets)
                errors.extend(list(loss.cpu().numpy()))

            error_list.append(errors)
            error_mean_list.append(np.mean(errors))
    return error_list, error_mean_list


def compute_K_sample_transformer(args, model, tokenizer, dataset, min_gamma, max_gamma):
    def est_K(prior, x):
        # estimate k within a certain gamma range given prior
        gamma_grid = np.exp(np.linspace(np.log(min_gamma), np.log(max_gamma), 10))
        print('searching for K4....')
        error_list, error_mean_list = gen_output_transformer(args, model, tokenizer, prior, dataset, 10)
        while min(func_sum(x, gamma_grid, error_list, error_mean_list)) < -1e-20:
            x = x * 1.5
        while min(func_sum(x, gamma_grid, error_list, error_mean_list)) > 0:
            x = x / 1.1
        return x

    prior_list = np.exp(np.linspace(-6, -2, 8))
    K_list = [1e-3]
    for i in range(len(prior_list)):
        K_list.append(est_K(prior_list[i], K_list[-1]))
    K_list = K_list[1:]

    # make lists monotonically increasing
    ks, priors = [], []
    cur_max_k = 0
    for k, p in zip(K_list, prior_list):
        if k < cur_max_k:
            ks.append(cur_max_k)
            priors.append(p)
        else:
            ks.append(k)
            priors.append(p)
            cur_max_k = k

    return priors, ks


def fun_K_auto(x, exp_prior_list, K_list):
    n = len(exp_prior_list)
    y = K_list[0] + torch.relu(x - exp_prior_list[0]) * (K_list[1] - K_list[0]) / (
            exp_prior_list[1] - exp_prior_list[0])
    slope = (K_list[1] - K_list[0]) / (exp_prior_list[1] - exp_prior_list[0])
    for i in range(n - 2):
        slope = -slope + (K_list[i + 2] - K_list[i + 1]) / (exp_prior_list[i + 2] - exp_prior_list[i + 1])
        y += torch.relu(x - exp_prior_list[i + 1]) * slope
    return y


def fun_K_auto_new(x, exp_prior_list, K_list):
    n = len(exp_prior_list)
    i = 0
    print("x {}\t prior {}".format(x, exp_prior_list))
    while x > exp_prior_list[i]:
        i += 1
        if i == n - 1:
            break
    if i == 0:
        fa = K_list[0] + exp_prior_list[0]
        fb = K_list[0]
        a = 0
        b = exp_prior_list[0]
    else:
        fa = K_list[i - 1]
        fb = K_list[i]
        a = exp_prior_list[i - 1]
        b = exp_prior_list[i]
    return (b - x) / (b - a) * fa + (x - a) / (b - a) * fb


# def resume(model, file):
#     checkpoint = torch.load(file+'_s1.pt')
#     model.load_state_dict(checkpoint['model_state_dict'])
#     start_epoch = checkpoint['epoch']
#     others = checkpoint['others']
#     p = checkpoint['p']
#     w0 = checkpoint['w0']
#     prior = checkpoint['prior']
#     k = checkpoint['k']
#     return start_epoch, p, w0, prior, k, others

# def weight_decay_layer(model, w0):
#     # noise injection
#     k, weights = 0, []
#     for i, param in enumerate(model.parameters()):
#         t = len(param.view(-1))
#         weights.append(torch.norm(param.view(-1)-w0[k:(k+t)])**2)
#         k += t
#     return weights

# def weight_decay_layerp(model, w0):
#     k, weights = 0, 0
#     for i, param in enumerate(model.parameters()):
#         t = len(param.view(-1))
#         if torch.norm(w0[k:(k+t)])>1e-6:
#             alpha = (param.view(-1)*w0[k:(k+t)]).sum()/torch.norm(w0[k:(k+t)])**2
#         else:
#             alpha = 1
#         weights += torch.norm(param.view(-1)-alpha*w0[k:(k+t)])**2
#         k += t
#     return weights

# def get_kl_term_layer(K4, weights, p, model):
#     k, kl = 0, 0.0
#     for i, param in enumerate(model.parameters()):
#         t = len(param.view(-1))
#         denominator = weights[i] + torch.norm(torch.exp(p[k:(k+t)]))**2

#         we = torch.clip(t/denominator, max=1e3)
#         kl += -t*torch.log(we) -2*p[k:(k+t)].sum() -t +we*denominator

#         k += t
#     loss2 = K4*( 6*(0.5*kl+30) /5e4 )**0.5
#     return loss2, kl, 1/we**0.5

# def get_children(model: torch.nn.Module):
#     children = list(model.children())
#     flatt_children = []
#     if children == []:
#         # if model has no children; model is last child! :O
#         return model
#     else:
#        # look for children from children... to the last child!
#        for child in children:
#             try:
#                 flatt_children.extend(get_children(child))
#             except TypeError:
#                 flatt_children.append(get_children(child))
#     return flatt_children
class myAdam_old(Optimizer):
    r"""Implements Adam algorithm.
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, args, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        self.args = args
        super(myAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(myAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        factor = 20
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                '''
                #m_t
                bias_correction1 = 1 - beta1 ** state['step']
                #v_t
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                #print('exp_avg shape"{}\texp_avg_sq shape:{}'.format(exp_avg.shape,exp_avg_sq.shape))
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                # step_size = group['lr'] / bias_correction1

                # p.data.addcdiv_(-step_size, exp_avg, denom)

                pretrain_step_size = self.args.lr4clf / bias_correction1
                clf_step_size = self.args.lr4clf / bias_correction1
                #print(pretrain_step_size, clf_step_size)
                p.data[:self.args.pretrain_dim].addcdiv_(-1 * pretrain_step_size, exp_avg[:self.args.pretrain_dim],
                                                         denom[:self.args.pretrain_dim])
                p.data[self.args.pretrain_dim:].addcdiv_(-1 * clf_step_size, exp_avg[self.args.pretrain_dim:],
                                                         denom[self.args.pretrain_dim:])
                '''

                m_t = beta1 * state['exp_avg'] + (1 - beta1) * p.grad
                v_t = beta2 * state['exp_avg_sq'] + (1 - beta2) * torch.square(p.grad)

                state['exp_avg'], state['exp_avg_sq'] = m_t, v_t
                factor = factor * math.pow(0.9, int(state['step'] / 20))
                factor = max(factor, 1)
                m_t_hat = m_t / (1 - math.pow(beta1, state['step']))
                v_t_hat = v_t / (1 - math.pow(beta2, state['step']))

                p.data[:self.args.pretrain_dim] -= 0.5 * m_t_hat[:self.args.pretrain_dim] / (
                        torch.sqrt(v_t_hat[:self.args.pretrain_dim]) + group['eps'])
                p.data[self.args.pretrain_dim:] -= factor * self.args.lr4clf * m_t_hat[self.args.pretrain_dim:] / (
                        torch.sqrt(v_t_hat[self.args.pretrain_dim:]) + group['eps'])

        return loss


class myAdam(Optimizer):
    r"""Implements Adam algorithm.
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, args, pretrain_dims, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        self.args = args
        self.pretrain_dim = pretrain_dims
        super(myAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(myAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        factor = 50
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                '''
                #m_t 
                bias_correction1 = 1 - beta1 ** state['step']
                #v_t
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                #print('exp_avg shape"{}\texp_avg_sq shape:{}'.format(exp_avg.shape,exp_avg_sq.shape))
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                # step_size = group['lr'] / bias_correction1

                # p.data.addcdiv_(-step_size, exp_avg, denom)

                pretrain_step_size = self.args.lr4clf / bias_correction1
                clf_step_size = self.args.lr4clf / bias_correction1
                #print(pretrain_step_size, clf_step_size)
                p.data[:self.args.pretrain_dim].addcdiv_(-1 * pretrain_step_size, exp_avg[:self.args.pretrain_dim],
                                                         denom[:self.args.pretrain_dim])
                p.data[self.args.pretrain_dim:].addcdiv_(-1 * clf_step_size, exp_avg[self.args.pretrain_dim:],
                                                         denom[self.args.pretrain_dim:])
                '''

                m_t = beta1 * state['exp_avg'] + (1 - beta1) * p.grad
                v_t = beta2 * state['exp_avg_sq'] + (1 - beta2) * torch.square(p.grad)

                state['exp_avg'], state['exp_avg_sq'] = m_t, v_t
                factor = factor * math.pow(0.9, int(state['step'] / 10))
                factor = max(factor, 1)
                # print(factor*self.args.lr4clf)
                m_t_hat = m_t / (1 - math.pow(beta1, state['step']))
                v_t_hat = v_t / (1 - math.pow(beta2, state['step']))

                p.data[:self.pretrain_dim] -= 0.1 * m_t_hat[:self.pretrain_dim] / (
                        torch.sqrt(v_t_hat[:self.pretrain_dim]) + group['eps'])
                p.data[self.pretrain_dim:] -= factor * self.args.lr4clf * m_t_hat[self.pretrain_dim:] / (
                        torch.sqrt(v_t_hat[self.pretrain_dim:]) + group['eps'])

        return loss
