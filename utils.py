import torch
import torch.nn as nn

import math, os, copy
import numpy as np
from scipy.optimize import bisect

import warnings

from torch.utils.data import DataLoader
import sys
from torch.optim import Optimizer
from myDataLoader import bucket_dataset

warnings.filterwarnings("ignore")


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

    p = nn.Parameter(torch.zeros(len(w0), device=device), requires_grad=True)
    p.data[:pretrain_dim] += 2 * torch.log(w0_pretrain.abs().mean())
    p.data[pretrain_dim:] += 1.0 * torch.log(w0_clf.abs().mean())

    return w0, p, num_layer, pretrain_dim, clf_dim, p.data[0], p.data[-1]


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


class myAdam(Optimizer):

    def __init__(self, args, pretrain_dims, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
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
        loss = None
        if closure is not None:
            loss = closure()
        factor = 50
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

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
