import itertools
import torch
import os
import copy
from datetime import datetime
import math
import numpy as np
import tqdm
from collections import defaultdict
from time import gmtime, strftime
import sys

import torch.nn.functional as F


def get_logging_print(fname):
    cur_time = strftime("%m-%d_%H:%M:%S", gmtime())

    def print_func(*args):
        str_to_write = ' '.join(map(str, args))
        filename = fname % cur_time if '%s' in fname else fname
        with open(filename, 'a') as f:
            f.write(str_to_write + '\n')
            f.flush()

        print(str_to_write)
        sys.stdout.flush()

    return print_func, fname % cur_time if '%s' in fname else fname


def flatten(lst):
    tmp = [i.contiguous().view(-1,1) for i in lst]
    return torch.cat(tmp).view(-1)


def unflatten_like(vector, likeTensorList):
    # Takes a flat torch.tensor and unflattens it to a list of torch.tensors
    #    shaped like likeTensorList
    outList = []
    i=0
    for tensor in likeTensorList:
        #n = module._parameters[name].numel()
        n = tensor.numel()
        outList.append(vector[:,i:i+n].view(tensor.shape))
        i+=n
    return outList


def LogSumExp(x,dim=0):
    m,_ = torch.max(x,dim=dim,keepdim=True)
    return m + torch.log((x - m).exp().sum(dim=dim,keepdim=True))


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(dir, epoch=None, name='checkpoint', **kwargs):
    state = {
        'epoch': epoch,
    }
    if epoch is not None:
       name = '%s-%d.pt' % (name, epoch)
    else:
       name = '%s.pt' % (name)
    state.update(kwargs)
    filepath = os.path.join(dir, name)
    torch.save(state, filepath)


def train_epoch(loader, model, criterion, optimizer, cuda=True, regression=False, verbose=False, subset=None):
    loss_sum = 0.0
    stats_sum = defaultdict(float)
    correct = 0.0
    verb_stage = 0

    num_objects_current = 0
    num_batches = len(loader)

    model.train()

    if subset is not None:
        num_batches = int(num_batches * subset)
        loader = itertools.islice(loader, num_batches)

    if verbose:
        loader = tqdm.tqdm(loader, total=num_batches)

    for i, (input, target) in enumerate(loader):
        if cuda:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        loss, output, stats = criterion(model, input, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        loss_sum += loss.data.item() * input.size(0)
        for key, value in stats.items():
            stats_sum[key] += value * input.size(0)

        if not regression:
            pred = output.data.argmax(1, keepdim=True)
            correct += pred.eq(target.data.view_as(pred)).sum().item()

        num_objects_current += input.size(0)

        if verbose and 10 * (i + 1) / num_batches >= verb_stage + 1:
            print('Stage %d/10. Loss: %12.4f. Acc: %6.2f' % (
                verb_stage + 1, loss_sum / num_objects_current,
                correct / num_objects_current * 100.0
            ))
            verb_stage += 1
    
    return {
        'loss': loss_sum / num_objects_current,
        'accuracy': None if regression else correct / num_objects_current * 100.0,
        'stats': {key: value / num_objects_current for key, value in stats_sum.items()}
    }


def eval(loader, model, criterion, cuda=True, regression=False, verbose=False):
    loss_sum = 0.0
    correct = 0.0
    stats_sum = defaultdict(float)
    num_objects_total = len(loader.dataset)

    model.eval()

    with torch.no_grad():
        if verbose:
            loader = tqdm.tqdm(loader)
        for i, (input, target) in enumerate(loader):
            if cuda:
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            loss, output, stats = criterion(model, input, target)

            loss_sum += loss.item() * input.size(0)
            for key, value in stats.items():
                stats_sum[key] += value

            if not regression:
                pred = output.data.argmax(1, keepdim=True)
                correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        'loss': loss_sum / num_objects_total,
        'accuracy': None if regression else correct / num_objects_total * 100.0,
        'stats': {key: value / num_objects_total for key, value in stats_sum.items()}
    }


def predict(loader, model, verbose=False):
    predictions = list()
    targets = list()

    model.eval()

    if verbose:
        loader = tqdm.tqdm(loader)

    offset = 0
    with torch.no_grad():
        for input, target in loader:
            input = input.cuda(non_blocking=True)
            output = model(input)

            batch_size = input.size(0)
            predictions.append(F.softmax(output, dim=1).cpu().numpy())
            targets.append(target.numpy())
            offset += batch_size

    return {
        'predictions': np.vstack(predictions),
        'targets': np.concatenate(targets)
    }


def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model, verbose=False, subset=None, **kwargs):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.

        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    num_batches = len(loader)

    with torch.no_grad():
        if subset is not None:
            num_batches = int(num_batches * subset)
            loader = itertools.islice(loader, num_batches)
        if verbose:
            loader = tqdm.tqdm(loader, total=num_batches)

        for input, _ in loader:
            input = input.cuda(non_blocking=True)
            input_var = torch.autograd.Variable(input)
            b = input_var.data.size(0)

            momentum = b / (n + b)
            for module in momenta.keys():
                module.momentum = momentum

            model(input_var, **kwargs)
            n += b

    model.apply(lambda module: _set_momenta(module, momenta))


def inv_softmax(x, eps = 1e-10):
    return torch.log(x/(1.0 - x + eps))


def predictions(test_loader, model, seed=None, cuda=True, regression=False, **kwargs):
    #will assume that model is already in eval mode
    #model.eval()
    preds = []
    targets = []
    for input, target in test_loader:
        if seed is not None:
            torch.manual_seed(seed)
        if cuda:
            input = input.cuda(non_blocking=True)
        output = model(input, **kwargs)
        if regression:
            preds.append(output.cpu().data.numpy())
        else:
            probs = F.softmax(output, dim=1)
            preds.append(probs.cpu().data.numpy())
        targets.append(target.numpy())
    return np.vstack(preds), np.concatenate(targets)


def schedule(epoch, lr_init, epochs, swa, swa_start=None, swa_lr=None):
    t = (epoch) / (swa_start if swa else epochs)
    lr_ratio = swa_lr / lr_init if swa else 0.01
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return lr_init * factor


def set_weights(model, vector, device=None):
    offset = 0
    for param in model.parameters():
        param.data.copy_(vector[offset:offset + param.numel()].view(param.size()).to(device))
        offset += param.numel()


def extract_parameters(model):
    params = []	
    for module in model.modules():	
        for name in list(module._parameters.keys()):	
            if module._parameters[name] is None:	
                continue	
            param = module._parameters[name]	
            params.append((module, name, param.size()))	
            module._parameters.pop(name)	
    return params


def set_weights_old(params, w, device):	
    offset = 0
    for module, name, shape in params:
        size = np.prod(shape)	       
        value = w[offset:offset + size]
        setattr(module, name, value.view(shape).to(device))	
        offset += size


def _zero_grad(model):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()


def _get_active_subspace(grads, n_dim=20, pct_var_explained=-1):
    grads = torch.stack(grads).to('cpu').numpy()
    from sklearn.decomposition import TruncatedSVD
    n_dim = min([n_dim, grads.shape[0], grads.shape[1]])
    svd = TruncatedSVD(n_components=n_dim, n_iter=7, random_state=123456)
    svd.fit(grads)

    if n_dim is None:
        if pct_var_explained <= 0 or pct_var_explained > 1:
            raise ValueError("Invalid value for pct_var_explained")
        else:
            raise NotImplementedError()
            # s = np.cumsum(svd.explained_variance_ratio_)
            # n_dim = (s <= pct_var_explained).sum()

    return torch.from_numpy(svd.components_[:n_dim])


# def get_active_subspace(func, sample_points, model, n_dim=20, pct_var_explained=-1):
#
#     grads = []
#     for i in range(len(sample_points)):
#         _zero_grad(model)
#         f = func(sample_points[i])
#         f.backward(retain_graph=True)
#         grads.append(flatten([p.grad for p in model.parameters()]))
#
#     return _get_active_subspace(grads, n_dim, pct_var_explained).to(sample_points[0].device)


# def functional_change_factory(model, sample_points):
#     def functional_change(params):
#         set_weights(model, params)
#         return model(sample_points).abs().mean()
#     return functional_change


# def get_alt_active_subspace1(sample_points, sample_parameters, model, n_dim=20, pct_var_explained=-1):
#     grads = []
#     for i in range(len(sample_parameters)):
#         _zero_grad(model)
#         set_weights(model, sample_parameters[i])
#         # grad_accum = torch.zeros_like(flatten([p.grad for p in model.parameters()]))
#         grad_accum = None
#         for j in range(len(sample_points)):
#             f = model(sample_points[j])
#             f.backward(retain_graph=True)
#             if grad_accum is None:
#                 grad_accum = flatten([p.grad for p in model.parameters()])
#             else:
#                 grad_accum += flatten([p.grad for p in model.parameters()])
#         grads.append(grad_accum)
#
#     return _get_active_subspace(grads, n_dim, pct_var_explained).to(sample_points[0].device)


def get_alt_active_subspace2(sample_points, sample_parameters, model, n_dim=20, pct_var_explained=-1):
    grads = []
    for i in range(len(sample_parameters)):
        _zero_grad(model)
        set_weights(model, sample_parameters[i])
        print('param num', i)
        farr = model(sample_points)[:, 0]
        for j in range(len(sample_points)):
            # f = model(sample_points[j:j+1])[0,0]
            f = farr[j]
            f.backward(retain_graph=True)
            grads.append(flatten([p.grad for p in model.parameters()]).to('cpu'))

    return _get_active_subspace(grads, n_dim, pct_var_explained).to(sample_points[0].device)


def online_get_active_subspace(sample_points, sample_parameters, model, n_dim=20, init_step_size=5., n_iters=1000,
                               anneal=False):
    """Approximate the linear map to a low-dimensional latent space with online gradient descent."""
    def sample_grad():
        while True:
            _zero_grad(model)
            i = np.random.randint(0, len(sample_points))
            j = np.random.randint(0, len(sample_parameters))
            set_weights(model, sample_parameters[j])
            f = model(sample_points[i:i+1])[0,0]
            f.backward(retain_graph=True)
            yield flatten([p.grad for p in model.parameters()])

    online_svd(sample_grad(), len(sample_parameters[0]), n_dim, init_step_size, iters=n_iters, anneal=anneal)







class LowDimMap(torch.nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.d = d
        self.k = k
        W = torch.randn(d, k)
        W = W / torch.norm(W, dim=0)
        self.register_parameter('W',
                                torch.nn.Parameter(W, requires_grad=True))

    def forward(self, x):
        return x.matmul(self.W).matmul(self.W.transpose(0, 1))


def online_svd(sampler_oracle, d, k, init_step_size, iters, anneal=False):
    """

    :param sampler_oracle: an iterable/generator that generates data samples one at a time.
    :param d: dimensionality of the space
    :param k: rank of the low-rank approximator
    :param init_step_size: initial step size/learning rate size.
    :param iters: number of iterations to continue for.
    :return: low-rank approximation W
    """

    ldm = LowDimMap(d, k)
    optim = torch.optim.SGD(ldm.parameters(), lr=init_step_size)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda t: 1/math.sqrt(1+t))
    scores = []
    for iter in range(iters):
        optim.zero_grad()
        data = sampler_oracle.send(None)
        pred = ldm(data)
#         print(data - pred)
        loss = (data - pred).pow_(2).mean() #+ subspace.W.pow(2).sum()
#         print(data.detach().numpy(), pred.detach().numpy(), loss.item())
        print(loss.item())
        loss.backward()
        optim.step()
        if anneal:
            sched.step()
        scores.append(loss.item())

    return ldm.W.data, scores
