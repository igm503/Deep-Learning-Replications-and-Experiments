
from model import Model
from data_gen import *
from train import *
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

def get_grad(model, data, criterion, vocab_size, smoothing):
    model.zero_grad(set_to_none=True)
    outputs = model(data[:, :-1])
    index = data[:, 1:].type(torch.long)
    target = nn.functional.one_hot(index, vocab_size)
    target = target.type(torch.float32)

    # Label Smoothing Target
    target[target == 1] -= smoothing
    target = target + smoothing / vocab_size
    loss = criterion(outputs, target)

    loss.backward()

    grad = []
    i = 0
    for param in model.parameters():
        grad.append(param.grad.flatten())

    grad = torch.cat(grad)
    return torch.Tensor(grad).reshape(-1, 1)

def get_grads(model, data_list, criterion, n, device, vocab_size, smoothing):
    grads_list = []
    num_batches = len(data_list)
    for i in range(n):
        data_index = int(torch.rand(1) * num_batches)
        data = data_list[data_index].to(device)
        grad = get_grad(model, data, criterion, vocab_size, smoothing)
        grads_list.append(grad)
    return grads_list

def grad_var(model, data_list, criterion, n, device, vocab_size, smoothing):
    grads_list = get_grads(model, data_list, criterion, n, device, vocab_size, smoothing)
    gradients = torch.cat(grads_list, dim=1)
    var = torch.var(gradients, dim=1)
    tr_sigma = var.sum().item()
    grad_estimate = gradients.mean(dim=1)
    grad_squared = (grad_estimate ** 2).sum(dim=0).item()
    return tr_sigma, grad_squared


def get_batch_noise(model, data_list, criterion, n, device, vocab_size, smoothing):
    tr_sigma, grad_squared = grad_var(model, data_list, criterion, n, device, vocab_size, smoothing)
    batch_tokens = torch.numel(data_list[0])
    print(batch_tokens)
    return batch_tokens * tr_sigma / (grad_squared + 1e-6) # avoid division by 0


def grad_var_alt(model, data_list, criterion, n, device, vocab_size, smoothing):
    grads_list = get_grads(model, data_list, criterion, n, device, vocab_size, smoothing)
    gradients = torch.cat(grads_list, dim=1)
    grad_estimate = torch.mean(gradients, 1, True)
    zeroed_gradients = gradients - grad_estimate
    expected_diff_squared = (zeroed_gradients ** 2).sum(dim=0).mean().item()
    grad_squared = (grad_estimate ** 2).sum(dim=0).item()
    return expected_diff_squared, grad_squared

def get_batch_noise_alt(model, data_list, criterion, n, device, vocab_size, smoothing):
    expected_diff_squared, grad_squared = grad_var_alt(model, data_list, criterion, n, device, vocab_size, smoothing)
    batch_tokens = torch.numel(data_list[0])
    print(batch_tokens)
    return batch_tokens * expected_diff_squared/ (grad_squared + 1e-6) # avoid division by 0
