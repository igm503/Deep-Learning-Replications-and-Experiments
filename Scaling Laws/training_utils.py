import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import math
from data_loaders import MNIST_dataloaders, EMNIST_dataloaders


def evaluate(testloader, net, criterion, device):
    with torch.no_grad():
        running_loss = 0
        for batch_num, data in enumerate(testloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    return running_loss / (batch_num + 1)


def log(net, model_size, batch_size, num_params, device, test_loader, rate, criterion, train_loss, df, step, filename):
    net.eval()
    eval_loss = evaluate(test_loader, net, criterion, device)
    df.loc[len(df.index)] = [model_size, rate, step *
                             batch_size, train_loss, eval_loss, num_params]
    df.to_csv(filename)
    net.train()
    return eval_loss


def get_log_steps(data_set, batch_size, num_logs, data_augment):
    if data_set == 'MNIST':
        n = 60000
    else:
        n = 240000
    n *= data_augment
    num_steps = n // batch_size
    log_max = np.log10(num_steps - 10)
    log_steps = np.logspace(2, log_max, num_logs)
    return list(log_steps.astype(int))


def keep_least(df, group_var, min_var):
    minimum = 1e9
    min_list = []
    group = df[group_var][0]
    for i in range(len(df)):
        if math.isclose(group, df[group_var][i]):
            minimum = min(minimum, df[min_var][i])
        else:
            minimum = df[min_var][i]
            group = df[group_var][i]
        min_list.append(minimum)
    return min_list


def add_vars(df, batch_size, log_type):
    if log_type == 'lr':
        group_var = 'lr'
    else:
        group_var = 'model_size'
    df['eval_min'] = keep_least(df, group_var, 'eval_loss')
    df['train_min'] = keep_least(df, group_var, 'train_loss')
    df['compute'] = (df['params']) * df['step'] * batch_size


def train_one_epoch(net, model_size, batch_size, num_params, device, train_loader, test_loader, rate, optimizer, criterion, df, scheduler, log_steps, filename):
    for step, data in tqdm(enumerate(train_loader)):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())
        if step in log_steps:
            print(step, 'logging')
            train_loss = loss.item()
            eval_loss = log(net, model_size, batch_size, num_params, device, test_loader,
                            rate, criterion, train_loss, df, step, filename)


def train(model_type, model_size, batch_size, device, data_set, data_augment, base_rate, df, log_steps, filename):
    if data_set == 'EMNIST':
        train_loader, test_loader = EMNIST_dataloaders(
            data_augment, batch_size)
    else:
        train_loader, test_loader = MNIST_dataloaders(data_augment, batch_size)
    net = model_type(model_size)
    net = net.to(device)
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    criterion = nn.CrossEntropyLoss()
    rate = base_rate / np.sqrt(model_size)
    optimizer = optim.Adam(net.parameters(), lr=rate,
                           betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=60000//(batch_size), verbose=True)
    train_one_epoch(net, model_size, batch_size, num_params, device, train_loader,
                    test_loader, rate, optimizer, criterion, df, scheduler, log_steps, filename)


def run_test(df, log_type, num_logs, test_values, num_tests, model_type, filename, model_size=1, batch_size=128, base_rate=0.01, data_set='MNIST', data_augment=1, device='mps'):
    log_steps = get_log_steps(data_set, batch_size, num_logs, data_augment)
    for i in range(num_tests):
        if log_type == 'lr':
            for lr in tqdm(test_values):
                print(f'Beginning Test {i} for learning rate {lr}')
                train(model_type, model_size, batch_size, device, data_set,
                      data_augment, lr, df, log_steps, filename)
        elif log_type == 's':
            for size in tqdm(test_values):
                print(f'Beginning Test {i} for model size {size}')
                train(model_type, size, batch_size, device, data_set, data_augment,
                      base_rate, df, log_steps, filename)
        else:
            print('Invalid log_type. Must be \'s\' or \'lr\'')
    add_vars(df, batch_size, log_type)
