from model import Model
from data_gen import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm


def rate(step, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        d_model ** (-0.5) * min(step ** (-0.5), step * 2 * warmup ** (-1.5))
    )

def train(model, optimizer, criterion, lr_scheduler, batch_size, training_data, eval_data, epochs, out_dict=None):
    for epoch in tqdm(range(epochs)):
        print('starting epoch', epoch)
        
        # Create batches and shuffle
        data_size = len(training_data)
        shuffler = torch.randperm(data_size)
        data = training_data[shuffler, :]
        data_list = [data[i * batch_size: (i + 1) * batch_size, :] for i in range(data_size // batch_size)]
        train_one_epoch(model, optimizer, criterion, lr_scheduler, data_list, eval_data, out_dict)
        print(len(data_list), 'batches')

def train_one_epoch(model, optimizer, criterion, lr_scheduler, data_list, eval_data, out_dict=None):
    running_loss = 0
    for i, data in tqdm(enumerate(data_list)):
        data = data.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(data[:, :-1])
        index = data[:, 1:].type(torch.long)
        target = nn.functional.one_hot(index, vocab_size)
        target = target.type(torch.float32) 
        # Label Smoothing Target
        target[target == 1] -= smoothing
        target = target + smoothing / vocab_size
        loss = criterion(outputs, target)
        # Gradient Descent
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        running_loss += loss.item()
        if i % 50 == 0:
            optimizer.zero_grad(set_to_none=True)
            model.eval()
            print(i, f'batches done | training loss: {running_loss: .2f}')
            lr = optimizer.param_groups[0]["lr"]
            print(f'learning rate: {lr: .2f}')
            running_loss = 0
            eval_loss = eval(model, criterion, eval_data, out_dict)
            print(f'Eval Loss: {eval_loss: .2f}')
            model.train()

def eval(model, criterion, data, out_dict=None):
    data = data.to(device)
    outputs = model(data[:, :-1])
    index = data[:, 1:].type(torch.long)
    target = nn.functional.one_hot(index, vocab_size)
    target = target.type(torch.float32) 
    # Label Smoothing Target
    target[target == 1] -= smoothing
    target = target + smoothing / vocab_size
    loss = criterion(outputs, target)
    
    # Show Example
    example_targ = index[0]
    example_pred = torch.argmax(outputs, -1)[0]
    if out_dict is not None:
        example_targ = [out_dict[num] for num in example_targ.tolist()]
        example_pred = [out_dict[num] for num in example_pred.tolist()]
    print(example_targ[:])
    print(example_pred[:])
    
    return loss.item()


if __name__ == '__main__':
    # Hyperparameters
    d_model = 256
    layers = 4
    attention_heads = 8
    d_query = d_model // attention_heads
    batch_size = 64
    data_length = 15
    device = 'mps'
    training_data_size = 1000000
    eval_size = 100
    vocab_size = 10
    smoothing = 0.001
    epochs = 1

    # Generate data

    #training_data, eval_data, vocab_size, in_dict, out_dict = generate_shakespeare(100, 20)
    #training_data, vocab_size, in_dict, out_dict = generate_bible(50, 20, 1000000)
    #training_data, eval_data = generate_num_data(data_size, eval_size, vocab_size, data_length)
    training_data, eval_data = gen_forwards_backwards(training_data_size, eval_size, vocab_size, data_length)
    
    max_len = training_data.shape[1]
    
    model = Model(layers, d_model, attention_heads, d_query, vocab_size, max_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.KLDivLoss(reduction='batchmean')
    lr_scheduler = LambdaLR(optimizer=optimizer, 
                            lr_lambda=lambda step: rate(step, factor=1, warmup=3000))

    # Train
    model.to(device)
    train(model, optimizer, criterion, lr_scheduler, batch_size, training_data, eval_data, epochs)

