import torch
from torch import nn
from tqdm import tqdm
from critical_batch import get_batch_noise, get_batch_noise_alt


def rate(step, d_model, factor, warmup):
    if step == 0:
        step = 1
    return factor * (
        d_model ** (-0.5) * min(step ** (-0.5), step * 2 * warmup ** (-1.5))
    )

def train(model, optimizer, criterion, lr_scheduler, batch_size, training_data, eval_data, epochs, device, smoothing, vocab_size, rec_batch_noise=False, out_dict=None):
    for epoch in tqdm(range(epochs)):
        print('starting epoch', epoch)
        
        # Create batches and shuffle
        data_size = len(training_data)
        shuffler = torch.randperm(data_size)
        data = training_data[shuffler, :]
        data_list = [data[i * batch_size: (i + 1) * batch_size, :] for i in range(data_size // batch_size)]
        train_one_epoch(model, optimizer, criterion, lr_scheduler, data_list, eval_data, device, smoothing, vocab_size, rec_batch_noise, out_dict)
        print(len(data_list), 'batches')

def train_one_epoch(model, optimizer, criterion, lr_scheduler, data_list, eval_data, device, smoothing, vocab_size, rec_batch_noise=False, out_dict=None):
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

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        running_loss += loss.item()
        if i % 50 == 0:
            optimizer.zero_grad(set_to_none=True)
            model.eval()
            lr = optimizer.param_groups[0]["lr"]
            eval_loss = eval(model, criterion, eval_data, smoothing, device, vocab_size, out_dict)
            if rec_batch_noise:
                batch_noise = get_batch_noise(model, data_list, criterion, rec_batch_noise, device, vocab_size, smoothing)
                print(f'Batch Noise Estimate: {batch_noise}')
                alt_batch_noise = get_batch_noise_alt(model, data_list, criterion, rec_batch_noise, device, vocab_size, smoothing)
                print(f'Alt Batch Noise Estimate: {alt_batch_noise}')
            print(i, f'batches done | training loss: {running_loss / 50: .2f}')
            print(f'learning rate: {lr}')
            print(f'Eval Loss: {eval_loss: .2f}')
            running_loss = 0
            model.train()

def eval(model, criterion, data, smoothing, device, vocab_size, out_dict=None):
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