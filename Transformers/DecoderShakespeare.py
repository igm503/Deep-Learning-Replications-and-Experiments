from model import Model
from data_gen import *
from train import *
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

if __name__ == '__main__':
    # Hyperparameters
    d_model = 256
    layers = 4
    attention_heads = 8
    d_query = d_model // attention_heads
    batch_size = 32
    data_length = 15
    device = 'cpu'
    training_data_size = 1000000
    eval_size = 32
    vocab_size = 10
    smoothing = 0.001
    epochs = 1

    # Generate data

    training_data, eval_data, vocab_size, in_dict, out_dict = gen_text_nltk('shakespeare.txt', 100, 100, eval_size)
    #training_data, eval_data = generate_num_data(data_size, eval_size, vocab_size, data_length)
    #training_data, eval_data = gen_repeat(training_data_size, eval_size, vocab_size, data_length)
    
    max_len = training_data.shape[1]
    
    model = Model(layers, d_model, attention_heads, d_query, vocab_size, max_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.KLDivLoss(reduction='batchmean')
    lr_scheduler = LambdaLR(optimizer=optimizer, 
                            lr_lambda=lambda step: rate(step, d_model, factor=1, warmup=3000))

    # Train
    model.to(device)
    train(model, optimizer, criterion, lr_scheduler, batch_size, training_data, eval_data, epochs, device, smoothing, vocab_size, out_dict)
