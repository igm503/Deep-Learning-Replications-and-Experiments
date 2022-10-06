import math
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import re

class Embedder(nn.Module):
    "Takes tokens from sequence and embeds them as vectors of size d_model"
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(vocab_size, d_model)
        self.dropout = nn.Dropout()
        # For positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe) # Don't understand this

    def forward(self, text):
        #text = text.unsqueeze(-1)
        text = nn.functional.one_hot(text.type(torch.long), vocab_size).type(torch.float32)
        text = self.linear(text)
        text = text + self.pe[:, : text.size(-2)].requires_grad_(False) # Positional Encoding
        return self.dropout(text)


class Decoder(nn.Module):
    '''Takes embedded vector from Embedder and translates it to embedded vector
    for the next token in the sequence'''
    def __init__(self):
        super().__init__()
        self.layers = layers
        self.self_attent = nn.ModuleList([SelfAttent() for _ in range(self.layers)])
        self.linear1 = nn.ModuleList([nn.Linear(d_model, 4 * d_model) for _ in range(self.layers)])
        self.linear2 = nn.ModuleList([nn.Linear(4 * d_model, d_model) for _ in range(self.layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2 * self.layers + 1)]) # Has final Norm Layer for Decoder Output
        self.dropout = nn.Dropout(p=0.1)
        self.relu = nn.functional.relu

    def forward(self, embedding, mask):
        out = embedding
        for i in range(self.layers):
            residual = out
            out = self.self_attent[i](embedding, mask) # Attention Layer
            out = self.norm[2 * i](self.dropout(out).to(device)) # Regularization
            out += residual # First residual
            residual = out 
            out = self.linear2[i](self.relu(self.linear1[i](out))) # FFN Layer
            out = self.norm[2 * i + 1](self.dropout(out)) # Regularization
            out += residual # Second residual
        return self.norm[-1](out)

class Generator(nn.Module):
    "Takes output vector from decoder module and translates it to vocab vector"
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, embedded_out):
        out = self.linear(embedded_out)
        out = nn.functional.relu(out)
        return nn.functional.log_softmax(out, dim=-1)

class SelfAttent(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])

    def forward(self, embedding, mask):
        query = embedding
        key = embedding
        value = embedding
        query, key, value = [linear(x).view(batch_size, -1, attention_heads, d_query).transpose(1, 2) 
                             for linear, x in zip(self.linear, (query, key, value))]
        x = dot_product_attent(query, key, value, mask)
        del query
        del key
        del value
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, attention_heads * d_query)
        return self.linear[-1](x)

def dot_product_attent(query, key, value, mask):
    scores = torch.matmul(query, key.transpose(-2, -1)) 
    scores /= math.sqrt(query.size(-1)) # Norm by size of query
    mask = mask.unsqueeze(1)
    scores = scores.masked_fill(mask==0, -1e9)
    scores = nn.functional.softmax(scores, dim=-1)
    return torch.matmul(scores, value)

def make_std_mask(text, pad=-1):
    "Create a mask to hide padding and future words."
    size = text.size(-1)
    tgt_mask = (text != pad)
    tgt_mask = (text != pad).unsqueeze(-2)
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type_as(tgt_mask.data)
    subsequent_mask = subsequent_mask == 0
    tgt_mask = tgt_mask & subsequent_mask
    return tgt_mask
        
class Model(nn.Module):
    '''Puts all the components of the model together'''
    def __init__(self):
        super().__init__()
        self.embed = Embedder()
        self.decode = Decoder()
        self.generate = Generator()
    
    def forward(self, text):
        embedding = self.embed(text)
        mask = make_std_mask(text)
        embedded_out = self.decode(embedding, mask)
        text_out = self.generate(embedded_out)
        return text_out

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

def generate_num_data(data_size, eval_size, vocab_size, data_length):
    training_data = torch.randint(0, vocab_size // 60,(data_size, data_length)).type(torch.float32)
    eval_data = torch.randint(0, vocab_size // 50 ,(eval_size, data_length)).type(torch.float32)
    for i in range(1, training_data.shape[-1]):
        if i > 1:
            training_data[:, i] = training_data[:, i - 1] + training_data[:, i-2]
            eval_data[:, i] = eval_data[:, i - 1] + eval_data[:, i-2]
        else:
            training_data[:, i] = training_data[:, i - 1] 
            eval_data[:, i] = eval_data[:, i - 1]

    return training_data, eval_data

def gen_forwards_backwards(data_size, eval_size, vocab_size, data_length):
    training_data = torch.randint(0, vocab_size,(data_size, 2 * data_length)).type(torch.float32)
    eval_data = torch.randint(0, vocab_size,(eval_size, 2 * data_length)).type(torch.float32)
    training_data[:, 0] = 0
    eval_data[:, 0] = 0
    for i in range(data_length):
        if i == 0:
            training_data[:, data_length] = 0
            eval_data[:, data_length] = 0
        else:
            training_data[:, data_length + i] = training_data[:, i]
            eval_data[:, data_length + i] = eval_data[:, i]
    return training_data, eval_data

def generate_shakespeare(context_length, stride):
    with open('shakespeare.txt') as file:
        vocab_to_int = {}
        int_to_vocab = {}
        index = 0
        text = file.read().lower()
        while '\n' in text:
            text = text.replace('\n', '')
        text_list = re.split(r"\b", text)
        text_list = [word for word in text_list if word not in {' '}]
        for word in text_list:
            if word not in vocab_to_int:
                vocab_to_int[word] = index
                int_to_vocab[index] = word
                index += 1
        print('vocab size:', index)
    vocab_size = index
    text_as_int = [vocab_to_int[word] for word in text_list]
    training_data = [text_as_int[stride * i: stride * i + context_length] for i in range(400)]
    training_data = torch.tensor(training_data)
    return training_data, vocab_size, vocab_to_int, int_to_vocab

def generate_bible(context_length, stride, size):
    with open('all.txt') as file:
        vocab_to_int = {}
        int_to_vocab = {}
        index = 0
        text = file.read(99999999).lower()
        while '\n' in text:
            text = text.replace('\n', ' ')
        text_list = re.split(r"\b", text)
        text_list = [word for word in text_list if word not in {' '}]
        for word in text_list:
            if word not in vocab_to_int:
                vocab_to_int[word] = index
                int_to_vocab[index] = word
                index += 1
        print('vocab size:', index)
    vocab_size = index
    text_as_int = [vocab_to_int[word] for word in text_list]
    training_data = [text_as_int[stride * i: stride * i + context_length] for i in range(size)]
    training_data = torch.tensor(training_data)
    return training_data, vocab_size, vocab_to_int, int_to_vocab

def evaluate_text(model):
    train_batch_size = batch_size
    batch_size = 1   
    '''input = torch.tensor([[1, 2, 3, 4, 5, 5, 4, 3, 2, ]]).to(device)
    
    print(torch.argmax(model(input), dim=-1))'''
    #Random example
    input_text = ['1', ':', '1', 'in', 'the', 'beginning',  'moses',  'looked', 'upon']
    input_ints = [in_dict[word] for word in input_text]
    for i in range(30):
        input = torch.tensor(input_ints).unsqueeze(0).to(device)
        output = torch.argmax(model(input),dim=-1).type(torch.long)
        input_ints.append(int(output[0, -1]))
    print([out_dict[word] for word in input_ints])
    #Training example
    input_text = ['19', ':', '16', 'and', 'when', 'the', 'syrians', 'saw', 'that', 'they']
    input_ints = [in_dict[word] for word in input_text]
    for i in range(30):
        input = torch.tensor(input_ints).unsqueeze(0).to(device)
        output = torch.argmax(model(input),dim=-1).type(torch.long)
        input_ints.append(int(output[0, -1]))
    print([out_dict[word] for word in input_ints])
    batch_size = train_batch_size
    

    

if __name__ == '__main__':
    # Hyperparameters
    d_model = 512
    layers = 6
    attention_heads = 8
    d_query = d_model // attention_heads
    batch_size = 20
    max_len = 500
    data_length = 10
    device = 'cpu'
    data_size = 100000
    eval_size = 100
    vocab_size = 10
    smoothing = 0.001


    # Generate data

    #training_data, vocab_size, in_dict, out_dict = generate_shakespeare(100, 20)
    #training_data, vocab_size, in_dict, out_dict = generate_bible(50, 20, 1000000)
    #training_data, eval_data = generate_num_data(data_size, eval_size, vocab_size, data_length)
    training_data, eval_data = gen_forwards_backwards(data_size, eval_size, vocab_size, data_length)
    
    
    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=.01, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.KLDivLoss(reduction='sum')
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, factor=.001, warmup=3000
        ),
    )

    # Train
    model.to(device)
    trainable_params = sum(
	p.numel() for p in model.parameters())
    print('Trainable Params:', trainable_params)
    print('vocab size:', vocab_size)
    for epoch in tqdm(range(500)):
        running_loss = 0.0
        print('starting epoch')
        model.train()
        # Create batches and shuffle
        shuffler = torch.randperm(data_size)
        data = training_data[shuffler, :].to(device)
        data_list = [data[i:i + batch_size, :] for i in range(data_size // batch_size)]
        
        for i, data in tqdm(enumerate(data_list)):
            optimizer.zero_grad(set_to_none=True)
            outputs = model(data[:, :-1])
            outputs = outputs[:, data_length - 1:]
            # Make target
            index = data[:, data_length:].type(torch.long)
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
            # Eval statistics
            running_loss += loss.item()
            if i > 100:
                break
       
        #evaluate_text(model)
        
        
        
        print('epoch finished, running loss:', running_loss)
        lr = optimizer.param_groups[0]["lr"]
        print('learning rate:', lr)

        # Eval 
        eval_loss = 0
        training_batch = batch_size
        batch_size = eval_size
        eval_input = eval_data[:, 0:-1]
        eval_out = model(eval_input)
        index = eval_data[:, data_length:].type(torch.long)
        target = nn.functional.one_hot(index, vocab_size)
        target = target.type(torch.float32) 
        eval_out = eval_out[:, data_length - 1:]
        print(eval_out.shape)
        print(target.shape)
        loss = criterion(eval_out, target)
        print('Eval Loss:', loss.item())
        # Example
        batch_size = 1
        example = torch.tensor([[0, 1, 2, 3, 4, 1, 2, 3, 4, 1, 0, 1, 2, 3, 4, 1, 2, 3, 4, 1]])
        ex_out = model(example).argmax(dim=-1)
        print('example input:', example)
        print('example output:', ex_out)


        batch_size = training_batch