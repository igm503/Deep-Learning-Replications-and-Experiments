import math
import torch
import torch.nn as nn


class Embedder(nn.Module):
    "Takes tokens from sequence and embeds them as vectors of size d_model"
    def __init__(self, vocab_size, d_model, max_len):
        super().__init__()
        self.vocab_size = vocab_size
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
        text = nn.functional.one_hot(text.type(torch.long), self.vocab_size).type(torch.float32)
        text = self.linear(text)
        text = text + self.pe[:, : text.size(-2)].requires_grad_(False) # Positional Encoding
        return self.dropout(text)


class Decoder(nn.Module):
    '''Takes embedded vector from Embedder and translates it to embedded vector
    for the next token in the sequence'''
    def __init__(self, layers, d_model, attention_heads, d_query):
        super().__init__()
        self.layers = layers
        self.self_attent = nn.ModuleList([SelfAttent(d_model, attention_heads, d_query) for _ in range(self.layers)])
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
            out = self.norm[2 * i](self.dropout(out)) # Regularization
            out += residual # First residual
            residual = out 
            out = self.linear2[i](self.relu(self.linear1[i](out))) # FFN Layer
            out = self.norm[2 * i + 1](self.dropout(out)) # Regularization
            out += residual # Second residual
        return self.norm[-1](out)

class Generator(nn.Module):
    "Takes output vector from decoder module and translates it to vocab vector"
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, embedded_out):
        out = self.linear(embedded_out)
        out = nn.functional.relu(out)
        return nn.functional.log_softmax(out, dim=-1)

class SelfAttent(nn.Module):
    def __init__(self, d_model, attention_heads, d_query):
        super().__init__()
        self.attention_heads = attention_heads
        self.d_query = d_query
        self.linear = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])

    def forward(self, embedding, mask):
        batch_size = embedding.shape[0]
        query = embedding
        key = embedding
        value = embedding
        query, key, value = [linear(x).view(batch_size, -1, self.attention_heads, self.d_query).transpose(1, 2) 
                             for linear, x in zip(self.linear, (query, key, value))]
        x = dot_product_attent(query, key, value, mask)
        del query
        del key
        del value
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.attention_heads * self.d_query)
        return self.linear[-1](x)

def dot_product_attent(query, key, value, mask):
    scores = torch.matmul(query, key.transpose(-2, -1)) 
    scores /= math.sqrt(query.size(-1)) # Norm by size of query
    mask = mask.unsqueeze(1)
    scores = scores.masked_fill(mask==0, -1e9)
    scores = nn.functional.softmax(scores, dim=-1)
    return torch.matmul(scores, value)

def make_std_mask(text, pad=-1):
    "Creates a mask to hide padding and future words."
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
    def __init__(self, layers, d_model, attention_heads, d_query, vocab_size, max_len):
        super().__init__()
        self.embed = Embedder(vocab_size, d_model, max_len)
        self.decode = Decoder(layers, d_model, attention_heads, d_query)
        self.generate = Generator(d_model, vocab_size)
    
    def forward(self, text):
        embedding = self.embed(text)
        mask = make_std_mask(text)
        embedded_out = self.decode(embedding, mask)
        text_out = self.generate(embedded_out)
        return text_out