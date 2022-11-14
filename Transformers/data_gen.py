import torch
import re

def generate_num_data(data_size, eval_size, vocab_size, data_length):
    training_data = torch.randint(0, vocab_size - data_length,(data_size, data_length)).type(torch.float32)
    eval_data = torch.randint(0, vocab_size - data_length ,(eval_size, data_length)).type(torch.float32)
    for i in range(1, training_data.shape[-1]):
        if i > 0:
            training_data[:, i] = training_data[:, i - 1] + 1
            eval_data[:, i] = eval_data[:, i - 1] + 1
    return training_data, eval_data

def gen_forwards_backwards(data_size, eval_size, vocab_size, data_length):
    training_data = torch.randint(1, vocab_size,(data_size, 2 * data_length)).type(torch.float32)
    eval_data = torch.randint(1, vocab_size,(eval_size, 2 * data_length)).type(torch.float32)
    training_data[:, 0] = 0
    eval_data[:, 0] = 0
    # remove pattern from training data and adding to eval data
    for i in range(data_size):
        if training_data[i, 0] == 1 and training_data[i, 1] == 2 and training_data[i, 2] == 3:
            training_data[i, 0] = 3
            training_data[i, 1] = 2
            training_data[i, 2] = 1
    for i in range(eval_size):
        eval_data[i, 0] = 1
        eval_data[i, 1] = 2
        eval_data[i, 2] = 3
    # make digits in first half repeat in second half
    for i in range(data_length):
        if i == 0:
            training_data[:, data_length] = 0
            eval_data[:, data_length] = 0
        else:
            training_data[:, data_length  + i] = training_data[:, i]
            eval_data[:, data_length + i] = eval_data[:, i]
    print(eval_data[0:5])
    
    return training_data, eval_data

def generate_shakespeare(context_length, stride):
    with open('shakespeare.txt') as file:
        vocab_to_int = {}
        int_to_vocab = {}
        index = 0
        text = file.read().lower()
        while '\n' in text:
            text = text.replace('\n', ' ')
        text_list = re.split(r"\b", text)
        text_list = [word for word in text_list if word not in '      ']
        for word in text_list:
            if word not in vocab_to_int:
                vocab_to_int[word] = index
                int_to_vocab[index] = word
                index += 1
    vocab_size = index
    total_tokens = len(text_list)
    num_contexts = (total_tokens - context_length) // stride
    print('vocab size:', vocab_size, '| total tokens:', total_tokens, '| total contexts:', num_contexts)
    text_as_int = [vocab_to_int[word] for word in text_list]
    training_data = [text_as_int[stride * i: stride * i + context_length] for i in range(num_contexts)]
    training_data = torch.tensor(training_data)
    rand_idx = torch.randperm(len(training_data))
    training_data = training_data[rand_idx]
    print('shape of training_data', training_data.shape)
    eval_data = training_data[-10:, :]
    training_data = training_data[0:-10, :]
    print('shape of training_data', training_data.shape)
    return training_data, eval_data, vocab_size, vocab_to_int, int_to_vocab

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