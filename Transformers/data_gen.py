import torch
import re
from nltk.tokenize import word_tokenize


def generate_num_data(data_size, eval_size, vocab_size, data_length):
    '''
    Returns a data_size number of tensors of length data_length that begin with a random integer and 
    then "count" upwards. Also returns an eval_size number of tensors of the same form.

    Note: vocab_size must be larger than data_length

    Example: generate_num_data(2, 1, 5, 2) returns

    (train)
    [3, 4]
    [1, 2]

    (eval)
    [3, 4]
    [4, 5]

    The task is to learn that 2 follows 1, 3 follows 2, etc.
    '''
    training_data = torch.randint(0, vocab_size - data_length,(data_size, data_length)).type(torch.float32)
    eval_data = torch.randint(0, vocab_size - data_length ,(eval_size, data_length)).type(torch.float32)
    for i in range(1, training_data.shape[-1]):
        if i > 0:
            training_data[:, i] = training_data[:, i - 1] + 1
            eval_data[:, i] = eval_data[:, i - 1] + 1
    return training_data, eval_data

def gen_repeat(data_size, eval_size, vocab_size, data_length):
    '''
    Returns a data_size number of tensors of length 2 * data_length + 1 that begin with data_length//2 random
    integers between 0 and vocab_size, followed by 0, followed by the sequence before 0. Also returns 
    an eval_size number of tensors of the same sort, but with a unique starting sequence (1, 2, 3) not seen in 
    the training data.

    Note: data_length must be at least 3.

    Example: gen_repeat(2, 1, 5, 4) returns

    (train)
    [2, 4, 1, 1, 0, 2, 4, 1, 1]
    [1, 2, 4, 3, 0, 1, 2, 4, 3]

    (eval)
    [1, 2, 3, 1, 0, 1, 2, 3, 1]
    [1, 2, 3, 2, 0, 1, 2, 3, 1]

    The task is to learn that for i > data_length, the integer at position i is the integer at position i - data_length - 1
    '''
    training_data = torch.randint(1, vocab_size,(data_size, 2 * data_length + 1)).type(torch.float32)
    eval_data = torch.randint(1, vocab_size,(eval_size, 2 * data_length + 1)).type(torch.float32)
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
    # make integers in first half repeat in second half after putting 0 in the middle of the sequence
    training_data[:, data_length] = 0
    eval_data[:, data_length] = 0
    for i in range(data_length):
        training_data[:, data_length  + i + 1] = training_data[:, i]
        eval_data[:, data_length + i + 1] = eval_data[:, i]
    print(eval_data[0:5])
    
    return training_data, eval_data

def gen_text(filename, context_length, stride, eval_size):
    '''
    Returns tensors of length context_length in which each integer uniquely represents a word in 
    the filename text file. Also removes eval_size randomly selected tensors from this group and 
    returns these as evaluation data. Finally, it returns dictionaries, vocab_to_int and int_to_vocab
    that translate between from words to integers and vice versa. 
    '''
    with open(filename) as file:
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
    eval_data = training_data[-eval_size:, :]
    training_data = training_data[0:-eval_size, :]
    return training_data, eval_data, vocab_size, vocab_to_int, int_to_vocab

def gen_text_nltk(filename, context_length, stride, eval_size):
    '''
    Returns tensors of length context_length in which each integer uniquely represents a word in 
    the filename text file. Also removes eval_size randomly selected tensors from this group and 
    returns these as evaluation data. Finally, it returns dictionaries, vocab_to_int and int_to_vocab
    that translate between from words to integers and vice versa. 
    '''
    with open(filename) as file:
        vocab_to_int = {}
        int_to_vocab = {}
        index = 0
        text = file.read().lower()
        while '\n' in text:
            text = text.replace('\n', ' ')
        
        text_list =  word_tokenize(text)
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
    eval_data = training_data[-eval_size:, :]
    training_data = training_data[0:-eval_size, :]
    return training_data, eval_data, vocab_size, vocab_to_int, int_to_vocab

