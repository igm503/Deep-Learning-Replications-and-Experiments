from training_utils import run_test
import math
import pandas as pd


if __name__ == '__main__':
    filename = 'results.csv'
    df = pd.DataFrame({'model_size': [], 'lr': [], 'step': [],  'train_loss': [], 'eval_loss': []})
    log_type = 's'
    log_interval =100
    num_tests=1
    model_size=1
    base_rate=0.01
    batch_size = 128
    data_augment = 1
    device = 'cuda'

    model_size_list = [math.sqrt(2) ** i for i in range(1)]
    
    run_test(df, log_type='s', log_interval=100, test_values=model_size_list)

    df.to_csv(filename)