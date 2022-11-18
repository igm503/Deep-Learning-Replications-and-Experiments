from training_utils import run_test
import math
import pandas as pd
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                    prog = 'MNIST Scaling Test',
                    description = 'Runs specificiable number of experiments to see\
                                 how different model sizes or learning rates affect\
                                 model performance given different amounts of training\
                                 data. log_type can be set to \'s\' to measure the\
                                 effect of changing model size and training length\
                                 or to \'lr\' to measure the effect of changing learning\
                                 rate')
    parser.add_argument('test_values', metavar='N', type=int, nargs='*')
    parser.add_argument('--log_type', metavar='string', dest='log_type', default='s')
    parser.add_argument('--interval', metavar='integer', dest='log_interval', type=int, default=100)
    parser.add_argument('--trials', metavar='integer', dest='num_tests', type=int, default=1)
    parser.add_argument('--size', metavar='integer', dest='model_size', type=int, default=1)
    parser.add_argument('--batch', metavar='integer', dest='batch_size', type=int, default=128)
    parser.add_argument('--lr', metavar='float', dest='base_rate', type=float, default=0.01)
    parser.add_argument('--data', metavar='integer', dest='data_augment', type=int, default=1)
    parser.add_argument('--device', dest='device', type=str, default='cpu')
    parser.add_argument('--filename', dest='filename', type=str, default='results.csv')
    args = parser.parse_args()

    if args.test_values == []:
        if args.log_type == 's':
            args.test_values = [math.sqrt(2) ** i for i in range(9)]
        if args.log_type == 'lr':
            args.test_values = [.1, .05, .01, .005, .001]

    print('Beginning Training')
    print()
    print('Test Settings')
    print('------------------')
    print('testing values:', [round(value, 2) for value in args.test_values])
    print('log_type:', args.log_type)
    print('logging interval:', args.log_interval)
    print('num trials:', args.num_tests)
    print('output filename:', args.filename)
    print()
    print('Hyperparameters')
    print('------------------')
    print('batch size:', args.batch_size)
    print('data multiplier:', args.data_augment)
    print('device:', args.device)
    if args.log_type == 's':
        print('base learning rate:', args.base_rate)
    elif args.log_type == 'lr':
        print('model size:', args.model_size)
    print()

    df = pd.DataFrame({'model_size': [], 'lr': [], 'step': [],  'train_loss': [], 'eval_loss': []})

    run_test(df, args.log_type, args.log_interval, args.test_values, args.num_tests, args.model_size, 
            args.batch_size, args.base_rate, args.data_augment, args.device)

    df.to_csv(args.filename)