from training_utils import run_test
import matplotlib.pyplot as plt

import math
import pandas as pd

import seaborn as sns


if __name__ == '__main__':
    scaling_df = pd.DataFrame({'model_size': [], 'lr': [], 'step': [],  'train_loss': [], 'eval_loss': []})
    
    batch_size = 128
    data_augment = 1
    device = 'cuda'

    model_size_list = [math.sqrt(2) ** i for i in range(1)]
    
    run_test(scaling_df, log_type='s', log_interval=100, test_values=model_size_list)

    scaling_df.to_csv('scaling_data.csv')

    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(16, 12))
    ax = ax.flatten()

    palette = sns.color_palette('tab10')

    sns.lineplot(data=scaling_df, x = 'compute', y='train_min', hue='params', palette=palette[:9], ci=10, ax=ax[0])
    sns.lineplot(data=scaling_df, x = 'compute', y='eval_min', hue='params', palette=palette[:9], ci=10, ax=ax[1])
    #sns.lineplot(data=scaling_df, x = 'compute', y='train_ewm', hue='params', palette=palette[:9], ci=10, ax=ax[2])
    #sns.lineplot(data=scaling_df, x = 'compute', y='train_min', hue='params', palette=palette[:9], ci=10, ax=ax[3])
    for i in range(4):
        ax[i].set_xscale('log')
    ax[0].set_ylim(0, .06)
    ax[1].set_ylim(0, .1)

    plt.tight_layout()