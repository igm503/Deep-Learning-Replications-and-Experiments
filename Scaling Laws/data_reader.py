import pandas as pd
import math
from itertools import product

def prepare_df(df):
    new_df = df.copy()
    new_df['compute'] = (new_df['model_size'] ** 2) * (new_df['data_size'])
    new_df = new_df.round(2)
    return new_df

with open('./Desktop/raw_data.txt') as file:
    results = file.readlines()

results = results[2: :2]
for i in range(len(results)):
    results[i] = results[i][-20:]
    results[i] = results[i].split(']')[-1]
    results[i] = float(results[i].rstrip('\n').lstrip())

model_size_list = [math.sqrt(2) ** i for i in range(9)]
data_size_list = [1 / i for i in model_size_list]
test_combos = list(product(model_size_list, data_size_list))

new_df = pd.DataFrame({'data_size': [], 'model_size': [], 'eval_loss': []})

count = 0
for i in range(6):
    for model, data, in test_combos:
        new_df.loc[len(new_df.index)] = [data, model, results[count]]
        count += 1

df = prepare_df(new_df)

df.to_csv('df.csv')