import os, json
import numpy as np
import pandas as pd

original_path = '../original_data/OnlineNewsPopularity/OnlineNewsPopularity.csv'
processed_path = 'data/'

# Read original data
original_data = pd.read_csv(original_path)
print(len(original_data)) # 39644
col_names = original_data.columns
print(col_names)

# Split into train_data and test_data
test_data = original_data.sample(8000, random_state=42)
train_data = original_data[~original_data.index.isin(test_data.index)]

# Drop 'url' and 'timedelta', transform 'shares' into 'label'
train_json = {
	'col_names': col_names.values[2:-2].tolist(), 
	'data': train_data.values[:,2:-2].tolist(), 
	'label': (train_data.values[:,-1] >= 1400).tolist()
}
test_json = {
	'col_names': col_names.values[2:-2].tolist(), 
	'data': test_data.values[:,2:-2].tolist(), 
	'label': (test_data.values[:,-1] >= 1400).tolist()
}

# Save into json files
json.dump(train_json, open(processed_path + 'train.json', 'w'))
json.dump(test_json, open(processed_path + 'test.json', 'w'))
