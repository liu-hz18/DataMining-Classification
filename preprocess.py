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

# Drop data with error
original_data = original_data[original_data[' n_unique_tokens'] <= 1]  # 39644

# Replace col with easy ones
new_col_names, weekday_is, data_channel_is = [], [], []
for col in col_names:
	if 'weekday_is' in col:
		weekday_is.append(col)
		if 'monday' in col:
			new_col_names.append(' weekday')
	elif 'data_channel_is' in col:
		data_channel_is.append(col)
		if 'lifestyle' in col:
			new_col_names.append(' data_channel')
	else:
		new_col_names.append(col)
print(weekday_is)
print(data_channel_is)

for index, row in original_data.iterrows():
	w, d = -1, -1
	for i, _ in enumerate(weekday_is):
		if row[_] > 0:
			# w = _.split('_')[-1]
			w = i
			break
	for i, _ in enumerate(data_channel_is):
		if row[_] > 0:
			# d = _.split('_')[-1]
			d = i
			break
	original_data.loc[index, ' weekday'] = w
	original_data.loc[index, ' data_channel'] = d
original_data = original_data[new_col_names]
col_names = original_data.columns
print(col_names)


# Split into train_data and test_data
test_data = original_data.sample(8000, random_state=42)
train_data = original_data[~original_data.index.isin(test_data.index)]

# Drop 'url' and 'timedelta', transform 'shares' into 'label'
train_json = {
	'col_names': col_names.values[2:-1].tolist(), 
	'data': train_data.values[:,2:-1].tolist(), 
	'label': (train_data.values[:,-1] >= 1400).tolist()
}
test_json = {
	'col_names': col_names.values[2:-1].tolist(), 
	'data': test_data.values[:,2:-1].tolist(), 
	'label': (test_data.values[:,-1] >= 1400).tolist()
}

# Save into json files
json.dump(train_json, open(processed_path + 'train.json', 'w'))
json.dump(test_json, open(processed_path + 'test.json', 'w'))
