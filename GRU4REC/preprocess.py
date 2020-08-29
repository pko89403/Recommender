import numpy as np 
import pandas as pd 
import datetime as dt 

raw_data_path = "./Dataset/yoochoose-clicks.dat"
# -Session ID – the id of the session.
# -Timestamp – the time when the click occurred. 
# -Item ID – the unique identifier of the item that has been clicked.
# -Category – the context of the click.
data_columns = ["SessionID", "TimeStamp", "itemID", "Category"]
data = pd.read_csv(raw_data_path, sep=",", header=None, names=data_columns)
print(data.head())

data['time'] = data.TimeStamp.apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp())
del(data['TimeStamp'])

session_lengths = data.groupby('SessionID').size()
# np.in1d(x, y) : 첫번째 배열 x가 두번째 배열 y의 원소를 포함하고 있는지 여부의 불리언 배열 반환
data = data[np.in1d(data.SessionID, session_lengths[session_lengths > 1].index)]

ITEM_COUNT_THRESHOLD = 5
item_supports = data.groupby('itemID').size()
data = data[np.in1d(data.itemID, item_supports[item_supports >= ITEM_COUNT_THRESHOLD].index)]

session_lengths = data.groupby('SessionID').size()
data = data[np.in1d(data.SessionID, session_lengths[session_lengths>=2].index)]

# Tain & Test
time_max = data.time.max()

session_max_times = data.groupby("SessionID").time.max()
session_train = session_max_times[session_max_times < time_max - 86400].index
session_test = session_max_times[session_max_times >= time_max - 86400].index

train = data[np.in1d(data.SessionID, session_train)]
test = data[np.in1d(data.SessionID, session_test)]
test = test[np.in1d(test.itemID, train.itemID)] 
ts_length = test.groupby('SessionID').size()
test = test[np.in1d(test.SessionID, ts_length[ts_length >= 2].index)]

TRAIN_DATA_PATH = "./Dataset/processed/train_full.txt"
TEST_DATA_PATH = "./Dataset/processed/test_full.txt"
train.to_csv(TRAIN_DATA_PATH, sep='\t', index=False)
test.to_csv(TEST_DATA_PATH, sep='\t', index=False)

### Train_Train & Train_Valid
time_max = train.time.max()

train_session_max_times = train.groupby("SessionID").time.max()
session_train_tr = train_session_max_times[train_session_max_times < time_max - 86400].index
session_valid = train_session_max_times[train_session_max_times >= time_max - 86400].index

train_tr = data[np.in1d(data.SessionID, session_train_tr)]
valid = data[np.in1d(data.SessionID, session_valid)]
valid = valid[np.in1d(valid.itemID, train_tr.itemID)] 
ts_length = valid.groupby('SessionID').size()
valid = valid[np.in1d(valid.SessionID, ts_length[ts_length >= 2].index)]

TRAIN_TR_DATA_PATH = "./Dataset/processed/train_train.txt"
VALID_DATA_PATH = "./Dataset/processed/train_valid.txt"
train_tr.to_csv(TRAIN_TR_DATA_PATH, sep='\t', index=False)
valid.to_csv(VALID_DATA_PATH, sep='\t', index=False)