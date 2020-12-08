import pandas as pd 
import numpy as np
import pickle
from timer import timer
import json 
import os 

@timer
def load_dataset(data_dir):
  train_csv = "train.csv"
  test_csv = "test.csv"
  neg_csv = "neg.csv"  
  user_lookup_csv = "user_lookup.csv"
  item_lookup_csv = "item_lookup.csv"

  df_train = None 
  df_test = None 
  user_lookup = None 
  item_lookup = None 
  users = None 
  items = None 
  df_neg = None 

  if(os.path.exists(train_csv) and os.path.exists(test_csv)):
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    if(os.path.exists(user_lookup_csv)):  
      user_lookup = pd.read_csv(user_lookup_csv)

    if(os.path.exists(item_lookup_csv)):  
      item_lookup = pd.read_csv(item_lookup_csv)

  else:
    df = pd.read_csv(data_dir, sep='\t')
  
    df = df.drop(df.columns[1], axis=1)
    df.columns = ['user', 'item', 'plays']
    df = df.dropna()
    df = df.loc[df.plays != 0]


    df_count = df.groupby(['user']).count()
    df['count'] = df.groupby('user')['user'].transform('count')
    df = df[df['count'] > 1]

  
    # Return Series of codes as well as the index.
    df['user_id'] = df['user'].astype('category').cat.codes
    df['item_id'] = df['item'].astype('category').cat.codes  


    item_lookup = df[['item_id', 'item']].drop_duplicates()
    item_lookup['item_id'] = item_lookup.item_id.astype(str)
    item_lookup.to_csv(item_lookup_csv)

    user_lookup = df[['user_id', 'user']].drop_duplicates()
    user_lookup['user_id'] = user_lookup.user_id.astype(str)
    user_lookup.to_csv(user_lookup_csv)

    df = df[['user_id', 'item_id', 'plays']]
    df_train, df_test = train_test_split(df)
    df_train.to_csv(train_csv)
    df_test.to_csv(test_csv)


  users = list(np.sort(user_lookup.user_id))
  items = list(np.sort(item_lookup.item_id))
  

  rows = df_train.user_id.astype(int)
  cols = df_train.item_id.astype(int)

  values = list(df_train.plays)

  uids = np.array(rows.tolist())
  iids = np.array(cols.tolist())

  if(os.path.exists(neg_csv)):
    df_neg = pd.read_csv(neg_csv)
  else:
    df_neg = get_negatives(uids, iids, items, df_test, neg_cnt=1)
    df_neg.to_csv(neg_csv)

  return uids, iids, df_train, df_test, df_neg, users, items, item_lookup

def mask_first(x):
  result = np.ones_like(x)
  result[0] = 0
  return result

def train_test_split(df):
  df_test = df.copy(deep=True)
  df_train = df.copy(deep=True)

  # Group by user and select only the first item for each user 
  df_test = df_test.groupby(['user_id']).first()
  df_test['user_id'] = df_test.index
  df_test = df_test[['user_id', 'item_id', 'plays']]
  df_test.rename(index={'name': ''}, inplace=True)

  # Remove the same items as we for our test set in our training set
  mask = df.groupby(['user_id'])['user_id'].transform(mask_first).astype(bool)
  df_train = df.loc[mask]

  return df_train, df_test

def get_negatives(uids, iids, items, df_test, neg_cnt=5):
  negativeList = []
  test_u = df_test['user_id'].values.tolist()
  test_i = df_test['item_id'].values.tolist()

  test_ratings = list(zip(test_u, test_i))
  zipped = set(zip(uids, iids))

  for (u, i) in test_ratings:
    negatives = []
    negatives.append((u, i))
    for t in range(neg_cnt):
      j = np.random.randint(len(items)) # Get random item id
      while (u, j) in zipped: # Check if there is an interaction
        j = np.random.randint(len(items)) # If yes, generate a new item id 
      negatives.append(j) # Once a negative interaction is found we add it
    negativeList.append(negatives)

  df_neg = pd.DataFrame(negativeList)

  return df_neg