import numpy as np 
from timer import timer
import math 
import pickle   
import os 

@timer
def get_train_instances(uids, iids, items, num_neg):
  train_instance_dict_fname = "train_instance_dict.pickle"

  if(os.path.exists(train_instance_dict_fname)):
    with open(train_instance_dict_fname, 'rb') as handle:
      return pickle.load(train_instance_dict_fname)

  
  train_instance_dict = dict()
  train_instance_dict['user_input'] = list()
  train_instance_dict['item_input'] = list()
  train_instance_dict['labels'] = list()


  zipped = set(zip(uids, iids))

  for (u, i) in zip(uids, iids):
    # Add positive interaction
    train_instance_dict['user_input'].append(u)
    train_instance_dict['item_input'].append(i)
    train_instance_dict['labels'].append(1)

    # Sample random negative interaction
    for t in range(num_neg):
      j = np.random.randint(len(items))
      while (u, i) in zipped:
        j = np.random.randint(len(items))
      
      train_instance_dict['user_input'].append(u)
      train_instance_dict['item_input'].append(j)
      train_instance_dict['labels'].append(0)

    with open('train_instance_dict.pickle', 'wb') as handle:
      pickle.dump(train_instance_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
    return train_instance_dict

def random_mini_batches(U, I, L, mini_batch_size=256):
  mini_batches = []
  
  shuffled_U, shuffled_I, shuffled_L = shuffle(U, I, L)

  num_complete_batches = int(math.floor(len(U)/mini_batch_size))
  for k in range(0, num_complete_batches):
    mini_batch_U = shuffled_U[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
    mini_batch_I = shuffled_I[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
    mini_batch_L = shuffled_L[k * mini_batch_size : k * mini_batch_size + mini_batch_size]

    mini_batch = (mini_batch_U, mini_batch_I, mini_batch_L)
    mini_batches.append(mini_batch)
  
  if len(U) % mini_batch_size != 0:
    mini_batch_U = shuffled_U[num_complete_batches * mini_batch_size : len(U)]
    mini_batch_I = shuffled_I[num_complete_batches * mini_batch_size : len(U)]
    mini_batch_L = shuffled_L[num_complete_batches * mini_batch_size : len(U)]

    mini_batch = (mini_batch_U, mini_batch_I, mini_batch_L)
    mini_batches.append(mini_batch)

  return mini_batches 