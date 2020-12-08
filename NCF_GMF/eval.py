import tensorflow as tf 
import numpy as np 
import pandas as pd 
import math
import heapq


def get_hits(h_ranked, holdout):
    for item in h_ranked:
        if item == holdout:
            return 1
        return 0

def eval_rating(idx, test_ratings, test_negatives, K):
    map_item_score = {}

    # get the negative interactions our userr
    items = test_negatives[idx]
    # Get the user idx
    user_idx = test_ratings[idx][0]
    # Get the item idx -> holdout item
    holdout = test_ratings[idx][1]

    # Add the holdout to the end of the negative interactions list.
    items.append(holdout)

    # Prepare our user and item arrays for tensorflow
    predict_user = np.full(len(items), user_idx, dtype='int32').reshape(-1,1)
    np_items = np.array(items).reshape(-1, 1)

    # Feed user and items into the TF graph.

    # Get the predicted score to item id 
    for i in range(len(items)):
        current_item = items[i]
        map_item_score[current_item] = predictions[i]
    
    # Get the K highest ranked items as a list
    h_ranked = heapq.nlargest(K, map_item_score, key=map_item_score.get)

    # Get a list of hit or no hit.
    hits = get_hits(h_ranked, holdout)

    return hits 

def evaluate(df_test, df_neg, K=10):
  hits = []

  test_u = df_test['user_id'].values.tolist()
  test_i = df_test['item_id'].values.tolist()

  test_ratings = list(zip(test_u, test_i))

  df_neg = df_neg.drop(df_neg.columns[0], axis=1)
  test_negatives = df_neg.values.tolist()

  for idx in range(len(test_ratings)):
    hitrate = eval_rating(idx, test_ratings, test_negatives, K)
    hits.append(hitrate)

  return hits

def loss(model, user, item, label):
  labels = tf.cast(label, tf.float32)
  logits = tf.cast(model(user, item), tf.float32)
  loss_object = tf.nn.sigmoid_cross_entropy_with_logits(labels = labels,
                                                        logits = logits)
  return loss_object
  
def grad(model, user, item, label):
  with tf.GradientTape() as tape:
    loss_value = loss(model, user, item, label)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)