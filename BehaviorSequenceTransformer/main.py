import os 

import tensorflow as tf 
from tensorflow import keras 

from dataloader.movie_lens import create_datasets
from model import bst

HIDDEN_UNITS = [256, 128]
DROPOUT_RATE = 0.1
NUM_HEADS = 3
COLUMNS = ["user_id", "sequence_movie_ids", "sequence_ratings", "sex", "age_group", "occupation"]
SEQUENCE_LENGTH = 4
DATASET_PATH = './datasets'
    
model = bst.create_model(seq_length=SEQUENCE_LENGTH,
                         num_heads=NUM_HEADS,
                         dropout_rate=DROPOUT_RATE,
                         hidden_units=HIDDEN_UNITS)

def watch_dataset(dataset):  
    for batch in dataset.take(1):
        print(batch)
        break 


def train_model(train_file):
    train_dataset = create_datasets(train_file, COLUMNS, shuffle=False, batch_size=256)
    
    model.compile(
    optimizer=keras.optimizers.Adagrad(learning_rate=0.01),
    loss=keras.losses.MeanSquaredError(),
    metrics=[keras.metrics.MeanAbsoluteError()],
    )

    model.fit(train_dataset, epochs=1)

def evaluate_model(test_file):
    eval_dataset = create_datasets(test_file, COLUMNS, shuffle=False, batch_size=1)

    _, rmse = model.evaluate(eval_dataset, verbose=0)
    print(f"Test MAE : {round(rmse, 3)}")


def sample_predict_model():
    import numpy as np 
    from collections import OrderedDict

    sequence_movie_ids = tf.reshape(tf.convert_to_tensor(np.asarray(['movie_551' ,'movie_2858' ,'movie_2033'])), 
                                    shape=(1,3))
    sequence_ratings = tf.reshape(tf.convert_to_tensor(np.asarray([3.,3.,3.]), dtype=tf.float32), 
                                shape=(1,3))
    user_id = tf.constant('user_10', shape=(1,))
    sex = tf.constant('F', shape=(1,))
    age_group = tf.constant('group_35', shape=(1,))
    target_movie_id = tf.constant('movie_3155', shape=(1,))
    occupation = tf.constant('occupation_1', shape=(1,))

    model_inputs = OrderedDict()
    model_inputs['user_id'] = user_id
    model_inputs['sequence_movie_ids'] = sequence_movie_ids
    model_inputs['sequence_ratings'] = sequence_ratings
    model_inputs['sex'] = sex
    model_inputs['age_group'] = age_group
    model_inputs['occupation'] = occupation
    model_inputs['target_movie_id'] = target_movie_id

    print(f"Predict Result - {model.predict(model_inputs)}")


if __name__ == "__main__":
    train_filename = os.path.join(DATASET_PATH, "train_data.csv")
    test_filename = os.path.join(DATASET_PATH, "test_data.csv")
    model.summary()
    train_model(train_filename)
    evaluate_model(test_filename)
    sample_predict_model()