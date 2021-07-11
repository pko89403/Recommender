import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import StringLookup

import math, json
import pandas as pd 


def create_datasets(csv_file_path, col_names, shuffle=False, batch_size=128):
    def process(features):
        movie_ids_string = features['sequence_movie_ids']
        sequence_movie_ids = tf.strings.split(movie_ids_string, ",").to_tensor()
        
        # 시퀀스 내 마지막 영화가 타겟 영화가 된다.
        features['target_movie_id'] = sequence_movie_ids[:, -1]
        features['sequence_movie_ids'] = sequence_movie_ids[:, :-1]
        
        ratings_string = features['sequence_ratings']
        sequence_ratings = tf.strings.to_number(
            tf.strings.split(ratings_string, ","), tf.dtypes.float32
        ).to_tensor()
        
        # 시퀀스 내 마지막 평점이 모델이 예측해야할 타겟이 된다.
        target = sequence_ratings[:, -1]
        features['sequence_ratings'] = sequence_ratings[:, :-1]
        
        return features, target 

    dataset = tf.data.experimental.make_csv_dataset(
        csv_file_path,
        batch_size=batch_size,
        column_names=col_names, 
        num_epochs = 1, 
        header= False,
        field_delim= "|",
        shuffle=shuffle,
    ).map(process)
    
    return dataset 
