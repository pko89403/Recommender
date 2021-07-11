import math
import json 

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
import pandas as pd 

def model_inputs(sequence_length = 4):
    model_inputs = dict()
    model_inputs['user_id'] = layers.Input(name="user_id", shape=(1,), dtype=tf.string)
    model_inputs['sequence_movie_ids'] = layers.Input(
        name="sequence_movie_ids", shape=(sequence_length-1, ), dtype=tf.string
    )
    model_inputs['sequence_ratings'] = layers.Input(
        name="sequence_ratings", shape=(sequence_length-1, ), dtype=tf.float32
    )
    model_inputs['sex'] = layers.Input(name='sex', shape=(1,), dtype=tf.string)
    model_inputs['age_group'] = layers.Input(name='age_group', shape=(1,), dtype=tf.string)
    model_inputs['occupation'] = layers.Input(name='occupation', shape=(1,), dtype=tf.string)

    model_inputs['target_movie_id'] = layers.Input(
        name="target_movie_id", shape=(1,), dtype=tf.string
    )
    
    return model_inputs

class CustomEmbed(layers.Layer):
    def __init__(self, emb_name, vocab):
        super(CustomEmbed, self).__init__()
        
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.output_dim = int(math.sqrt(self.vocab_size))
        
        
        self.custom_embed = layers.Embedding(
            input_dim = self.vocab_size,
            output_dim = self.output_dim,
            name = f"{emb_name}_embedding"
        )
        self.stringLookUp = StringLookup(vocabulary=self.vocab, mask_token=None, num_oov_indices=0)
        print(emb_name, self.output_dim)
        
        
        
    def call(self, input):
        return self.custom_embed(self.stringLookUp(input))


class EmbeddingBags(layers.Layer):
    def __init__(self,
                    sequence_length, 
                    include_user_id=True,
                    include_user_features=True,
                    include_movie_features=True):
        super(EmbeddingBags, self).__init__()
        self.sequence_length = sequence_length
        self.include_user_id = include_user_id
        self.include_user_features = include_user_features
        self.include_movie_features = include_movie_features
        
        
        self.config = None
        with open("./datasets/metadata.json") as json_file:
            self.config = json.load(json_file)
        
        
        self.other_feature_embeddings = {}
        self.encoded_transformer_features = [] 
        self.other_feature_names = [] 

        if include_user_id:
            self.other_feature_names.append("user_id")
        if include_user_features:
            self.other_feature_names.extend(self.config['USER_FEATURES'])
        
        
        for features_name in self.other_feature_names:
            custom_emb = CustomEmbed(emb_name=features_name, vocab=self.config['CATEGORICAL_FEATURES_WITH_VOCABULARY'][features_name])
            self.other_feature_embeddings[features_name] = custom_emb

        self.movie_id_emb = CustomEmbed(emb_name="movie", vocab=self.config["CATEGORICAL_FEATURES_WITH_VOCABULARY"]['movie_id'])
        
        
        genre_vectors = pd.read_csv("./datasets/genres.csv", sep="|", header=None).to_numpy()
        self.movie_genres_emb = layers.Embedding(
            input_dim = genre_vectors.shape[0], 
            output_dim = genre_vectors.shape[1], 
            embeddings_initializer=tf.keras.initializers.Constant(genre_vectors),
            trainable=False,
            name="genres_vector",
        )    
        self.movie_embedding_processor = layers.Dense(
            units=self.movie_id_emb.output_dim, 
            activation='relu',
            name='process_movie_embedding_with_genres',
        )

        self.positional_embedding = layers.Embedding(
            input_dim = self.sequence_length,
            output_dim = self.movie_id_emb.output_dim,
            name='positional_embedding'
        )

    def encode_movie(self, movie_id, include_movie_features):
        encoded_movie = self.movie_id_emb(movie_id)
        
        if include_movie_features:
            movie_idx = self.movie_id_emb.stringLookUp(movie_id)
            movie_genre_vector = self.movie_genres_emb(movie_idx)        
            encoded_movie = self.movie_embedding_processor(
                layers.concatenate([encoded_movie, movie_genre_vector])
            )
    
        return encoded_movie

    def call(self, input):
        
        encoded_other_features = []
        
        for feature_name in self.other_feature_names:
            feature_emb = self.other_feature_embeddings[feature_name](input[feature_name])
            encoded_other_features.append(feature_emb)
    
        # 유저 피처들을 위한 임베딩 벡터를 생성한다.
        if len(encoded_other_features) > 1:
            encoded_other_features = layers.concatenate(encoded_other_features)
        elif len(encoded_other_features) == 1:
            encoded_other_features = encoded_other_features[0]
        else:
            encoded_other_features = None 
            
        encoded_target_movie = self.encode_movie(input['target_movie_id'], self.include_movie_features)
        encoded_sequence_movies = self.encode_movie(input['sequence_movie_ids'], self.include_movie_features)
        positions = tf.range(start=0, limit=self.sequence_length-1, delta=1)
        encoded_positions = self.positional_embedding(positions)

        sequence_ratings = tf.expand_dims(input['sequence_ratings'], -1)
        
        encoded_sequence_movies_with_position_and_rating = layers.Multiply()(
            [(encoded_sequence_movies + encoded_positions), sequence_ratings]
        )
        
        encoded_transformer_features = []
        for encoded_movie in tf.unstack(encoded_sequence_movies_with_position_and_rating, axis=1):
           encoded_transformer_features.append(tf.expand_dims(encoded_movie, 1))
        
        
        encoded_transformer_features.append(encoded_target_movie)
        encoded_transformer_features = layers.concatenate(encoded_transformer_features, axis=1)
        
        return encoded_transformer_features, encoded_other_features        
    
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(
            num_heads = num_heads,
            key_dim = embed_dim,
            dropout = dropout_rate
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.leakyrelu = layers.LeakyReLU()
        self.ffn = layers.Dense(units=ffn_dim)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        self.out = layers.Flatten()
        
    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        out2 = self.leakyrelu(out1)
        out2 = self.ffn(out2)
        out2 = self.dropout2(out2)
        out3 = self.layernorm2(out1 + out2)
        
        return self.out(out3)
    
class FCN(layers.Layer):
    def __init__(self, hidden_dims, dropout_rate=0.1):
        self.dense = layers.Dense(units=hidden_dims)
        self.batchnorm = layers.BatchNormalization()
        self.leakyrelu = layers.LeakyReLU()
        self.dropout = layers.Dropout(dropout_rate)
    
    def call(self, inputs):
        out = self.dense(inputs)
        out = self.batchnorm(inputs)
        out = self.leakyrelu(inputs)
        out = self.dropout(inputs)
        return out 
