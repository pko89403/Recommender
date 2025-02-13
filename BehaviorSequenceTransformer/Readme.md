# Behavior Sequence Transformer with MovieLens Using Keras
## MovieLens Dataset
```
wget http://files.grouplens.org/datasets/movielens/ml-1m.zip
```
## Project Architecture
```
.

├── __init__.py
├── dataloader
│   ├── __init__.py
│   └── movie_lens.py
├── datasets
│   ├── genres.csv
│   ├── metadata.json
│   ├── ml-1m
│   │   ├── README
│   │   ├── movies.dat
│   │   ├── ratings.dat
│   │   └── users.dat
│   ├── test_data.csv
│   └── train_data.csv
├── model
│   ├── __init__.py
│   ├── bst.py
│   └── custom_layers.py
├── Readme.md
├── dataset_prepare.ipynb
└── main.py
```
## Network Architecture
```
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
age_group (InputLayer)          [(None, 1)]          0
__________________________________________________________________________________________________
occupation (InputLayer)         [(None, 1)]          0
__________________________________________________________________________________________________
sequence_movie_ids (InputLayer) [(None, 3)]          0
__________________________________________________________________________________________________
sequence_ratings (InputLayer)   [(None, 3)]          0
__________________________________________________________________________________________________
sex (InputLayer)                [(None, 1)]          0
__________________________________________________________________________________________________
target_movie_id (InputLayer)    [(None, 1)]          0
__________________________________________________________________________________________________
user_id (InputLayer)            [(None, 1)]          0
__________________________________________________________________________________________________
embedding_bags (EmbeddingBags)  ((None, 4, 62), (Non 781090      age_group[0][0]
                                                                 occupation[0][0]
                                                                 sequence_movie_ids[0][0]
                                                                 sequence_ratings[0][0]
                                                                 sex[0][0]
                                                                 target_movie_id[0][0]
                                                                 user_id[0][0]
__________________________________________________________________________________________________
transformer_block (TransformerB (None, 248)          50902       embedding_bags[0][0]
__________________________________________________________________________________________________
reshape (Reshape)               (None, 84)           0           embedding_bags[0][1]
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 332)          0           transformer_block[0][0]
                                                                 reshape[0][0]
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 256)          85248       concatenate[0][0]
__________________________________________________________________________________________________
batch_normalization (BatchNorma (None, 256)          1024        dense_1[0][0]
__________________________________________________________________________________________________
leaky_re_lu_1 (LeakyReLU)       (None, 256)          0           batch_normalization[0][0]
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 128)          32896       leaky_re_lu_1[0][0]
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 128)          512         dense_2[0][0]
__________________________________________________________________________________________________
leaky_re_lu_2 (LeakyReLU)       (None, 128)          0           batch_normalization_1[0][0]
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 1)            129         leaky_re_lu_2[0][0]
==================================================================================================
Total params: 951,801
Trainable params: 881,139
Non-trainable params: 70,662
__________________________________________________________________________________________________
```
## Run
```
python main.py
```

## Training
```
1658/1658 [==============================] - 25s 14ms/step - loss: 1.6344 - mean_absolute_error: 0.9874
```
## Evaluation
```
Test MAE : 0.806
```
## Sample Predict For API
```
Predict Result - [[3.1236796]]
```

## Reference
- https://keras.io/examples/structured_data/movielens_recommendations_transformers/
