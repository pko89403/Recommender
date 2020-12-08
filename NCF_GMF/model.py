import tensorflow as tf 
import numpy as np 

def model(type="MLP",
          user_size=100,
          item_size=100):
  model = None

  if type=="MLP":
    model = MLP(user_size=user_size, item_size=item_size)
  elif type=="GMF":
    model = GMF(user_size=user_size, item_size=item_size)
  elif type=="NCF":
    model = NCF(user_size=user_size, item_size=item_size)
  else:
    pass 
  
  return model 


class NCF(tf.keras.Model):
  def __init__(self):
    super(NCF, self).__init__(name='')
    pass 
  def call(self):
    pass 



class GMF(tf.keras.Model):
  def __init__(self):
    super(GMF, self).__init__(name='')
    pass 
  def call(self):
    pass 



class MLP(tf.keras.Model):
  def __init__(self, user_size, item_size):
    super(MLP, self).__init__(name='')

    # User Embedding
    self.u_var = tf.keras.layers.Embedding(input_dim=user_size,
                                      output_dim=32,
                                      embeddings_initializer='uniform')
    self.u_flatten = tf.keras.layers.Flatten()

    # Item Embedding
    self.i_var = tf.keras.layers.Embedding(input_dim=item_size,
                                      output_dim=32,
                                      embeddings_initializer='uniform')
    self.i_flatten = tf.keras.layers.Flatten()

    # Concatenate our two Embedding vectors togehter
    self.concatenated = tf.keras.layers.Concatenate(axis=1)
    self.dropout = tf.keras.layers.Dropout(0.2)

    # Below we add our four hidden layers along with batch
    # Normalization and Dropout. We use relu as the Activation Funtion
    self.layer_1 = tf.keras.layers.Dense(64, activation='relu', name='layer1')
    self.batch_norm1 = tf.keras.layers.BatchNormalization(name='batch_norm1')
    self.dropout1 = tf.keras.layers.Dropout(0.2, name='dropout1')

    self.layer_2 = tf.keras.layers.Dense(32, activation='relu', name='layer2')
    self.batch_norm2 = tf.keras.layers.BatchNormalization(name='batch_norm2')
    self.dropout2 = tf.keras.layers.Dropout(0.2, name='dropout2')

    self.layer_3 = tf.keras.layers.Dense(16, activation='relu', name='layer3')
    self.layer_4 = tf.keras.layers.Dense(8, activation='relu', name='layer4')

    # Our final single neuron output layer
    self.output_layer = tf.keras.layers.Dense(1,
      kernel_initializer="lecun_uniform",
      name='output_layer')


  def call(self, userId, itemId, training=False):
    user = self.u_var(userId)
    user = self.u_flatten(user)

    item = self.i_var(itemId)
    item = self.i_flatten(item)

    concat_ui = self.concatenated([user, item])
    drop_concat = self.dropout(concat_ui)

    x = self.layer_1(drop_concat)
    x = self.batch_norm1(x, training=training)
    x = self.dropout1(x)
    x = self.layer_2(drop_concat)
    x = self.batch_norm2(x, training=training)
    x = self.dropout2(x)
    x = self.layer_3(x)
    x = self.layer_4(x)

    out = self.output_layer(x)
    return out

def test():
    model_MLP = MLP(user_size=100, item_size=100)
    model_MLP(userId = np.array([1]), itemId = np.array([1]))
