from tensorflow import keras
from tensorflow.keras import layers

from . import custom_layers

def create_model(seq_length, num_heads, dropout_rate, hidden_units):
    inputs = custom_layers.model_inputs() 
    transformer_features, other_features = custom_layers.EmbeddingBags(sequence_length=seq_length)(inputs)
    transformer_output = custom_layers.TransformerBlock(transformer_features.shape[2], num_heads, transformer_features.shape[2])(transformer_features)
    
    
    if other_features is not None:
        reshaped_other_features = layers.Reshape([other_features.shape[-1]])(other_features)
        features = layers.concatenate([transformer_output,reshaped_other_features])
        

    for num_units in hidden_units:
        features = layers.Dense(num_units)(features)
        features = layers.BatchNormalization()(features)
        features = layers.LeakyReLU()(features)
        features - layers.Dropout(dropout_rate)(features)
        
    outputs = layers.Dense(units=1)(features)
    model = keras.Model(inputs=inputs, outputs=outputs)

    print(model.summary())
    return model