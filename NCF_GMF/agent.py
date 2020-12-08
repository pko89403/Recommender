from model import model
from data_utils import load_dataset
from data_loader import get_train_instances
import tensorflow as tf 
DATA_PATH = "/Users/kangseokwoo/Recsys_test/NCF_GMF/lastfm-dataset-360K/usersha1-artmbid-artname-plays.tsv"
MODEL_TYPE = "MLP"

if __name__ == "__main__":
    
    uids, iids, df_train, df_test, df_neg, users, items, item_lookup = load_dataset(DATA_PATH)      

    # Hyper-Parameters
    num_neg = 1
    latent_features = 8
    epochs = 10
    batch_size = 256
    learning_rate = 1e-3
    
    # Model
    model = model(type="MLP",
                user_size=len(users),
                item_size=len(items))



    for epoch in range(epochs):
        train_instances = get_train_instances(uids=uids, 
                                              iids=iids, 
                                              items=items, 
                                              num_neg=num_neg)

        dataset = tf.data.Dataset.from_tensor_slices(train_instances.items())
        dataset = dataset.batch(batch_size).shuffle()
        
        for elem in dataset:
            print(elem)
            break
        break 