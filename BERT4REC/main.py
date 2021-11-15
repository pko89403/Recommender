import numpy as np
import torch 
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, model_checkpoint
from pytorch_lightning import Trainer
import pandas as pd 

from dataset import MovieLens
from models import BERT4REC
from datamodules import MovieLen_Module
from utils import pad_list
from loss import masked_cross_entropy
from metrics import (
    masked_accuracy,
    masked_recall_at_k
)



def test_all():
    dataset = MovieLens('./ml-latest-small/ratings.csv', 'train', 60, 5, .8)
    model = BERT4REC(total_items=10000,
                    emb_dims=32,
                    num_heads=1,
                    dropout_rate=.8,
                    learning_rate=1e-3)
    print(model)

    test_dataset = DataLoader(dataset, batch_size=1)
    for i, data in enumerate(test_dataset):
        x, y_label, mask = data

        y_pred = model(x)
        y_pred = y_pred.view(-1, y_pred.size(2))
        y_label = y_label.view(-1)

        x = x.view(-1)
        mask = mask.view(-1)
        print(mask, mask.shape)
        print(x, x.shape)
        print(x.shape, y_label.shape, y_pred.shape)

        loss = masked_cross_entropy(y_pred, y_label, mask)
        acc = masked_accuracy(y_pred, y_label, mask)
        recall = masked_recall_at_k(y_pred, y_label, mask, 10) 
        print(loss, acc, recall )
        break 

def train(data_dir, save_dir, batch_size, epochs, total_items, emb_dims, seq_len, num_heads, dropout_rate, learning_rate):

    checkpoint = ModelCheckpoint(monitor='val_loss', 
                                mode='min')

    model = BERT4REC(total_items = total_items,
                    emb_dims = emb_dims,
                    num_heads = num_heads,
                    dropout_rate = dropout_rate,
                    learning_rate = learning_rate)

    module = MovieLen_Module(data_dir = data_dir,
                            seq_len = seq_len,
                            valid_sample_size = 5,
                            masking_rate = .8,
                            batch_size = batch_size,
                            shuffle = True)

    trainer = Trainer(max_epochs = epochs, 
                    callbacks=[checkpoint],
                    default_root_dir=save_dir)
    trainer.fit(model, module)
    trainer.test(model, module, verbose=True)

    return model 

def test(model, data_dir, batch_size, seq_len):
    module = MovieLen_Module(data_dir = data_dir,
                            seq_len = seq_len,
                            valid_sample_size = 5,
                            masking_rate = .8,
                            batch_size = batch_size,
                            shuffle = True)
    result = Trainer().test(model, module, verbose=True)
    return result 


def prediction(model, movies, item2idx, idx2item, seq_len, k=30):
    model.eval()
    input = pad_list([item2idx[i] for i in movies] + [1], 'left', seq_len, 0)
    input = torch.tensor(input, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        out = model(input)

    out = out[0, -1].numpy()
    out = np.argsort(out).tolist()[::-1]
    out = [idx2item[i] for i in out if i in idx2item]
    out = out[:k]
    print(out)
    return out[:k]

if __name__ == '__main__':
    test_all()
    # items = pd.read_csv('./ml-latest-small/movies.csv')
    # idx2item = {i+2 : v['title'] for i, v in items.iterrows()}   
    # idx2item[0], idx2item[1] = 'PAD', 'MASK'
    # item2idx =  {v: k for k, v in idx2item.items()}

    # total_items = len(item2idx)

    # model = None
    # try:
    #     model = BERT4REC(total_items=total_items,
    #                     emb_dims=128,
    #                     num_heads=4, 
    #                     dropout_rate=.8,
    #                     learning_rate=1e-8)
    #     model = model.load_from_checkpoint('artifacts/lightning_logs/version_5/checkpoints/epoch=0-step=4.ckpt')

    # except Exception as e:
    #     model = train(data_dir = './ml-latest-small/ratings.csv',
    #                 save_dir = './artifacts',
    #                 batch_size = 256, 
    #                 epochs = 50, 
    #                 total_items = total_items,
    #                 seq_len = 120,
    #                 emb_dims = 128, 
    #                 num_heads = 4,
    #                 dropout_rate = .8,
    #                 learning_rate = 1e-2) 

    

    # test_result = test(model=model,
    #                 data_dir = './ml-latest-small/ratings.csv',
    #                 batch_size=128,
    #                 seq_len=120)
    # print(test_result)


    # test_movie_list = ["Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) (2001)",
    #         "Harry Potter and the Chamber of Secrets (2002)",
    #         "Harry Potter and the Prisoner of Azkaban (2004)",
    #         "Harry Potter and the Goblet of Fire (2005)"]

    # prediction(model, test_movie_list, item2idx, idx2item, 120, k=10)
