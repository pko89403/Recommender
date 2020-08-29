import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim 
import random 

from data_utils import Dataset_Generator, Train_Dataset
from torch.utils.data import Dataset, DataLoader
from model import Paragraph2Vec_DM
import numpy as np 

def train():
    items_data_file = "/my_custom_dataset/items.csv"
    dataset = Dataset_Generator(items_data_file)
    train_dataset = Train_Dataset(dataset)
    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=64,
                            shuffle=True) 
    
    total_items = len(train_dataset.title_to_titleIdx)
    total_words = len(train_dataset.word_to_index)

    print('Total Items : ', total_items)
    print('Total Items : ', total_words)

    model = Paragraph2Vec_DM(embed_dim=32,
                             total_items=total_items,
                             total_words=total_words)
    
    #criterion = nn.CrossEntropyLoss() # log_softmax + NLLLoss == CrossEntropyLoss
    #criterion = nn.MultiLabelSoftMarginLoss()
    criterion = nn.NLLLoss()
    print(model.parameters)
    optimizer = optim.Adam(model.parameters(), lr=1e-5) #  ?

    losses = []
    
    epoch = 0
    #for epoch in range(total_epoch):
    with open('losses.txt', 'w') as loss_file:
        model.train()
        while(True):
            epoch+=1
            running_loss = list()

            for train_iter, train_data in enumerate(train_loader):  
                sample_data = train_data
                context = sample_data.context
                product = sample_data.product
                labels = sample_data.target#.view(-1,1)
                
                outputs = model(context,product,None)

                #print(outputs.shape)
                #print(labels.shape)
                optimizer.zero_grad()
                loss = criterion(outputs, labels)

                
                loss.backward()
                optimizer.step()

                running_loss.append( loss.item() )
                
                if(train_iter % 100 == 0):
                    print(f"Iter {train_iter}, running_losses : {loss.item()}")
                # break
            avg_mean = np.mean(running_loss)
            print(f"\tEpoch {epoch}'s Average Loss : {avg_mean}")
            losses.append(avg_mean)

            wline = str(epoch) + ',\t' + str(avg_mean) 
            ofile = './model_save/model_' + str(epoch%5) + '_checkpoint.pt'
            loss_file.write(wline)
            torch.save(model, ofile)
            

if __name__ == "__main__":
    train()