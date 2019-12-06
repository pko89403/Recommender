# https://github.com/wolfecameron/music_recommendation/blob/master/music_ratings_learner.ipynb
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data

import pandas as pd
from Recommender_EDA import shuffle_data
import numpy as np
import matplotlib.pyplot as plt

class Recommender(nn.Module):
    def __init__(self, num_users, num_artists, num_factors):
        super().__init__()
        # 두 개의 임베딩 매트릭스를 선언한다.
        # nn.Embedding(총 유저 수, 임베딩 시킬 벡터의 차원) 총 유저에 대해 해당 차원 만큼 임베딩 매트릭스를 생성함.
        self.u = nn.Embedding(num_users, num_factors)
        self.a = nn.Embedding(num_artists, num_factors)
        self.u.weight.data.uniform_(-.01, .01)
        self.a.weight.data.uniform_(-.01, .01)
        # bias 를 추가한다.
        self.ub = nn.Embedding(num_users, 1)
        self.ab = nn.Embedding(num_artists, 1)
        self.ub.weight.data.uniform_(-.01, .01)
        self.ab.weight.data.uniform_(-.01, .01)

    def forward(self, cats, conts):
        # 해당 인덱스에 위치하는 유저, 아티스트 두 벡터를 가져온다.
        # 내적해서 rating을 예측한다.
        users, artists = cats[:, 0], cats[:, 1]
        us, art = self.u(users), self.a(artists)
        dp = (us * art).sum(1)
        # Bias 추가한 것을 더한다. 혓
        dpb = dp + self.ub(users).squeeze() + self.ab(artists).squeeze()
        return dpb

class RecDataset(data.Dataset):
    'Create custom class for pytorch data set'
    def __init__(self, path, filename):
        "Initialize DataFrame"
        self.data = pd.read_csv(path + filename)
    
    def __len__(self):
        'get total number of samples'
        return self.data.shape[0]

    def __getitem__(self, index):
        'get data sample from dataset'
        return (np.array(self.data.loc[index, :]))

def get_loader(path, filename, bs):
    """ Method for getting a data loader from a csv file"""
    shuffle_data(path, filename)
    dataset = RecDataset(path, filename)
    dataLoader = data.DataLoader(dataset, batch_size=bs, num_workers=2)
    return dataLoader

if __name__ == "__main__":
    path = "/Users/amore/Recsys_test/MusicRec_MF/Dataset/csv/"
    
    # Read in data frame created by EDA notebook
    contig_df = pd.read_csv(path + "contig_music_ratings.csv")
    # separate music data into separate training and testing files
    shuffle_data(path, "contig_music_ratings.csv")
    train_amt = .7
    data_size = len(contig_df)
    train_size = int(train_amt*data_size)
    test_size = data_size - train_size

    #get seperate dataframes with train and test data
    train_data = contig_df.iloc[:train_size, :]
    test_data = contig_df.iloc[train_size:, :]
    print(f"{train_size} -> {len(train_data)}")
    print(f"{test_size} -> {len(test_data)}")

    # write these train and test data to separate csv file
    train_data.to_csv(path + "music_train.csv", index=False)
    test_data.to_csv(path + "music_test.csv", index=False)

    #find number of artists and users being  used 1
    num_users = contig_df.loc[:, "userID"].nunique()
    num_artists = contig_df.loc[:, "artistID"].nunique()
    print(num_users)
    print(num_artists)

    full_data = RecDataset(path, 'contig_music_ratings.csv')

    print(full_data[0])
    print(full_data[1])
    print(full_data[2])
    print(full_data[3])
    print(full_data[4])
    full_data.data.head()    
    
    # create data loaders for training and test data
    train_loader = get_loader(path, "music_train.csv", 32)
    #print(train_loader)
    #print(next(iter(train_loader)))

    # declare the size of the embeddings to be used
    num_factors = 40
    wd = 1e-3
    criterion = nn.MSELoss()

    # Find the optimal learning rate with chich to begin training

    losses = []
    lrs = []
    lr = 1e-6
    practice_rec_model = Recommender(num_users, num_artists, num_factors)
    opt = optim.SGD(practice_rec_model.parameters(), lr, weight_decay=wd, momentum=0.9)

    while( lr <= 1):
        test_data = next(iter(train_loader))
        inputs = test_data[:, :2].long()
        expected = autograd.Variable(test_data[:, 2].float())

        opt.zero_grad()

        # run model
        outputs = practice_rec_model(inputs, None)
        loss = criterion(outputs, expected)
        loss.backward()
        opt.step()
        losses.append(loss.data)
        lrs.append(lr)

        # go to the next learning rate
        lr *= 1.25

        # change the learning rate in the optimizer
        for param_group in opt.param_groups:
            param_group['lr'] = lr
        
    #plt.plot(lrs, losses)
    #plt.show()
    training_loss = []

    # declare training constants
    n_epoch = 3
    lr = .1

    # declare loss criterion for the model
    criterion = nn.MSELoss()

    # create model and optimzer
    recommender_model = Recommender(num_users, num_artists, num_factors)
    opt = optim.SGD(recommender_model.parameters(), lr, weight_decay=wd, momentum=0.9)


    # Create training loop to train thre recommender model on test data
    train_loader = get_loader(path, "music_train.csv", 32)
    for epoch in range(n_epoch): # loop over the dataset multiple times
        
        running_loss = 0.0

        for i, train_data in enumerate(train_loader):
            # get the inputs
            inputs = train_data.long()
            actual_out  = autograd.Variable(train_data[:, 2].float())

            # zero the parameter gradients
            opt.zero_grad()

            # forward + backward + optimize
            outputs = recommender_model.forward(inputs, None)
            loss = criterion(outputs, actual_out)
            loss.backward()
            opt.step()

            # print statistics
            running_loss += loss
            
            #input()
            if i % (len(train_loader)) == (len(train_loader) - 1):
                training_loss.append((running_loss/len(train_loader)))
                print(f"[{epoch + 1}, {i + 1}] loss : {running_loss/len(train_loader)}")
                running_loss = 0.0 

    print('Finished Training')                 
    
    # Continue Training while reducing the learning rate to arrive at an optimal solution
    # decrease the learning rate then train model again
    plt.title("Recommender System Training")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.plot(training_loss)
    plt.show()

    torch.save(recommender_model.state_dict(), path + "custom_model")

    print(num_users, num_artists)

    rec_model = Recommender(num_users, num_artists, 40)
    rec_model.load_state_dict(torch.load(path + "custom_model"))

    # get dictionaries that contain index mappings to study embeddings
    import pickle
    user_to_index_dict = pickle.load(open(path + "user_to_index_dict.txt", "rb"))
    index_to_user_dict = pickle.load(open(path + "index_to_user_dict.txt", "rb"))
    artist_to_index_dict = pickle.load(open(path + "artist_to_index_dict.txt", "rb"))
    index_to_artist_dict = pickle.load(open(path + "index_to_artist_dict.txt", "rb"))

    # get the dataframe of artists mapped to IDs
    import pandas as pd
    artist_df = pd.read_csv(path + 'artists_cleaned.csv')
    artist_df.head()

    # get data from artist embeddings into a numpy array
    np_artist_embedding = rec_model.ab.weight.data.numpy()
    
    np_embed = np.array(np_artist_embedding, copy=True)
    np_embed.sort(axis=0)

    # find the 5 largest values to see which artists they are for
    best_indices = np.argsort(np_artist_embedding, axis=0)[-5:, :]

    # take indices and convert to the indices for embedding
    for i in range(best_indices.shape[0]):
        best_indices[i, :] = index_to_artist_dict[best_indices[i, :].item()]

    sorted_embeddings = np.sort(np_artist_embedding, axis=0)
    best_embeddings = sorted_embeddings[-5:, :]

    best_artists = []
    for i in range(5):
        artist = best_indices[i, :].item()
        artist_name = artist_df.loc[artist_df["id"] == artist].iloc[:, 1].item()
        artist_embedding = best_embeddings[i, :].item()
        best_artists.append((artist_name, artist_embedding))
    
    best_artists = [i for i in reversed(best_artists)]
    print( best_artists )