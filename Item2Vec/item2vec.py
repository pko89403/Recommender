import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import random
from collections import Counter
import pandas as pd
class item2vec(nn.Module): # Skip-Gram Model with Negative Sampling 
    def __init__(self, item_size, embedding_dim):
        super(item2vec, self).__init__()

        self.embedding_v = nn.Embedding(item_size, embedding_dim)
        self.embedding_u = nn.Embedding(item_size, embedding_dim)
        self.log_sigmoid = nn.LogSigmoid()

        init_range = (2.0 / (item_size + embedding_dim)) ** 0.5 
        self.embedding_v.weight.data.uniform_(-init_range, init_range)
        self.embedding_u.weight.data.uniform_(-init_range, init_range)

    def forward(self, input_items, pos_items, neg_items):
        input_embeds = self.embedding_v(input_items)
        pos_embeds = self.embedding_u(pos_items)
        neg_embeds = -self.embeding_u(neg_items)

        pos_score = pos_embeds.bmm(input_embeds.transpose(1,2)).squeeze(2)
        neg_score = torch.sum(neg_embeds.bmm(input_embeds.transpose(1,2)).squeeze(2), 1).view(input_items.size(0), -1)

        loss = self.log_sigmoid(pos_score) + self.log_sigmoid(neg_score)
        return -torch.mean(loss)

    def prediction(self, input_items):
        return self.embedding_v(input_items)
    

def dataETL(filename="./MovieLens/ratings.csv"):
    # 영화 X 영화 매트릭스를 만든다.
    # 데이터셋의 헤더 userId,movieId,rating,timestamp
    raw_data = pd.read_csv(filename)
    
    line = raw_data['userId'].unique()
    
    datasets = list()
    for i in line:
        datasets.append( raw_data[raw_data['userId'] == i ]['movieId'].tolist() )
    
    fla = lambda k : [i for sublist in k for i in sublist]
    item_counter = Counter(fla(datasets))
    item = [w for w, c in item_counter.items()]
    total_items = len(item)


    item2index = {}
    for vo in item:
        if( item2index.get(vo) is None):
            item2index[vo] = len(item2index)
    index2word = {v: k for k, v in item2index.items()}

    new_movie_lists = []
    for m in datasets:
        m = [item2index[n] for n in m]
        new_movie_lists.append(m)
    
    uni_table = list()
    f = sum([item_counter[it] ** 0.75 for it in item])
    z = 0.0001
    for it in item:
        uni_table.extend([it] * int(((item_counter[it] ** 0.75) / f) / z))

    return total_items, new_movie_lists, uni_table

    
def get_positive_sample(sample_list):
    positive_samples = list()
    
    for sublist in sample_list:
        sublist_len = len(sublist)
        for ite in sublist:
            ite_index = sublist.index(ite)
            for j in range(sublist_len):
                if ite_index != j:
                    positive_samples.append([ite, sublist[j]])
    
    target_items = []
    context_items = []
    
    for item_pair in positive_samples:
        target_items.append(item_pair[0])
        context_items.append(item_pair[1])
    
    return target_items, context_items

def get_negative_sample(input_items, target_items, un_table, k):
    batch_size = len(targets)
    negative_samples = list()

    for i in range(batch_size):
        neg_sample = list()
        input_index = input_items[i][0]
        target_index = target_items[i][0]

        while( len(neg_sample) < k ):
            neg = random.sample(un_table)
            if neg == target_index or neg == input_index:
                continue
            neg_sample.append(neg)
        negative_samples.append(neg_sample)
    return negative_samples

def get_batch_sample(batch_size, train_data):
    random.shuffle(train_data)

    start_index = 0
    end_index = batch_size

    while( end_index < len(train_data)):
        batch_data = train_data[start_index:end_index]
        temp = end_index
        end_index = end_index + batch_size
        start_index = temp
        yield batch_data
    
    if( end_index >= len(train_data)):
        batch_data = train_data[start_index:]
        yield batch_data

if __name__ == "__main__":
    
    total_items, movie_sample_data, uni_table =  dataETL()
    target_items, context_items = get_positive_sample(movie_sample_data)
    train_data = [[target_items[i], context_items[i]] for i in range(len(target_items))]
    print( total_items )
    model = item2vec(item_size=total_items, embedding_dim=10)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        for i, batch in enumerate(get_batch_sample(2048, train_data)):
            target = [[p[0]] for p in batch]
            context = [[q[1]] for q in batch]
            negative = get_negative_sample(target_items, context_items, uni_table)
            
            target = Variable(torch.LongTensor(target_items))
            context = Variable(torch.LongTensor(context_items))
            negative = Variable(torch.LongTensor(negative))

            item2vec.zero_grad()
            loss = item2vec(target, context, negative)

            loss.backward()
            optimizer.step()

            print(f"Epoch : {epoch+1}, Batch : {i+1}, loss : {loss}")