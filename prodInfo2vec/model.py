import torch 
import torch.nn as nn
import torch.nn.functional as F
class Paragraph2Vec_DM(nn.Module):
    def __init__(self, embed_dim, total_items, total_words):
        super(Paragraph2Vec_DM, self).__init__()
        self.embed_dim = embed_dim
        # paragraph matrix
        self.item_embedding_layer = nn.Embedding(total_items, embed_dim)
        self.context_embedding_layer = nn.Embedding(total_words, embed_dim)
        self.output_layer_layer = nn.Linear(embed_dim, total_words)

        item_emb_init_range = (2.0 / (total_items + embed_dim)) ** 0.5
        self.item_embedding_layer.weight.data.uniform_(-item_emb_init_range, item_emb_init_range)

        word_emb_init_range = (2.0 / (total_words + embed_dim)) ** 0.5
        self.context_embedding_layer.weight.data.uniform_(-word_emb_init_range, word_emb_init_range)

    def forward(self, context_ids, item_id, negative_samples_ids):
        # context_ids : (-1, num_of_context_ids)
        # item_id : (-1, 1)
        # negatvie_samples_ids : (-1, negative_samples_ids)

        item_id_embedding = self.item_embedding_layer(item_id)
        item_id_embedding = torch.unsqueeze(item_id_embedding, axis=0)

        context_ids = torch.stack(context_ids)
        context_ids_embedding = self.context_embedding_layer(context_ids)
        
        total_embedding = torch.cat([item_id_embedding, context_ids_embedding], axis=0)


        mean_embedding = torch.mean(total_embedding, axis=0, keepdim=True)       
        
        flat_mean_embedding = torch.reshape(mean_embedding, (-1, self.embed_dim))
        
        outrage = self.output_layer_layer(flat_mean_embedding)
        #outrage = F.log_softmax(self.output_layer_layer(flat_mean_embedding), dim=1)
        #print(outrage.shape)
        #return F.softmax(outrage, dim=1)
        return F.log_softmax(outrage)

    def criterion_and_loss(self):
        self.criterion = nn.CrossEntropyLoss() # .to(device)
        self.optimizer = torch.optim.SGD(Paragraph2Vec_DM.parameters(), lr=1e-2, momentum=0.9)
        self.lr_sche = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.9)
    
    def predict_item_embedding(self, item_id):
        self.item_embedding_layer(item_id)

    def predict_word_embedding(self, word_id):
        self.context_embedding_layer(word_id)

        
from data_utils import Dataset_Generator, Train_Dataset
from torch.utils.data import Dataset, DataLoader

def test():
    items_data_file = "/my_custom_dataset/items.csv"
    dataset = Dataset_Generator(items_data_file)
    train_dataset = Train_Dataset(dataset)
    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=2)
                            #shuffle=True) 
    
    for i, d in enumerate(train_loader):
        print(d) 
        sample_data = d
        context = sample_data.context
        product = sample_data.product
        target = sample_data.target

        model = Paragraph2Vec_DM(3,4000,4000)
        model(context,product,target)
        break

 
if __name__ == "__main__":
    test()    

