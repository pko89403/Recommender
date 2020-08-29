import pandas as pd 
import re
from konlpy.tag import Mecab
import numpy as np 
from numpy.random import choice
from collections import namedtuple

class Title_Tokenizer(object):
    def __init__(self):
        self.tokenizer = Mecab()
        self.rgx_list = ['[^\w]'] # will match anything that's not alphanumeric or underscore

    def clean_text(self, text):
        
        new_text = text.strip().lower()
        
        for rgx_match in self.rgx_list:
            new_text = re.sub(rgx_match, ' ', new_text)
        return new_text


    def tokenize_item_name(self, item_name):
        """
        tokenized_item_names = []

        for prd_nm in item_name:
            prd_nm = self.clean_text(prd_nm)
            prd_nm_tokens_mecab = self.tokenizer.morphs(prd_nm)
            #print(f"{prd_nm} -> {prd_nm_tokens_mecab}")

            tokenized_item_names.append(prd_nm_tokens_mecab)

        return tokenized_item_names
        """
        item_name_cleaned = self.clean_text(item_name)
        item_name_tokens = self.tokenizer.morphs(item_name_cleaned)
        return item_name_tokens


class Dataset_Generator(object):
    # https://github.com/inejc/paragraph-vectors/blob/33f6465208f738a5dac69810d709459cc99ed704/paragraphvec/data.py#L279
    def __init__(self, raw_data_path):

        self.raw_data = pd.read_csv(raw_data_path)

        self.dataset = self.raw_data[['prd_cd', 'prd_nm']].copy()
        self.dataset = self.dataset.drop_duplicates()
        
        self.prdnm_to_prdcd = {prdnm:prdcd for prdcd, prdnm in zip(self.dataset['prd_cd'], self.dataset['prd_nm'])}
        self.prdcd_to_prdnm = {prdcd:prdnm for prdnm, prdcd in self.prdnm_to_prdcd.items()}
        
        self.title_to_titleId, self.titleId_to_title, self.titleId_to_titleIdx, self.title_to_titleIdx, self.splitted_title = self.title_splitter()
        self.vocab_freq, self.word_to_index, self.index_to_word = self.build_vocab()

        self.vocab_size = len(self.vocab_freq)
        self._init_noise_dist()
    
    def title_splitter(self):
        title_splitter = Title_Tokenizer()
    
        
        #titleId_to_title = {i:v for i, v in enumerate(self.dataset['prd_nm'])}
        #title_to_titleId = {v:i for i, v in titleId_to_title.items()}

        titleId_to_title = {prdcd:prdnm for prdnm, prdcd in self.prdnm_to_prdcd.items()}
        title_to_titleId = {prdnm:prdcd for prdcd, prdnm in zip(self.dataset['prd_cd'], self.dataset['prd_nm'])}
        titleId_to_titleIdx = {}
        title_to_titleIdx = {}

        splitted_title = []

        idx = 0
        for titleId, title in titleId_to_title.items():
            
            tokenized_title = title_splitter.tokenize_item_name(title)
            splitted_title.append(tokenized_title)
            titleId_to_titleIdx[titleId] = idx
            title_to_titleIdx[title] = idx
            idx += 1
        #splitted_title = title_splitter.tokenize_item_name(self.dataset['prd_nm'])

        #print(f"Whole Item Title Count : {len(title_to_titleId)}")
        # print(title_to_titleId)

        return title_to_titleId, titleId_to_title, titleId_to_titleIdx, title_to_titleIdx, splitted_title
    
    def get_title_to_titleIdx(self):
        return self.title_to_titleIdx

    def get_titleId_to_titleIdx(self):
        return self.titleId_to_titleIdx

    def get_title_to_titleID(self):
        return self.title_to_titleId

    def get_titleID_to_title(self):
        return self.titleId_to_title

    def build_vocab(self):
        vocab_freq = {}
        for item_name in self.splitted_title:
            for name_token in item_name:
                try:
                    vocab_freq[name_token] +=1 
                except:
                    vocab_freq[name_token] = 1

        word_to_index = { w:i for i, w in enumerate(vocab_freq.keys())}
        index_to_word = { w:i for i, w in word_to_index.items()}
        
        return vocab_freq, word_to_index, index_to_word

    def _init_noise_dist(self):
        probs = np.zeros(self.vocab_size)


        for word, freq in self.vocab_freq.items():
            probs[self._word_to_index(word)] = freq

        probs = np.power(probs, 0.75)
        probs /= np.sum(probs)

        self.sample_noise = lambda: choice(
            probs.shape[0], 5, p=probs).tolist()

    def _word_to_index(self, word):
        return self.word_to_index[word]
    def _index_to_word(self, index):
        return self.index_to_word[index]

    def _title_as_indexes(self):
        title_splitted = self.splitted_title
        title_as_indexses = list()
        for words in title_splitted:
            #print(words)
            indexes = [self._word_to_index(w) for w in words]
            title_as_indexses.append(indexes)
        
        return title_as_indexses, self.word_to_index, self.index_to_word

class Train_Dataset(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.dataset_as_indexes, self.word_to_index, self.index_to_word = self.dataset._title_as_indexes()
        
        self.titleId_to_title = self.dataset.get_titleID_to_title()
        self.titleId_to_titleIdx = self.dataset.get_titleId_to_titleIdx()
        self.title_to_titleIdx = self.dataset.get_title_to_titleIdx()
        self.title_to_titleId = {v:k for k,v in self.titleId_to_title.items()}
        self.titleIdx_to_titleId = {v:k for k,v in self.titleId_to_titleIdx.items()}
        self.titleIdx_to_title = {v:k for k,v in self.title_to_titleIdx.items()}
        
        self.window_size = 3

        self.filt_dataset = self.filt_under_window()
        self.doc_id = 0
        self.train_dataset = self.create_xy()

    def filt_under_window(self):

        filt_dataset = []
        # titleId - title - titleIndex
        filtered_cnt = 0
        for i, v in enumerate(self.dataset_as_indexes):
            if( len(v) >= self.window_size):
                filt_dataset.append(v)
            else:
                #print("Cut - ", self.dataset.titleId_to_title[i], v)
                filtered_cnt += 1
                del_titleIdx = i
                del_title = self.titleIdx_to_title[del_titleIdx]
                del_title_id = self.title_to_titleId[del_title]
                
                #print(del_title, del_title_id, del_titleIdx)

                del self.titleId_to_title[del_title_id]
                del self.titleId_to_titleIdx[del_title_id]
                del self.title_to_titleIdx[del_title]
                del self.title_to_titleId[del_title]
                del self.titleIdx_to_titleId[i]
                del self.titleIdx_to_title[i]
                
        print('filtered Item count : ', filtered_cnt)
        #print( len(self.titleId_to_title))
        #print( len(self.titleId_to_titleIdx))
        #print( len(self.title_to_titleIdx))
        #print( len(self.title_to_titleId))      
        #print( len(self.titleIdx_to_titleId))      
        #print( len(self.titleIdx_to_title))      

        #print("REBUILD TITLE - TITLEID DICT")
        #print( max(self.titleId_to_title.keys()))
        titleIdx_to_title_update = {}
        title_to_titleIdx_update = {}
        for i, dict_val in enumerate(self.titleIdx_to_title.values()):
            titleIdx_to_title_update[i] = dict_val
            title_to_titleIdx_update[dict_val] = i

        self.titleIdx_to_title = titleIdx_to_title_update
        self.title_to_titleIdx = title_to_titleIdx_update
        #print('ReBuilded Title Count : ',max(self.titleId_to_title.keys()))

        return filt_dataset

    def _titleId_to_title(self, titleID):
        return self.titleId_to_title[titleID]

    def _titleIndex_to_title(self, titleIndex):
        return self.titleIdx_to_title[titleIndex]

    def create_xy(self):
        train_data = namedtuple("train_data", "product target context")
        
        train_dataset = []

        for product_index, title_as_indexes in enumerate(self.filt_dataset):
            title_length = len(title_as_indexes)
            for inner_index in range(0, title_length):
                start = inner_index
                end = inner_index + self.window_size
                added = []

                if( end >= title_length ):
                    added = title_as_indexes[0:end-title_length]
                window = title_as_indexes[start:end] + added

                x = window.pop(0) # x - input
                y = window # y - label                
                
                train_dataset.append(train_data(product=product_index, target=x, context=y))

        return train_dataset

    def __len__(self):
        return len(self.train_dataset)

    def __getitem__(self,idx):
        #label, input, product = self.train_dataset[idx].target, self.train_dataset[idx].context, self.train_dataset[idx].product
        return self.train_dataset[idx]

if __name__ == "__main__":
    items_data_file = "/my_custom_dataset/items.csv"

    #items = pd.read_csv(items_data_file)
    #items_name = items['prd_nm'].copy()
    #items_name = items_name.drop_duplicates()

    dataset = Dataset_Generator(items_data_file)
    train_loader = Train_Dataset(dataset)
    
    for i, j in enumerate(train_loader):
        #print( j.target, j.context, j.product)
        
        target_word = dataset._index_to_word(j.target)
        context_word = [dataset._index_to_word(t) for t in j.context]
        product_name = train_loader._titleIndex_to_title(j.product)
        
        print(target_word, context_word, product_name)
        

