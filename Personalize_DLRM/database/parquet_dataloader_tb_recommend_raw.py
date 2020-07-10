# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
from torch.utils import data
import pandas as pd
import sys
import numpy as np
import os
from collections import OrderedDict
import pyarrow as pa 
import pyarrow.parquet as pq
import glob
import json
import random 
import pprint
from config.config import config as cf

class TrainData(Dataset):

    def __init__(self):
        super().__init__()
        
        self.train_data = None
        self.len = 0

        self._data = OrderedDict()
        self._label = dict()

        # 전체 컬럼들 ,,, offset 등을 포함한다
        self.whole_col_name = OrderedDict()

        self.encoded_list = ["cust_grd_nm", "dvce_tp_cd", "emp_yn", "prd_brnd_nm", "prd_cd", "prd_tp_cat_vl", "sex_cd"]
        self.encoding_dict = self.read_encode_dict(prefix="/Users/amore/ap-recsys-model/tb_recommend_raw", cols=self.encoded_list)

        self.items_dataset = self.read_parquets("/Users/amore/ap-recsys-model/tb_recommend_raw/item_meta").set_index('prd_cd')
        self.items_dataset['prd_cd'] = self.items_dataset.index        
        self.users_dataset = self.read_parquets("/Users/amore/ap-recsys-model/tb_recommend_raw/user_meta")


        self.users_dataset['label'] = 1

        
        self.negative_labels(path='/Users/amore/ap-recsys-model/tb_recommend_raw/neg_sample')
        self.total_dataset_length = len(self.users_dataset)

        self.dataset = self.users_dataset[['age', 'dvce_tp_cd',  'sex_cd', 'emp_yn', 'cust_grd_nm',
                                            'seq_cnt', 'prd_cd', 'prd_brnd_nm', 'prd_norm_prc', 'prd_tp_cat_vl', 
                                            'tg_prd_cd', 'tg_prd_brnd_nm', 'tg_prd_norm_prc', 'tg_prd_tp_cat_vl', 'label']]        

        self.first_layer = ['user', 'product']
        self.second_layer = ['dense', 'sparse']
        self.third_layer = ['single', 'seq']

        self.feature_dict = OrderedDict()
        self.feature_dict['user'] = OrderedDict()
        self.feature_dict['product'] = OrderedDict()
        self.feature_dict['user']['dense'] = OrderedDict()
        self.feature_dict['user']['dense'] = OrderedDict()
        self.feature_dict['user']['sparse'] = OrderedDict()
        self.feature_dict['user']['sparse'] = OrderedDict()
        self.feature_dict['product']['dense'] = OrderedDict()
        self.feature_dict['product']['dense'] = OrderedDict()
        self.feature_dict['product']['sparse'] = OrderedDict()
        self.feature_dict['product']['sparse'] = OrderedDict()
    
        self.feature_dict['user']['dense']['single'] = np.array(['age'])
        self.feature_dict['user']['dense']['seq'] = np.array([])
        self.feature_dict['user']['sparse']['single'] = np.array(['dvce_tp_cd', 'sex_cd', 'emp_yn' , 'cust_grd_nm'])
        self.feature_dict['user']['sparse']['seq'] = np.array([])
        self.feature_dict['product']['dense']['single'] = np.array(['tg_prd_norm_prc'])
        self.feature_dict['product']['dense']['seq'] = np.array(['prd_norm_prc'])
        self.feature_dict['product']['sparse']['single'] = np.array(['tg_prd_cd', 'tg_prd_brnd_nm', 'tg_prd_tp_cat_vl'])
        self.feature_dict['product']['sparse']['seq'] = np.array(['prd_cd', 'prd_brnd_nm', 'prd_tp_cat_vl'])
        # self.feature_dict['seq_cnt'] = np.array(['seq_cnt'])

        self.write_unique_file()
        self.sparse_col_len = (len(self.feature_dict['user']['sparse']['single']) 
                                + len(self.feature_dict['user']['sparse']['seq']) 
                                + len(self.feature_dict['product']['sparse']['single']) 
                                + len(self.feature_dict['product']['sparse']['seq'])) 
        self.col_name = self.feature_dict


        self.batch_size = cf().path["data"]["batch_size"]
        self.shuffle=cf().path["data"]["shuffle"]
        self.num_workers=cf().path["data"]["num_workers"]
        self.drop_last=True
            

    def read_encode_dict(self, prefix="./", cols=["cust_grd_nm", "dvce_tp_cd", "emp_yn", "prd_brnd_nm", "prd_cd", "prd_tp_cat_vl", "sex_cd"]):
        encoding_dict = dict()
        for col in cols:
            temp = self.read_parquets(os.path.join(prefix, col))
            temp_dict = dict(zip(temp[col], temp[col+"_index"]))
            encoding_dict[col] = temp_dict
            
        return encoding_dict


    def read_parquets(self, path):
        fList = glob.glob(path + "/*.parquet")
        data = [pd.read_parquet(f) for f in fList]
        
        
        if(len(data) > 1):
            merged_data = pd.concat(data, ignore_index=True)
        merged_data = data[0]

        return merged_data

    def write_parquets(self, df, path):
        df.to_parquet(path + "negative.parquet")

    def negative_labels(self, path='', neg_sample_cnt=1):
        if(path != ''):
            neg_sample_df = self.read_parquets(path)
            self.users_dataset = self.users_dataset.append(neg_sample_df, ignore_index=True)
            print(f"{len(neg_sample_df)} negative sample added. ")
        else:    
            neg_sample_df = pd.DataFrame(columns=self.users_dataset.columns)
            for idx in self.users_dataset.index:
                
                
                cur_seq = self.users_dataset.loc[idx,:].copy()
                prd_cd_seq = cur_seq['prd_cd']
                tg_prd_cd = cur_seq['tg_prd_cd']

                pos_prd = set(prd_cd_seq)
                pos_prd.add(str(tg_prd_cd))

                remains = neg_sample_cnt
                neg_prd = []
                
                # Negative Sampling Part
                while( remains > 0):
                    neg_sample = random.choices(population = self.items_dataset.index,
                                            weights=self.items_dataset.accum_prob,
                                            k=remains)
                    sample = list(set(neg_sample) - pos_prd)
                    neg_prd += sample
                    remains -= len(sample)


                for neg_prd_cd in neg_prd:            
                    neg_item_info = self.items_dataset.loc[neg_prd_cd, :].copy()

                    for idx in neg_item_info.index.tolist():
                        if 'tg_'+idx in cur_seq.index: cur_seq['tg_'+idx] = neg_item_info[idx]
                    
                    cur_seq['label'] = 0
                    neg_sample_df.loc[len(neg_sample_df)] = cur_seq
            
            neg_sample_df.to_parquet(path+"/neg_sample_df.parquet")
            self.users_dataset = self.users_dataset.append(neg_sample_df, ignore_index=True)
                
    def __len__(self):
        return self.total_dataset_length
    
    def __getitem__(self, row_num):
        row = self.dataset.loc[row_num, :]
        
        out = dict()
        out['dense'] = dict()
        out['sparse'] = dict()

        for f_l in self.first_layer:
            for s_l in self.second_layer:
                for t_l in self.third_layer:
                    if(len(self.feature_dict[f_l][s_l][t_l]) < 1):   continue
                    
                    for col in self.feature_dict[f_l][s_l][t_l]:
                        
                        if(t_l == 'seq'):
                            out[s_l][col] = row[col].astype(int)
                        else:
                            out[s_l][col] = int(row[col])
                        
                        
        
        # out['dense']['offset'] = row['seq_cnt']
        out['sparse']['offset'] = row['seq_cnt']
        
        # print(out['dense'])
        #pprint.pprint(out['sparse'])
        return  out['dense'], out['sparse'], row['label']


    def get_dataloader(self):
        batch_size = self.batch_size
        shuffle = self.shuffle
        num_workers = self.num_workers
        drop_last = self.drop_last

        dataloader = data.DataLoader(self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last
        )
        
        return dataloader

    def write_unique_file(self):
        users = dict()
        items = dict()

        for f_l in self.first_layer:
            for s_l in self.second_layer:
                for t_l in self.third_layer:
                    if(len(self.feature_dict[f_l][s_l][t_l]) < 1):   continue
                    
                    for feature in self.feature_dict[f_l][s_l][t_l]:
                        try:
                            if(f_l == 'user'):
                                users[feature] = len(self.encoding_dict[feature])
                            else:
                                items[feature] = len(self.encoding_dict[feature])
                                items['tg_'+feature] = len(self.encoding_dict[feature])
                            
                        except:
                            pass
        
        with open('config/unique_client_tb_recommend_raw.json', 'w') as cf:
            json.dump(users, cf)
        with open('config/unique_product_tb_recommend_raw.json', 'w') as pf:
            json.dump(items, pf)
            
test = TrainData()
d = test.get_dataloader()
