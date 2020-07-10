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


class TrainData(Dataset):

    def __init__(self):
        super().__init__()
        
        self.train_data = None
        self.len = 0

        self._data = OrderedDict()
        self._label = dict()

        # 모델 와꾸 생성에 사용되는 컬럼 명 들
        self.col_name = OrderedDict()

        # 전체 컬럼들 ,,, offset 등을 포함한다
        self.whole_col_name = OrderedDict()

        self.encoded_list = ["cust_grd_nm", "dvce_tp_cd", "emp_yn", "prd_brnd_nm", "prd_cd", "prd_tp_cat_vl", "sex_cd"]
        self.encoding_dict = self.read_encode_dict(prefix="./", cols=self.encoded_list)

        self.items_dataset = self.read_parquets("./item_meta").set_index('prd_cd')
        self.items_dataset['prd_cd'] = self.items_dataset.index
        
        self.users_dataset = self.read_parquets("./user_meta")


        self.users_dataset['label'] = 1

        self.total_dataset_length = len(self.users_dataset)


        self.dataset = self.users_dataset[['user_index', 'age', 'dvce_tp_cd',  'sex_cd', 'emp_yn', 'cust_grd_nm',
                                            'seq_cnt', 'prd_cd', 'prd_brnd_nm', 'prd_norm_prc', 'prd_tp_cat_vl', 
                                            'tg_prd_cd', 'tg_prd_brnd_nm', 'tg_prd_norm_prc', 'tg_prd_tp_cat_vl', 'label']]        
        self.feature_dict = dict()
        self.feature_dict['user'] = dict()
        self.feature_dict['product'] = dict()
        self.feature_dict['user']['dense'] = dict()
        self.feature_dict['user']['dense'] = dict()
        self.feature_dict['user']['sparse'] = dict()
        self.feature_dict['user']['sparse'] = dict()
        self.feature_dict['product']['dense'] = dict()
        self.feature_dict['product']['dense'] = dict()
        self.feature_dict['product']['sparse'] = dict()
        self.feature_dict['product']['sparse'] = dict()
    
        self.feature_dict['user']['dense']['single'] = ['age']
        self.feature_dict['user']['dense']['seq'] = []
        self.feature_dict['user']['sparse']['single'] = ['user_index', 'dvce_tp_cd', 'sex_cd', 'emp_yn' , 'cust_grd_nm']
        self.feature_dict['user']['sparse']['seq'] = []
        self.feature_dict['product']['dense']['single'] = ['tg_prd_cd', 'tg_prd_brnd_nm', 'tg_prd_tp_cat_vl']
        self.feature_dict['product']['dense']['seq'] = ['prd_norm_prc']
        self.feature_dict['product']['sparse']['single'] = ['tg_prd_norm_prc'] 
        self.feature_dict['product']['sparse']['seq'] = ['prd_cd', 'prd_brnd_nm', 'prd_tp_cat_vl'] 
        self.feature_dict['seq_cnt'] = ['seq_cnt']




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

    def negative_labels(self, neg_sample_cnt=1):
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

            # print(cur_seq)
            # print(self.users_dataset.loc[idx,:])
            for neg_prd_cd in neg_prd:            
                neg_item_info = self.items_dataset.loc[neg_prd_cd, :].copy()

                for idx in neg_item_info.index.tolist():
                    if 'tg_'+idx in cur_seq.index: cur_seq['tg_'+idx] = neg_item_info[idx]
                
                cur_seq['label'] = 0
                self.users_dataset.append(cur_seq)
                
    def __len__(self):
        return self.total_dataset_length
    
    def __getitem__(self, row_num):
        row = self.dataset.loc[row_num, :]

        first_layer = ['user', 'product']
        second_layer = ['dense', 'sparse']
        third_layer = ['single', 'seq']


        out = dict()
        out['dense'] = dict()
        out['sparse'] = dict()

        for f_l in first_layer:
            for s_l in second_layer:
                for t_l in third_layer:
                    if(len(self.feature_dict[f_l][s_l][t_l]) < 1):   continue
                    
                    for col in self.feature_dict[f_l][s_l][t_l]:                
                        out[s_l][col] = row[col] 
        
        out['dense']['offset'] = row['seq_cnt']
        out['sparse']['offset'] = row['seq_cnt']
        return  out['dense'], out['sparse'], row['label']


    def get_dataloader(self):
        batch_size = 128
        shffule = True
        num_workers = 0 
        drop_last = True

        dataloader = data.DataLoader(self,
            batch_size=batch_size
            shuffle=shuffle
            num_workers=num_workers
            drop_last=drop_last
        )

        return dataloader
