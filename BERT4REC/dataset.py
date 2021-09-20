import random

import pandas as pd
import torch 
from torch.utils.data import Dataset

from utils import (
    generate_random_mask, 
    pad_list
)

PADDING_INDEX = 0 
MASK_INDEX = 1 

class MovieLens(Dataset):
    def __init__(self, ratings_dir, mode, seq_len, valid_sample_size, masking_rate):
        self.ratings_dir = ratings_dir
        self.mode = mode 
        self.seq_len = seq_len
        self.valid_sample_size = valid_sample_size
        self.masking_rate = masking_rate        
        self.user_group, self.user2idx, self.item2idx = self._preprocess()



    def _generate_sample(self, df : pd.DataFrame):
        '''
            모델에 들어갈 랜덤 데이터 샘플을 생성한다.
        '''
        start_index = 0 
        end_index = df.shape[0]

        if self.mode == 'train':
            end_index = random.randint(self.valid_sample_size, end_index-self.valid_sample_size)
        if self.mode in ['valid', 'test']:
            end_index = end_index

        start_index = max(0, end_index - self.seq_len)

        return df[start_index:end_index]


    def _preprocess(self):
        df = pd.read_csv(self.ratings_dir)

        # 유저와 아이템 인덱스 매핑을 생성한다
        # 아이템의 인덱스는 2부터 시작한다. ( pad -> 0, mask -> 1 )
        user2idx = {v: i for i, v in enumerate(df['userId'].unique())}
        item2idx = {v: i+2 for i, v in enumerate(df['movieId'].unique())}

        df['userId'] = df['userId'].map(user2idx)
        df['movieId'] = df['movieId'].map(item2idx)

        # 유저 ID 별 시퀀스를 생성할 것이므로 timestamp를 기준으로 정렬한다
        df.sort_values(by='timestamp', inplace=True)

        
        # 시퀀스 데이터 생성을 위해 userId로 그룹화 한다.
        return df.groupby(by='userId'), user2idx, item2idx


    def __len__(self):
        return len(list(self.user_group.groups))


    def __getitem__(self, index):
        group = list(self.user_group.groups)[index]
        df = self.user_group.get_group(group)
        
        sample = self._generate_sample(df)

        
        target_sample = sample['movieId'].tolist()
        source_sample, mask = generate_random_mask(target_sample, 
                                self.mode,
                                self.valid_sample_size,
                                self.masking_rate, 
                                len(self.item2idx), 
                                MASK_INDEX)

        padding_mode = 'left'
        if self.mode == 'train':
            padding_mode = random.choice(['left', 'right'])
        else:
            padding_mode = 'left'


        padded_source = pad_list(source_sample, padding_mode, self.seq_len, PADDING_INDEX)
        padded_target = pad_list(target_sample, padding_mode, self.seq_len, PADDING_INDEX)        
        padded_mask = pad_list(mask, padding_mode, self.seq_len, False)        

        source_tensor = torch.tensor(padded_source, dtype=torch.long)
        target_tensor = torch.tensor(padded_target, dtype=torch.long)
        mask_tensor = torch.tensor(padded_mask, dtype=torch.bool)


        return source_tensor, target_tensor, mask_tensor
            