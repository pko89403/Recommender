import pandas as pd 
import numpy as np 
import torch 
from pathlib import Path 

base_path = Path(__file__).parent
file_path = (base_path / "../Dataset/processed/train_valid.txt").resolve()

class Dataset(object):
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path, sep='\t') # SessionID\titemID\tCategory\ttime        
        self.session_key = "SessionID"
        self.item_key = "itemID"
        self.category_key = "Category"
        self.time_key = "time"

        self.add_item_indices(item_map=None) # Add index to itemID
        self.data.sort_values([self.session_key, self.time_key], inplace=True)

        print("Data")
        print(self.data)
        print("*"*100)
        self.click_offsets = self.get_click_offset() # 각 세션의 오프셋(배치 데이터셋에서의 위치)들을 남긴다.

        print(self.click_offsets)
        self.session_idx_arr = self.order_session_idx()
        print(self.session_idx_arr) # 각 세션 아이디를 인덱싱 했다.

        print(len(self.click_offsets), len(self.session_idx_arr))

    def add_item_indices(self, item_map=None):
        if item_map is None:
            item_ids = self.data[self.item_key].unique()
            item2idx = pd.Series(data = np.arange(len(item_ids)),
                                 index=item_ids)
            item_map = pd.DataFrame({self.item_key: item_ids,
                                    'item_idx': item2idx[item_ids].values})

        self.item_map = item_map
        self.data = pd.merge(self.data, self.item_map, on=self.item_key, how='inner')
    
    def get_click_offset(self):
        """
            size of session key set
            offset is cumulative sum of the size of each session_id
        """
        offsets = np.zeros(self.data[self.session_key].nunique() + 1, dtype=np.int32)
        # nunique() - number of unique values

        offsets[1:] = self.data.groupby(self.session_key).size().cumsum()
        return offsets 
    
    def order_session_idx(self):
        session_idx_arr = np.arange(self.data[self.session_key].nunique())
        return session_idx_arr

class DataLoader():
    def __init__(self, dataset, batch_size=50):
        self.dataset = dataset 
        self.batch_size = batch_size
    
    def __iter__(self):
        data = self.dataset.data
        click_offsets = self.dataset.click_offsets
        session_idx_arr = self.dataset.session_idx_arr 

        iters = np.arange(self.batch_size)
        maxiter = iters.max()
        start = click_offsets[session_idx_arr[iters]] # 각 세션 아이디 인덱스 마다의 시작 오프셋을 가지고 온다
        end = click_offsets[session_idx_arr[iters] + 1] # 각 세션 아이디 인덱스 +1 이 각 세션 아이디 인덱스 마다의 끝 오프셋이다
        mask = [] # 한 세션이 끝남을 표현하는 마스크
        finished = False

        print(start, end, mask)
        while not finished:
            min_len = (end - start).min()
            # 첫번째 세션이 시작할 때의 클릭 인덱스들
            idx_target = data.item_idx.values[start]

            for i in range(min_len -1):
                # input 과 target을 만든다
                idx_input = idx_target
                idx_target = data.item_idx.values[start + i + 1]
                input = torch.LongTensor(idx_input)
                target = torch.LongTensor(idx_target)
                yield input, target, mask # well it actually returns
            
            # 두번째 세션의 마지막 원소일 때의 클릭 인덱스들
            start = start + (min_len - 1)
            mask = np.arange(len(iters))[(end - start) <= 1]
            for idx in mask:
                maxiter += 1
                if maxiter >= len(click_offsets) -1:
                    finished = True
                    break 

                # 다음 starting / ending point를 업데이트한다
                iters[idx] = maxiter 
                start[idx] = click_offsets[session_idx_arr[maxiter]]
                end[idx] = click_offsets[session_idx_arr[maxiter] + 1]

if __name__ == "__main__":
    file_path = (base_path / "../Dataset/processed/train_valid.txt").resolve()
    dataset = Dataset(data_path=file_path)
    dataloader = DataLoader(dataset=dataset, batch_size=50)
    dataloader.__iter__()