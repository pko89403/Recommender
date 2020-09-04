import random
from itertools import chain, cycle, islice
from torch.utils.data import Dataset, IterableDataset, DataLoader
import torch 
import time 
class MyIterableDataset(IterableDataset):

    def __init__(self, data_list, batch_size):
        self.data_list = data_list
        self.batch_size = batch_size
    
    @property
    def shuffled_data_list(self):
        return random.sample(self.data_list, len(self.data_list))

    def process_data(self, data):
        for x in data:
            worker = torch.utils.data.get_worker_info()
            worker_id = worker.id if worker is not None else -1

            start = time.time()
            time.sleep(0.1)
            end = time.time()

            yield x, worker_id, start, end
    
    def get_stream(self, data_list):
        return chain.from_iterable(map(self.process_data, cycle(data_list)))

    def get_streams(self):
        return zip(*[self.get_stream(self.shuffled_data_list)
                     for _ in range(self.batch_size)])

    def __iter__(self):
        return self.get_streams()

data_list = [
    [10, 11, 12, 13],
    [20, 21, 22, 23],
    [30, 31, 32, 33],
    [40, 41, 42, 43],
    [50, 51, 52, 53],
    [60, 61, 62, 63],
    [70, 71, 72, 73],
    [80, 81, 82, 83],
    [90, 91, 92, 93],
]
iterable_dataset = MyIterableDataset(data_list, batch_size=4)
loader = DataLoader(iterable_dataset, batch_size=None, num_workers=2)

for batch in islice(loader, 1):
    print(batch)