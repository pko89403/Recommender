from torch.utils.data import Dataset, IterableDataset, DataLoader
from itertools import islice, cycle, chain

class MyIterableDataset(IterableDataset):
    def __init__(self, data_list):
        self.data_list = data_list
    
    def process_data(self, data):
        for x in data:
            yield x
    
    def get_stream(self, data_list):
        return chain.from_iterable(map(self.process_data, cycle(data_list)))
    
    def __iter__(self):
        return self.get_stream(self.data_list)

data_list = [
    [12, 13, 14, 15, 16, 17],
    [27, 28, 29],
    [31, 32, 33, 34, 35, 36, 37, 38, 39],
    [40, 41, 42, 43],
]

iterable_dataset = MyIterableDataset(data_list)
loader = DataLoader(iterable_dataset, batch_size=4)

for batch in islice(loader, 10):
    print(batch)