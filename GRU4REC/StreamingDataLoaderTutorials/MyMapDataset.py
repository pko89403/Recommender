from torch.utils.data import Dataset, IterableDataset, DataLoader
from itertools import islice


class MyMapDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == "__main__":
    data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    map_dataset = MyMapDataset(data)
    loader = DataLoader(map_dataset, batch_size=4)
    for batch in loader:
        print(batch)
