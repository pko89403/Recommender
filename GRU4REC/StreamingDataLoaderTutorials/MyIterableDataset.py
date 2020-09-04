from torch.utils.data import Dataset, IterableDataset, DataLoader
class MyIterableDataset(IterableDataset):
    def __init__(self, data):
        self.data = data
    def __iter__(self):
        return iter(self.data)

if __name__ == "__main__":
    data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    iterable_dataset = MyIterableDataset(data)
    loader = DataLoader(iterable_dataset, batch_size=4)

    for batch in loader:
        print(batch)
