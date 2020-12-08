import random
from itertools import chain, cycle, islice
from torch.utils.data import Dataset, IterableDataset, DataLoader
import torch 
import time 

class CustomIterableDatasetV1(IterableDataset):
    def __init__(self, filename):
        self.filename = filename
    
    def preprocess(self, text):
        ### Do something with text here
        text_app = text.lower().strip()
        ###
        return text_app

    def line_mapper(self, line):
        # splits the line into text and label and applies preprocessing to the text
        text, label = line.split(',')
        text = self.preprocess(text)

        return text, label 
    
    def __iter__(self):
        # Create an Iterator
        file_iter =  open(self.filename)
        # Map each element using the line_mapper
        mapped_itr = map(self.line_mapperm file_iter)

        return mapped_itr

if __name__ == "__main__":
    dataset = CustomIterableDatasetV1()
    dataloader = DataLoader(dataset, batch_size=64)

    for X, y in dataloader:
        print(len(X))
        print(y.shape)

        # Do Something with X, y