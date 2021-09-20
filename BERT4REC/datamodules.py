from torch.utils.data.dataloader import DataLoader
from dataset import MovieLens
import pytorch_lightning as pl 

class MovieLen_Module(pl.LightningDataModule):
    def __init__(self, data_dir, seq_len, valid_sample_size, masking_rate, batch_size, shuffle):
        super().__init__()

        self.data_dir = data_dir
        self.seq_len = seq_len
        self.valid_sample_size = valid_sample_size
        self.masking_rate = masking_rate
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.train_dataset = MovieLens(ratings_dir=self.data_dir,
                                    mode='train',
                                    seq_len=self.seq_len,
                                    valid_sample_size=self.valid_sample_size, 
                                    masking_rate=self.masking_rate)
        self.val_dataset = MovieLens(ratings_dir=self.data_dir,
                                    mode='valid',
                                    seq_len=self.seq_len,
                                    valid_sample_size=self.valid_sample_size, 
                                    masking_rate=self.masking_rate)
        self.test_dataset = MovieLens(ratings_dir=self.data_dir,
                                    mode='test',
                                    seq_len=self.seq_len,
                                    valid_sample_size=self.valid_sample_size, 
                                    masking_rate=self.masking_rate)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
    