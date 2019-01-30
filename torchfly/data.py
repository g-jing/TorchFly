import time
import torch
from torch.utils.data import Dataset, DataLoader

from typing import Optional


class DataBunch():
    "Bind train_dl, valid_dl and test_dl in a data object"

    def __init__(self,
                 train_dl: DataLoader,
                 valid_dl: Optional[DataLoader] = None,
                 test_dl: Optional[DataLoader] = None
                 ):
        "Take raw dataloader as input"
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.test_dl = test_dl

    @property
    def empty_val(self)->bool:
        "Check if validation set is empty"
        if self.valid_dl is None:
            return True
        return len(self.valid_ds) == 0

    @property
    def train_ds(self)->Dataset:
        "Return Dataloader's Dataset"
        return self.train_dl.dataset

    @property
    def valid_ds(self)->Dataset:
        "Return Dataloader's Dataset"
        return self.valid_dl.dataset

    @property
    def test_ds(self)->Dataset:
        "Return Dataloader's Dataset"
        return self.test_dl.dataset

    @property
    def batch_size(self):
        return self.train_dl.batch_size

    @batch_size.setter
    def batch_size(self, num: int):
        self.train_dl.batch_size = num
        if self.valid_dl is not None:
            self.valid_dl.batch_size = num
        if self.test_dl is not None:
            self.test_dl.batch_size = num

    @property
    def num_workers(self)->int:
        return self.train_dl.num_workers

    @num_workers.setter
    def num_workers(self, num: int):
        self.train_dl.num_workers = num
        if self.valid_dl is not None:
            self.valid_dl.num_workers = num
        if self.test_dl is not None:
            self.test_dl.num_workers = num

    def benchmark(self):
        "This is to decide the number of workers needed"
        start = time.time()
        for _ in self.train_dl:
            pass
        duration = int((time.time() - start) * 1000)
        print("One epoch through Training DataLoader")
        print("Time: ", duration, " ms")