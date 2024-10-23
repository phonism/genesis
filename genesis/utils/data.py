import numpy as np
import gzip
import random

class Dataset:
    def __len__(self):
        raise NotImplementedError
    
    def __getitem__(self, idx):
        raise NotImplementedError


class IterableDataset(Dataset):
    def __iter__(self):
        raise NotImplementedError


class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        if isinstance(dataset, IterableDataset):
            self.is_iterable = True
        else:
            self.is_iterable = False
            self.indices = list(range(len(self.dataset)))

    def __iter__(self):
        if self.is_iterable:
            self.dataset_iter = iter(self.dataset)
        else:
            if self.shuffle:
                random.shuffle(self.indices)
            self.current_idx = 0
        return self

    def __next__(self):
        if self.is_iterable:
            batch = []
            try:
                for _ in range(self.batch_size):
                    batch.append(next(self.dataset_iter))
            except StopIteration:
                if not batch:
                    raise StopIteration
            return batch
        else:
            if self.current_idx >= len(self.indices):
                raise StopIteration
            batch_indices = self.indices[self.current_idx:self.current_idx + self.batch_size]
            batch = [self.dataset[i] for i in batch_indices]
            self.current_idx += self.batch_size
            return batch
