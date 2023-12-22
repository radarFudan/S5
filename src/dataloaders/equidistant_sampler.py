import random 

import torch
from torch.utils.data import Dataset, DataLoader

class EquidistantSampler(torch.utils.data.Sampler):
    def __init__(self, data_len, batch_size, shift=None):
        self.data_len = data_len
        self.batch_size = batch_size
        self.batch_len = data_len // batch_size
        # print("batch len", self.batch_len)
        self.shift_range = data_len - self.batch_len * batch_size
        # print("shift range", self.shift_range)
        self.shift = shift

    def __iter__(self):
        if self.shift:
            shift = random.randint(0, self.shift_range)
        else:
            shift = 0
        # print("Data shift", shift)
        for offset in range(self.batch_len):
            batch = []
            for i in range(offset, offset + self.batch_size * self.batch_len, self.batch_len):
                batch.append(i + shift)
            yield batch

    def __len__(self):
        return self.data_len // self.batch_len

# Example usage
class DummyDataset(Dataset):
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return index

if __name__ == "__main__":

    dataset = DummyDataset(23)
    batch_size=7

    batch_sampler = EquidistantSampler(len(dataset), batch_size)
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler, shuffle=False)

    for i, batch in enumerate(dataloader):
        print(batch)
    print("New epoch")
    for i, batch in enumerate(dataloader):
        print(batch)