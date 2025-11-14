import torch
from torch.utils.data import Dataset
from pandas import DataFrame
from Encoder import Encoder

class ChessDataset(Dataset):
    def __init__(self, dataset: DataFrame):
        encoder = Encoder()
        self.x = [torch.tensor(encoder.encode_FEN(fen), dtype=torch.float32) for fen in dataset['FEN'].values]
        self.y = torch.tensor(dataset['eval'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):        
        return self.x[i], self.y[i]
