import torch
from torch.utils.data import Dataset
import numpy as np

class ChessDataset(Dataset):
    def __init__(self, tensors_path: str, labels_path: str):
        self.x = np.load(tensors_path, mmap_mode='r')
        self.y = np.load(labels_path, mmap_mode='r')

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):        
        x_numpy_int8 = self.x[i]
        
        x_tensor = torch.from_numpy(x_numpy_int8).float()
        y_tensor = torch.tensor(self.y[i], dtype=torch.float32).unsqueeze(-1)
        
        return x_tensor, y_tensor