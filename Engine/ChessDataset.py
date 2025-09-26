from torch.utils.data import Dataset
from Encoder import Encoder
import torch

class ChessDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.encoder = Encoder()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        board_tensor = self.encoder.encode_FEN(row['board'], row['turn'], row['castling'], row['en_passant'])
        return torch.tensor(board_tensor, dtype=torch.float32), torch.tensor([row['eval']], dtype=torch.float32)
