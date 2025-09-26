import torch
from ChessNet import ChessNet
from torch.nn import SmoothL1Loss 
import pandas as pd
from ChessDataset import ChessDataset
from torch.utils.data import DataLoader

net = ChessNet()
net.load_state_dict(torch.load("chessnet_weights.pth"))
net.eval()

df = pd.read_csv("positions.csv")
df['eval'] = df['eval'].clip(-1000, 1000) / 1000.0
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df = df[9000:10000]

test_dataset = ChessDataset(df)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

loss_fn = SmoothL1Loss()
test_loss = 0.0

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        output = net(x_batch)
        test_loss += loss_fn(output, y_batch).item() * x_batch.size(0)  # sum over batch

test_loss /= len(test_dataset)

print(f"🎯 Final Test Loss: {test_loss:.6f}")
