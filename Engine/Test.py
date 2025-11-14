import torch
from ChessNet import ChessNet
from torch.nn import MSELoss 
import pandas as pd
from ChessDataset import ChessDataset
from torch.utils.data import DataLoader

net = ChessNet(num_res_blocks=4)
net.load_state_dict(torch.load("chessnet_weights.pth"))
net.eval()

df = pd.read_csv("testing_data.csv")
df['eval'] = df['eval'].clip(-1000, 1000) / 1000.0

test_dataset = ChessDataset(df)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

loss_fn = MSELoss()
test_loss = 0.0

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        output = net(x_batch)
        test_loss += loss_fn(output, y_batch).item() * x_batch.size(0)

test_loss /= len(test_dataset)

print(f"🎯 Final Test Loss: {test_loss:.6f}")
