import torch
import numpy as np
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from ChessNet import ChessNet
import torch.nn as nn

base_dir = Path(__file__).resolve().parent
training_data = np.load(base_dir.parent / "dataset" / "train.npz")
x_train, y_train = training_data["x"], training_data["y"]

x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)

train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

model = ChessNet()
algo = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

epochs = 20

for epoch in range(epochs):
    model.train()

    total_loss = 0
    total_cp_loss = 0
    total_samples = 0

    for xb, yb in train_loader:
        optimizer.zero_grad()

        pred = model(xb)

        loss = algo(pred, yb)

        loss.backward()
        optimizer.step()

        batch_size = xb.size(0)
        total_loss += loss.item() * batch_size

        pred_cp = 700 * torch.atanh(pred.clamp(-0.999, 0.999))
        true_cp = 700 * torch.atanh(yb.clamp(-0.999, 0.999))

        norm_loss = ((pred_cp - true_cp) ** 2).mean()

        total_cp_loss += norm_loss.item() * batch_size
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    norm_loss = (total_cp_loss / total_samples) ** 0.5

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"MSE: {avg_loss:.6f} | "
        f"CP RMSE: {norm_loss / 100.0:.2f}"
    )
torch.save(model.state_dict(), "chessnet_weights.pth")