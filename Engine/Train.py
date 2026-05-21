import torch
import numpy as np
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from ChessNet import ChessNet
import torch.nn as nn

base_dir = Path(__file__).resolve().parent
weights_path = base_dir.parent / "chessnet_weights.pth"
training_data = np.load(base_dir.parent / "dataset" / "train.npz")
x_train, y_train = training_data["x"], training_data["y"]

x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)

train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ChessNet().to(device)
if weights_path.exists():
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f"Loaded existing weights from {weights_path}")
else:
    print("No existing weights found. Starting from scratch.")

algo = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

epochs = 10

for epoch in range(epochs):
    model.train()

    total_loss = 0
    total_cp_loss = 0
    total_samples = 0

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        pred = model(xb)
        loss = algo(pred, yb)

        loss.backward()
        optimizer.step()

        batch_size = xb.size(0)
        total_loss += loss.item() * batch_size

        total_samples += batch_size

    avg_loss = total_loss / total_samples

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"MSE: {avg_loss:.6f} | "
    )
    torch.save(model.state_dict(), weights_path)
print("DONE, Weights saved")
