import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim.sgd import SGD
from ChessNet import ChessNet
from torch.nn import MSELoss
from ChessDataset import ChessDataset

# ----------------------------
# Load & Preprocess Data
# ----------------------------
df = pd.read_csv('dataset\\smaller_train.csv', usecols=['FEN', 'eval'])


train_dataset = ChessDataset(df)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# ----------------------------
# Model, Optimizer, Loss
# ----------------------------
net = ChessNet()
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4)
loss_fn = MSELoss()

# ----------------------------
# Training Loop
# ----------------------------
epochs = 30
for epoch in range(epochs):
    total_loss = 0.0
    net.train()
    for step, (boards, evals) in enumerate(train_loader, start=1):
        pred = net(boards).squeeze()
        loss = loss_fn(pred, evals)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        print(f"\rstep {step}/{len(train_loader)} | Loss: {loss:.3f} | Pred: {pred[0]:3f} | Actual: {evals[0]:3f}", end=' ')

    avg_loss = total_loss / len(train_loader)
    print(f"\nEpoch {epoch+1}/{epochs} | Average Loss: {avg_loss:.6f}")

# ----------------------------
# Save Model
# ----------------------------
torch.save(net.state_dict(), "chessnet_weights.pth")
print("DONE")