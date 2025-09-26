import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from ChessNet import ChessNet
from torch.nn import SmoothL1Loss 
from ChessDataset import ChessDataset


# ----------------------------
# Load & Preprocess Data
# ----------------------------
df = pd.read_csv('positions.csv', usecols=['board', 'turn', 'castling', 'en_passant', 'eval'])
df.dropna(inplace=True)
df['eval'] = df['eval'].clip(-1000, 1000) / 1000.0

train_df, test_df = train_test_split(df, train_size=0.8, random_state=42, shuffle=True)

train_dataset = ChessDataset(train_df)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)

# ----------------------------
# Model, Optimizer, Loss
# ----------------------------
net = ChessNet()
optimizer = AdamW(net.parameters(), lr=1e-4, weight_decay=5e-4)
loss_fn = SmoothL1Loss()

# ----------------------------
# Training Loop
# ----------------------------
epochs = 30
for epoch in range(epochs):
    total_loss = 0.0
    net.train()
    for step, (boards, evals) in enumerate(train_loader, start=1):
        pred = net(boards)
        loss = loss_fn(pred, evals)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        print(f"\rEpoch {epoch+1}/{epochs} | Step {step}/{len(train_loader)} | Loss: {loss.item():.6f}", end='')

    avg_loss = total_loss / len(train_loader)
    print(f"\nEpoch {epoch+1}/{epochs} | Average Loss: {avg_loss:.6f}")

# ----------------------------
# Save Model
# ----------------------------
torch.save(net.state_dict(), "chessnet_weights.pth")
print("✅ Weights saved to chessnet_weights.pth")

