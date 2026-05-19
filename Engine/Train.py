import torch
import os
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from ChessNet import ChessNet
from ChessDataset import ChessDataset

# ----------------------------
# Load & Preprocess Data
# ----------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))

train_dataset = ChessDataset(
    os.path.join(base_dir, "..", "dataset", "train_tensors.npy"),
    os.path.join(base_dir, "..", "dataset", "train_labels.npy"),
)
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=4, pin_memory=True)

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
        pred = net(boards)
        loss = loss_fn(pred, evals)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        print(f"\rstep {step}/{len(train_loader)} | Loss: {loss.item():.3f} | Pred: {pred[0].item():3f} | Actual: {evals[0].item():3f}", end=' ')

    avg_loss = total_loss / len(train_loader)
    print(f"\nEpoch {epoch+1}/{epochs} | Average Loss: {avg_loss:.6f}")

# ----------------------------
# Save Model
# ----------------------------
torch.save(net.state_dict(), "chessnet_weights.pth")
print("DONE")