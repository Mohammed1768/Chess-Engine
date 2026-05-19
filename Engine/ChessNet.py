import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        _x = F.relu(self.bn1(self.conv1(x)))
        _x = self.bn2(self.conv2(_x))
        _x += x
        return F.relu(_x)

class ChessNet(nn.Module):
    def __init__(self, input_channels=18, num_residual_blocks=5, num_filters=128):
        super().__init__()
        # Initial Convolution
        self.conv1 = nn.Conv2d(input_channels, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        
        self.res_blocks = nn.Sequential(
            * [ResidualBlock(num_filters) for _ in range(num_residual_blocks)]
        )
        
        self.val_conv = nn.Conv2d(num_filters, 32, kernel_size=1)  
        self.val_bn = nn.BatchNorm2d(32)
        
        self.fc1 = nn.Linear(32 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res_blocks(x)
        
        x = F.relu(self.val_bn(self.val_conv(x)))
        x = x.view(x.size(0), -1) # Flattens to (Batch_Size, 2048)
        
        x = F.relu(self.fc1(x))
        
        return torch.tanh(self.fc2(x))