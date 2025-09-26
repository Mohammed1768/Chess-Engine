import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessNet(nn.Module):
    def __init__(self, num_res_blocks=4, filters=128):
        super(ChessNet, self).__init__()
        
        # Initial convolution - process the 18 input planes
        self.conv_input = nn.Conv2d(18, filters, 3, padding=1)
        self.bn_input = nn.BatchNorm2d(filters)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(filters, filters) for _ in range(num_res_blocks)
        ])
        
        # Policy head (optional - for move prediction)
        self.conv_policy = nn.Conv2d(filters, 32, 3, padding=1)
        self.bn_policy = nn.BatchNorm2d(32)
        self.fc_policy = nn.Linear(32 * 8 * 8, 1839)  # 1839 possible moves in chess
        
        # Value head (position evaluation)
        self.conv_value = nn.Conv2d(filters, 32, 3, padding=1)
        self.bn_value = nn.BatchNorm2d(32)
        self.fc_value1 = nn.Linear(32 * 8 * 8, 128)
        self.fc_value2 = nn.Linear(128, 1)
        
    def forward(self, x):
        # Input shape: (batch, 18, 8, 8)
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # Residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Value head
        value = F.relu(self.bn_value(self.conv_value(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.fc_value1(value))
                
        return self.fc_value2(value)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        out = F.relu(out)
        return out