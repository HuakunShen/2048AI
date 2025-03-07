import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """
    Deep Q-Network for 2048 game
    Input: 4x4 game board state
    Output: Q-values for 4 possible actions (UP, DOWN, LEFT, RIGHT)
    """
    def __init__(self):
        super(DQN, self).__init__()
        # Input: 4x4 board
        self.conv1 = nn.Conv2d(1, 64, kernel_size=2, stride=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=2, stride=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        # Calculate the size after convolutions
        # Input: 4x4 -> After conv1: 3x3 -> After conv2: 2x2
        conv_output_size = 128 * 2 * 2
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 4)  # 4 actions (UP, DOWN, LEFT, RIGHT)
        
    def forward(self, x):
        # Convert input to log2 scale (since 2048 uses powers of 2)
        # Add 1 before log to handle zeros
        x = x + 1  # Add 1 to handle zeros
        x = torch.log2(x.float())  # Ensure input is float32
        
        # If input is a single state, add batch dimension
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            
        # Add channel dimension
        x = x.unsqueeze(1)  # Shape: [batch_size, 1, 4, 4]
        
        # Convolutional layers with batch normalization
        x = F.relu(self.bn1(self.conv1(x)))  # -> [batch_size, 64, 3, 3]
        x = F.relu(self.bn2(self.conv2(x)))  # -> [batch_size, 128, 2, 2]
        
        # Flatten
        x = x.view(x.size(0), -1)  # -> [batch_size, 128 * 2 * 2]
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x 