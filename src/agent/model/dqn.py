import torch
import torch.nn as nn
import torch.nn.functional as F

# Number of one-hot channels: exponents 0..15 (empty=0, 2^1..2^15).
# 2048 == 2^11, so 16 channels leaves headroom well past the win tile.
NUM_CHANNELS = 16


class DQN(nn.Module):
    """
    Deep Q-Network for 2048.

    Input : a raw 4x4 board (values 0, 2, 4, ... ) as a float/int tensor of
            shape [B, 4, 4] (or [4, 4] for a single state).
    Output: Q-values for the 4 actions (UP, DOWN, LEFT, RIGHT), shape [B, 4].

    The board is encoded internally as a 16-channel one-hot of tile exponents
    (log2). One-hot encoding is the single most important design choice for a
    2048 CNN: it lets the network treat each tile magnitude as a distinct
    feature instead of forcing it to learn a non-linear log mapping from a
    single scalar channel. No BatchNorm / Dropout is used, so eval() and
    train() behave identically and single-sample inference is well-defined.
    """

    def __init__(self, num_channels: int = NUM_CHANNELS, dueling: bool = True):
        super().__init__()
        self.num_channels = num_channels
        self.dueling = dueling
        # 4x4 -> 3x3 -> 2x2 spatial reduction with growing feature depth.
        self.conv1 = nn.Conv2d(num_channels, 128, kernel_size=2)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=2)
        self.fc1 = nn.Linear(128 * 2 * 2, 256)
        if dueling:
            # Dueling heads: Q(s,a) = V(s) + [A(s,a) - mean_a A(s,a)].
            # Separating the state value from per-action advantages lets the net
            # represent "all four moves are similar but this one is slightly
            # better", which a plain Q-head struggles with on 2048 (there Q(s,a)
            # collapses toward V(s) and the greedy policy degenerates to random).
            self.value_head = nn.Linear(256, 1)
            self.adv_head = nn.Linear(256, 4)
        else:
            self.fc2 = nn.Linear(256, 4)  # 4 actions

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Raw board [.., 4, 4] -> one-hot exponents [B, C, 4, 4] (float)."""
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = x.long()
        # exponent = log2(value); value 0 (empty) maps to exponent 0.
        exp = torch.zeros_like(x)
        nz = x > 0
        exp[nz] = torch.log2(x[nz].float()).round().long()
        exp = exp.clamp_(0, self.num_channels - 1)
        onehot = F.one_hot(exp, num_classes=self.num_channels)  # [B,4,4,C]
        return onehot.permute(0, 3, 1, 2).float()               # [B,C,4,4]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._encode(x)
        x = F.relu(self.conv1(x))   # -> [B, 128, 3, 3]
        x = F.relu(self.conv2(x))   # -> [B, 128, 2, 2]
        x = x.flatten(1)            # -> [B, 512]
        x = F.relu(self.fc1(x))
        if self.dueling:
            value = self.value_head(x)                        # [B, 1]
            adv = self.adv_head(x)                            # [B, 4]
            return value + adv - adv.mean(dim=1, keepdim=True)
        return self.fc2(x)
