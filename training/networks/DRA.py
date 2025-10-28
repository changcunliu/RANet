import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicRankProjection(nn.Module):
    def __init__(self, in_channels, base_rank=64, max_rank=1024):
        super().__init__()
        self.base_rank = base_rank
        self.max_rank = max_rank
        self.rank_controller = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 4, 1),
            nn.ReLU(),
            nn.Conv2d(4, 1, 1),
            nn.Sigmoid()
        )
        self.U = nn.Parameter(torch.randn(in_channels, max_rank))
        self.V = nn.Parameter(torch.randn(max_rank, in_channels))
        nn.init.orthogonal_(self.U)
        nn.init.orthogonal_(self.V)
        self.norm = nn.InstanceNorm2d(in_channels)
        self.higher_rank_module = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        rank_ratio = self.rank_controller(x).view(B, 1, 1)
        current_rank = self.base_rank + (self.max_rank - self.base_rank) * rank_ratio
        W_proj = self.U[:, :int(current_rank.mean().item())] @ self.V[:int(current_rank.mean().item()), :]
        x_flat = x.view(B, C, -1)
        x_proj = torch.einsum('bcn,cd->bdn', x_flat, W_proj)
        x_higher = self.higher_rank_module(x)
        x_proj1 = x_proj + x_higher.view(B, C, -1)
        return x_proj1.view(B, C, H, W)