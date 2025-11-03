import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleAttention(nn.Module):
    def __init__(self, channels, levels=[2, 3, 6], detail_level=8):
        super().__init__()
        self.levels = levels
        self.detail_level = detail_level
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 16, channels, 1),
            nn.Sigmoid()
        )
        self.local_atts = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d((scale, scale)),
                nn.Conv2d(channels, channels // 8, 1),
                nn.ReLU(),
                nn.Conv2d(channels // 8, channels, 1),
                nn.Sigmoid()
            ) for scale in levels
        ])
        self.detail_enhance = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
        self.enable_attention_check = False

    def forward(self, x):
        B, C, H, W = x.shape
        global_att_map = self.global_att(x)
        local_weights = []
        for att in self.local_atts:
            feat = att(x)
            feat_up = F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=False)
            local_weights.append(feat_up)
        local_att_map = torch.mean(torch.stack(local_weights), dim=0)
        detail_enhance_map = self.detail_enhance(x)
        return x * (global_att_map + local_att_map + detail_enhance_map)
