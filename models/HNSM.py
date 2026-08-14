import torch
import torch.nn as nn


class HNSM(nn.Module):
    """
    Hard-Negative Suppression Module

    Compare local texture and larger-context representation.
    Regions with strong local response but inconsistent context
    are treated as potential hard-negative regions.
    """

    def __init__(self, channels=8):
        super(HNSM, self).__init__()

        # Local texture branch
        self.local_branch = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False
            ),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                bias=False
            ),
            nn.GroupNorm(1, channels),
            nn.GELU()
        )

        # Larger-context branch
        # dilation=2 -> effective receptive field = 5x5
        self.context_branch = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                groups=channels,
                bias=False
            ),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                bias=False
            ),
            nn.GroupNorm(1, channels),
            nn.GELU()
        )

        # Hard-negative gate
        self.gate = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                bias=True
            ),
            nn.Sigmoid()
        )

    def forward(self, x):

        local_feat = self.local_branch(x)
        context_feat = self.context_branch(x)

        # Local-context inconsistency
        difference = torch.abs(
            local_feat - context_feat
        )

        # Higher gate = stronger local/context inconsistency
        hard_gate = self.gate(difference)

        # Normal regions retain local detail.
        # Suspected hard-negative regions rely more on context.
        corrected = (
            (1.0 - hard_gate) * local_feat
            + hard_gate * context_feat
        )

        out = x + corrected

        return out