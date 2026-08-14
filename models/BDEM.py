import torch
import torch.nn as nn
import torch.nn.functional as F


class BDEM(nn.Module):
    """
    Boundary Detail Enhancement Module

    Use Sobel-guided edge response to enhance crack boundary
    and fine-grained crack features.
    """

    def __init__(self, channels=8):
        super(BDEM, self).__init__()

        # Detail feature branch
        self.detail_branch = nn.Sequential(
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

        # Convert edge magnitude to learnable attention
        self.edge_proj = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                bias=True
            ),
            nn.Sigmoid()
        )

        # Fixed Sobel kernels
        sobel_x = torch.tensor(
            [[-1., 0., 1.],
             [-2., 0., 2.],
             [-1., 0., 1.]]
        )

        sobel_y = torch.tensor(
            [[-1., -2., -1.],
             [0.,   0.,  0.],
             [1.,   2.,  1.]]
        )

        self.register_buffer(
            "sobel_x",
            sobel_x.view(1, 1, 3, 3)
        )

        self.register_buffer(
            "sobel_y",
            sobel_y.view(1, 1, 3, 3)
        )

    def get_edge(self, x):
        c = x.shape[1]

        kernel_x = self.sobel_x.repeat(c, 1, 1, 1)
        kernel_y = self.sobel_y.repeat(c, 1, 1, 1)

        gx = F.conv2d(
            x,
            kernel_x,
            padding=1,
            groups=c
        )

        gy = F.conv2d(
            x,
            kernel_y,
            padding=1,
            groups=c
        )

        edge = torch.sqrt(
            gx.pow(2) + gy.pow(2) + 1e-6
        )

        return edge

    def forward(self, x):
        # Explicit boundary information
        edge = self.get_edge(x)

        # Learn boundary attention
        edge_attention = self.edge_proj(edge)

        # Learn fine local details
        detail = self.detail_branch(x)

        # Boundary-guided residual enhancement
        out = x + detail * edge_attention

        return out