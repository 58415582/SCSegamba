'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui@ieee.org
'''

import torch.nn as nn
import torch
from models.GBC import GBC, BottConv
from models.DySample import DySample
from models.BDEM import BDEM
from models.HNSM import HNSM

class MLP(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = self.proj(x)
        return x

class MFS(nn.Module):
    def __init__(self, embedding_dim, args=None):
        super(MFS, self).__init__()

        self.embedding_dim = embedding_dim
        self.use_bdem = getattr(args, 'use_bdem', False)
        self.use_hnsm = getattr(args, 'use_hnsm', False)
        self.linear_c4 = MLP(input_dim=128, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=64, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=32, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=16, embed_dim=embedding_dim)
        self.GBC_C = GBC(embedding_dim*4)
        self.GBC_8 = GBC(8, norm_type='IN')
        self.GN_C = nn.GroupNorm(num_channels=embedding_dim*4, num_groups=embedding_dim*4//16)
        self.linear_fuse = BottConv(embedding_dim*4, embedding_dim, embedding_dim//8, kernel_size=1, padding=0, stride=1)

        # Dual parallel enhancement modules
        if self.use_bdem:
            self.BDEM = BDEM(embedding_dim)

        if self.use_hnsm:
            self.HNSM = HNSM(embedding_dim)

        # if self.use_bdem and self.use_hnsm:
        #     self.dual_fuse = nn.Sequential(
        #         nn.Conv2d(
        #             embedding_dim * 2,
        #             embedding_dim,
        #             kernel_size=1,
        #             bias=False
        #         ),
        #         nn.GroupNorm(1, embedding_dim),
        #         nn.GELU(),

        #         nn.Conv2d(
        #             embedding_dim,
        #             embedding_dim,
        #             kernel_size=3,
        #             padding=1,
        #             groups=embedding_dim,
        #             bias=False
        #         ),

        #         nn.Conv2d(
        #             embedding_dim,
        #             embedding_dim,
        #             kernel_size=1,
        #             bias=False
        #         )
        #     )

        self.linear_pred = BottConv(embedding_dim, 1, 1, kernel_size=1)
        self.linear_pred_1 = nn.Conv2d(1, 1, kernel_size=1)
        self.dropout = nn.Dropout(p=0.1)

        self.DySample_C_2 = DySample(embedding_dim, scale=2)
        self.DySample_C_4 = DySample(embedding_dim, scale=4)
        self.DySample_C_8 = DySample(embedding_dim, scale=8)

    def forward(self, inputs):
        c4, c3, c2, c1 = inputs
        b, c, h, w = c4.shape
        out_c4 = self.linear_c4(c4.reshape(b, c, h*w).permute(0, 2, 1)).permute(0, 2, 1).reshape(b, self.embedding_dim, h, w)
        out_c4 = self.DySample_C_8(out_c4)

        b, c, h, w = c3.shape
        out_c3 = self.linear_c3(c3.reshape(b, c, h*w).permute(0, 2, 1)).permute(0, 2, 1).reshape(b, self.embedding_dim, h, w)
        out_c3 = self.DySample_C_4(out_c3)

        b, c, h, w = c2.shape
        out_c2 = self.linear_c2(c2.reshape(b, c, h*w).permute(0, 2, 1)).permute(0, 2, 1).reshape(b, self.embedding_dim, h, w)
        out_c2 = self.DySample_C_2(out_c2)

        b, c, h, w = c1.shape
        out_c1 = self.linear_c1(c1.reshape(b, c, h*w).permute(0, 2, 1)).permute(0, 2, 1).reshape(b, self.embedding_dim, h, w)

        out_c = self.GBC_C(
            torch.cat(
                [out_c4, out_c3, out_c2, out_c1],
                dim=1
            )
        )

        # Original MFS fused feature
        out_c = self.linear_fuse(out_c)
        out_c = self.dropout(out_c)

        # --------------------------------
        # Serial dual-module enhancement: BDEM -> HNSM
        # --------------------------------

        # Branch 1: crack boundary/detail enhancement
        # --------------------------------
        # Optional enhancement modules
        # --------------------------------

        if self.use_bdem and self.use_hnsm:
            out_c = self.BDEM(out_c)
            out_c = self.HNSM(out_c)

        elif self.use_bdem:
            out_c = self.BDEM(out_c)

        elif self.use_hnsm:
            out_c = self.HNSM(out_c)

        # If both are False:
        # out_c remains the original SCSegamba feature

        x = self.linear_pred_1(
            self.linear_pred(out_c)
        )

        return x
