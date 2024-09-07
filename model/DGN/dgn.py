import torch
import torch.nn as nn
import torch.nn.functional as F
from model.DGN.knn import get_graph_feature


class DGN(nn.Module):
    def __init__(self, in_channel, out_channel=256, k=5):
        super(DGN, self).__init__()
        self.edgeConv1 = EdgeConv(in_channel, 64, k)
        self.edgeConv2 = EdgeConv(64, 64, k)
        self.edgeConv3 = EdgeConv(64, 64, k)

        self.conv = nn.Conv1d(64, 1024, kernel_size=1)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.norm = nn.BatchNorm1d(1024)

        self.mlp1 = nn.Sequential(
            nn.Conv1d(2 * 64 + 1024, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5)
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5)
        )
        self.mlp3 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5)
        )
        self.mlp4 = nn.Conv1d(128, out_channel, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x Tensor(B, C, N): 归一化过的点云

        Returns:
            Tensor(B, out_channel, N)
        """
        B = x.shape[0]
        N = x.shape[-1]
        x1 = self.edgeConv1(x)
        x2 = self.edgeConv2(x1)
        x3 = self.edgeConv3(x2)

        x4 = self.conv(x3)  # [B, 1024, N]
        x4 = self.relu(x4)
        x4 = self.norm(x4)
        globalFeat = F.adaptive_max_pool1d(x4, 1).view(B, -1)  # [B, 1024]
        globalFeat = globalFeat.unsqueeze(-1).repeat(1, 1, N)  # [B, 1024, N]
        x = torch.cat([x2, x3, globalFeat], dim=1)  # [B, 2 * 64 + 1024, N]

        x = self.mlp1(x)  # [B, 256, N]
        x = self.mlp2(x)  # [B, 256, N]
        x = self.mlp3(x)  # [B, 128, N]
        return self.mlp4(x)


class EdgeConv(nn.Module):
    def __init__(self, in_channel, out_channel, k):
        super(EdgeConv, self).__init__()
        self.k = k
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(64, out_channel, kernel_size=1, bias=False)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.norm1 = nn.BatchNorm2d(64)
        self.norm2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        """
        Args:
            x Tensor[B, C, N]: 归一化过的点云

        Returns:
            Tensor[B, C, N]
        """
        x = get_graph_feature(x, self.k)  # [B, N, k, C]
        x = x.permute(0, 3, 2, 1)  # [B, C, k, N]

        x = self.conv1(x)
        x = self.relu(x)
        x = self.norm1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.norm1(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.norm2(x)

        x = x.max(dim=2, keepdim=False)[0]
        return x

