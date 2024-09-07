import math

import torch
from torch import nn

from model.DGN.knn import get_graph_feature


class RSAttention(nn.Module):
    def __init__(self, in_channel, k):
        super(RSAttention, self).__init__()
        self.k = k
        # self.linear_x1 = nn.Conv1d(in_channel, in_channel, kernel_size=1, bias=False)
        # self.linear_xn = nn.Conv2d(in_channel, in_channel, kernel_size=1, bias=False)
        # self.wk = nn.Conv2d(in_channel, in_channel, kernel_size=1, bias=False)
        # self.wq = nn.Conv1d(in_channel, in_channel, kernel_size=1, bias=False)
        # self.wv = nn.Conv2d(in_channel, in_channel, kernel_size=1, bias=False)
        # self.soft = nn.Softmax(dim=-1)
        self.wk = nn.Conv1d(in_channel, in_channel // 2, kernel_size=1, bias=False)
        self.wq = nn.Conv1d(in_channel, in_channel // 2, kernel_size=1, bias=False)
        self.wv = nn.Conv1d(in_channel, in_channel // 2, kernel_size=1, bias=False)
        self.soft = nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        """
        Args:
            x1 Tensor(B, C, N1): k, v
            x2 Tensor(B, C, N2): q

        Returns:
            Tensor(B, C, N2)
        """
        C = x1.shape[1]
        # xn = get_graph_feature(x1, self.k)  # [B, N1, k, C]
        # xn = xn.permute(0, 3, 2, 1)  # [B, C, k, N1]
        # xn = self.linear_xn(xn)
        # x1 = self.linear_x1(x1).unsqueeze(2)
        #
        # k = self.wk(xn - x1)  # [B, C, k, N1]
        # v = self.wv(x1).squeeze(2)
        # q = self.wq(x2)  # [B, C, N2]
        #
        # q = q.unsqueeze(1).permute(0, 1, 3, 2)  # [B, 1, N2, C]
        # k = k.permute(0, 2, 1, 3)  # [B, k, C, N1]
        # energy = torch.matmul(q, k)  # [B, k, N2, N1]
        # energy = energy.max(dim=1, keepdim=False)[0]
        # energy = self.soft(energy / math.sqrt(C))  # [B, N2, N1]
        # return torch.bmm(energy, v.permute(0, 2, 1)).permute(0, 2, 1)
        k = self.wk(x1)
        v = self.wv(x1)
        q = self.wq(x2)

        energy = torch.bmm(q.permute(0, 2, 1), k)
        energy = self.soft(energy / math.sqrt(C))
        return torch.bmm(energy, v.permute(0, 2, 1)).permute(0, 2, 1)


class MultiHeadRSAttention(nn.Module):
    def __init__(self, in_channel, cfg):
        super(MultiHeadRSAttention, self).__init__()
        self.n = cfg.n_head
        self.head = nn.ModuleList()
        for i in range(self.n):
            self.head.append(RSAttention(in_channel, cfg.att_k))
        self.conv = nn.Conv1d(in_channel // 2 * cfg.n_head, in_channel, kernel_size=1, bias=False)

    def forward(self, x1, x2):
        """
        Args:
            x1 Tensor(B, C, N1): k, v
            x2 Tensor(B, C, N2): q

        Returns:
            Tensor(B, C, N2)
        """
        for i in range(self.n):
            if i == 0:
                catAtt = self.head[0](x1, x2)
            else:
                tmp = self.head[i](x1, x2)
                catAtt = torch.cat((catAtt, tmp), dim=1)
        catAtt = self.conv(catAtt)
        return catAtt
