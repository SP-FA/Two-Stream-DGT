import torch
from torch import nn

from model.DGN.dgn import EdgeConv


class DynamicGraphRPN(nn.Module):
    def __init__(self, feature_channel, vote_channel, cfg):
        super(DynamicGraphRPN, self).__init__()
        self.vote_dgn = nn.Sequential(
            EdgeConv(feature_channel, vote_channel    , k=cfg.rpn_k),
            EdgeConv(vote_channel   , vote_channel    , k=cfg.rpn_k),
            EdgeConv(vote_channel   , vote_channel + 3, k=cfg.rpn_k),
        )
        self.cla_dgn = nn.Sequential(
            EdgeConv(feature_channel + 3 , feature_channel // 2, k=cfg.rpn_k),
            EdgeConv(feature_channel // 2, feature_channel // 2, k=cfg.rpn_k),
            EdgeConv(feature_channel // 2, 1, k=cfg.rpn_k),
        )
        self.sigmoid = nn.Sigmoid()
        self.box_dgn = nn.Sequential(
            EdgeConv(vote_channel + 3 + 1, vote_channel // 2, k=cfg.rpn_k),
            EdgeConv(vote_channel // 2, vote_channel // 4, k=cfg.rpn_k),
            EdgeConv(vote_channel // 4, 5, k=cfg.rpn_k),
        )

    def forward(self, xyz, feature):
        """
        Args:
            xyz Tensor(B, 3, N)
            feature Tensor(B, feature_channel, N)

        Returns:
            Tensor(B, 5, N): x, y, z, theta, conf
            Tensor(B, 1, N)
        """
        vote = self.vote_dgn(feature)
        voteOffset  = vote[:, :3, :]
        voteFeature = vote[:, 3:, :]  # [B, vote_channel, N]
        vote_xyz = xyz + voteOffset

        xyz_feature = torch.cat((xyz, feature), dim=1)
        cla = self.cla_dgn(xyz_feature)
        score = self.sigmoid(cla)  # [B, 1, N]

        voteFeature = torch.cat((voteFeature, score), dim=1)
        vote_xyz_feature = torch.cat((vote_xyz, voteFeature), dim=1)
        box = self.box_dgn(vote_xyz_feature)  # [B, 5, N]
        return box, cla
