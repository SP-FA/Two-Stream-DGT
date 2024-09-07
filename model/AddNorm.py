from torch import nn


class AddNormLayer(nn.Module):
    def __init__(self, dim, cfg):
        super(AddNormLayer, self).__init__()
        self.dropout = nn.Dropout(cfg.addnorm_drop).to(cfg.device)
        self.norm = nn.InstanceNorm1d(dim).to(cfg.device)

    def forward(self, src, feat):
        """
        Args:
            src Tensor(B, C, N): 过 attention 层之前的 data
            feat Tensor(B, C, N): 过 attention 层之后的 data

        Returns:
            Tensor(B, C, N): 计算结果
        """
        feat = self.dropout(feat)
        return self.norm(src + feat)
