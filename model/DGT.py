import torch
import torch.nn as nn
import torch.nn.functional as F

from model.AddNorm import AddNormLayer
from model.DGN.dgn import DGN, EdgeConv
from model.DGRPN import DynamicGraphRPN
from model.RSAttention import MultiHeadRSAttention


class DecoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel, cfg, isLast=False):
        super(DecoderBlock, self).__init__()
        self.isLast = isLast

        self.att = MultiHeadRSAttention(in_channel, cfg)
        self.addnorm = AddNormLayer(in_channel, cfg)
        self.dgn1 = DGN(in_channel, in_channel, k=cfg.dgn_k)

        self.appearCrossAtt = MultiHeadRSAttention(in_channel, cfg)
        self.motionCrossAtt = MultiHeadRSAttention(in_channel, cfg)

        box_dim = 64
        self.box = nn.Linear(7, box_dim, bias=False)

        self.dgn21 = DGN(in_channel                           , out_channel                           , k=cfg.dgn_k)
        self.dgn22 = DGN(in_channel + box_dim if isLast else 0, out_channel + box_dim if isLast else 0, k=cfg.dgn_k)

    def self_attention_block(self, search, template):
        # Multi-head Attention
        searchAtt   = self.att(search  , search  )
        templateAtt = self.att(template, template)

        return (self.addnorm(search  , searchAtt  ),
                self.addnorm(template, templateAtt))

    def dgn_block(self, dgn1, dgn2, search, template):
        searchFeat   = dgn1(search)
        templateFeat = dgn2(template)

        return (self.addnorm(search  , searchFeat  ),
                self.addnorm(template, templateFeat))

    def cross_attention_block(self, search, template):
        # Cross Multi-head Attention
        searchAtt   = self.appearCrossAtt(template, search  )  # (k, v), q
        templateAtt = self.motionCrossAtt(search  , template)

        return (self.addnorm(search  , searchAtt  ),
                self.addnorm(template, templateAtt))

    def forward(self, search, template, bbox):
        """
        Args:
            search Tensor(B, D2, P1)
            template Tensor(B, D2, P2)
            bbox Tensor(B, 7)

        Returns:
            Tensor(B, D2, P1): search feature
            Tensor(B, D2, P2): template feature
        """
        search, template = self.self_attention_block(search, template)
        search, template = self.dgn_block(self.dgn1, self.dgn1, search, template)
        search, template = self.cross_attention_block(search, template)

        # bbox concat with template
        if self.isLast:
            N = template.shape[-1]
            bbox = self.box(bbox).unsqueeze(dim=-1).expand(-1, -1, N)  # [B, box_dim, 1]
            template = torch.cat((template, bbox), dim=1)  # [B, D2 * n_head + box_dim, P2)

        search, template = self.dgn_block(self.dgn21, self.dgn22, search, template)
        return search, template


class MotionDetectModule(nn.Module):
    def __init__(self, in_channel, out_channel=1024, k=5):
        super(MotionDetectModule, self).__init__()
        self.edgeConv1 = EdgeConv(in_channel, 128, k)
        self.edgeConv2 = EdgeConv(128, 256, k)
        self.edgeConv3 = EdgeConv(256, 256, k)

        self.conv = nn.Conv1d(256, out_channel // 2, kernel_size=1)
        self.relu = nn.LeakyReLU(negative_slope=0.2)
        self.norm = nn.BatchNorm1d(out_channel // 2)

    def forward(self, x):
        """
        Args:
            x Tensor(B, in_channel, N)

        Returns:
            Tensor(B, out_channel, 1)
        """
        B = x.shape[0]
        x = self.edgeConv1(x)
        x = self.edgeConv2(x)
        x = self.edgeConv3(x)

        x = self.conv(x)
        x = self.relu(x)
        x = self.norm(x)

        motionFeat1 = F.adaptive_max_pool1d(x, 1).view(B, -1, 1)
        motionFeat2 = F.adaptive_avg_pool1d(x, 1).view(B, -1, 1)  # [B, out_channel // 2, 1]
        return torch.cat((motionFeat1, motionFeat2), dim=1)  # [B, out_channel, 1]


class TwoStreamDynamicGraphTransformer(nn.Module):
    def __init__(self, cfg):
        super(TwoStreamDynamicGraphTransformer, self).__init__()
        self.device = cfg.device

        if cfg.dataset.lower() == "kitti":
            dim = 3

        self.dgn = DGN(dim, out_channel=128, k=cfg.dgn_k)
        self.decoder = DecoderBlock(128, 128, cfg, isLast=True)
        self.motion = MotionDetectModule(128 + 64, 256, k=cfg.motion_k)
        self.rpn = DynamicGraphRPN(128 * 2 + 64 * 2, 256, cfg)

    def forward(self, data):
        search = data["search"].to(self.device)
        template = data["template"].to(self.device)
        bbox = data["trueBox"].to(self.device)  # [B, 7]
        xyz = search[:, :3, :]

        searchFeat   = self.dgn(search)  # [B, D2, P1]
        templateFeat = self.dgn(template)  # [B, D2, P2]

        appear, motion = self.decoder(searchFeat, templateFeat, bbox)  # [B, dim, P1], [B, dim + box_dim, P2]
        motionFeat = self.motion(motion)
        N = appear.shape[-1]
        jointFeat = torch.cat((appear, motionFeat.expand(-1, -1, N)), dim=1)  # [B, dim + motion_dim, P2]
        box, cla = self.rpn(xyz, jointFeat)

        finalBoxID = torch.argmax(box[:, -1, :], dim=1).view(-1, 1, 1)
        finalBox = torch.gather(box, 1, finalBoxID.expand(-1, -1, box.shape[-1]))
        finalBox = finalBox.squeeze(1)
        return {
            "predBox": box,
            "predCla": cla,
            "finalBox": finalBox,
        }
