import os
import pprint

import yaml
import argparse
from easydict import EasyDict
import torch
from pyquaternion import Quaternion
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from dataset_loader.kitti_loader import KITTI_Loader
from dataset_util.box_struct import Box
from dataset_util.kitti import KITTI_Util
from model.DGT import TwoStreamDynamicGraphTransformer
from metrics import Success, Precision
from metrics import estimateOverlap, estimateAccuracy


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def load_yaml(file_name):
    with open(file_name, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            config = yaml.load(f)
    return config

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default="./config/DGT_KITTI.yaml", help='the config_file')
    parser.add_argument('--test', action='store_true', default=False, help='test mode')
    parser.add_argument('--preloading', action='store_true', default=False, help='preload dataset into memory')
    parser.add_argument('--device', default="cuda:0")

    args = parser.parse_args()
    config = load_yaml(args.cfg)
    config.update(vars(args))
    return EasyDict(config)


def criterion(batch, output, device):
    """
    Args:
        batch: {
            "template": Tensor[B, D1, P1]
            "boxCloud": Tensor[B, D3, P1]
            "searchArea": Tensor[B, D1, P2]
            "claLabel": List[Box * B]
            "trueBox": List[B * Box]
        }
        output: {
            "predBox": Tensor[B, 4+1, num_proposal]
            "predCla": Tensor[B, N]
            "vote_xyz": Tensor[B, N, 3]
            "center_xyz": Tensor[B, num_proposal, 3]
            "finalBox": [B, 4]
        }

    Returns:
        {
        "loss_objective": float
        "loss_box": float
        "loss_cla": float
        "loss_vote": float
    }
    """
    predBox = output['predBox']  # [B, N, 5]
    predCla = output['predCla'].squeeze(1)  # [B, N]

    claLabel = batch['claLabel'].to(device)  # [B, N]
    trueBox = batch['trueBox'].to(device)  # [B, 4]
    # center_xyz = output["center_xyz"]  # B,num_proposal,3
    # vote_xyz = output["vote_xyz"]
    loss_cla = F.binary_cross_entropy_with_logits(predCla, claLabel)

    # loss_vote = F.smooth_l1_loss(vote_xyz, trueBox[:, None, :3].expand_as(vote_xyz), reduction='none')  # B,N,3
    # loss_vote = (loss_vote.mean(2) * claLabel).sum() / (claLabel.sum() + 1e-06)

    # dist = torch.sum((center_xyz - trueBox[:, None, :3]) ** 2, dim=-1)

    # dist = torch.sqrt(dist + 1e-6)  # B, K
    # objectness_label = torch.zeros_like(dist, dtype=torch.float)
    # objectness_label[dist < 0.3] = 1
    #  = predBox[:, :, 4]  # B, K
    # objectness_mask = torch.zeros_like(objectness_label, dtype=torch.float)
    # objectness_mask[dist < 0.3] = 1
    # objectness_mask[dist > 0.6] = 1
    # loss_objective = F.binary_cross_entropy_with_logits(objectness_score, objectness_label,
    #                                                     pos_weight=torch.tensor([2.0]).cuda())
    # loss_objective = torch.sum(loss_objective * objectness_mask) / (
    #         torch.sum(objectness_mask) + 1e-6)
    loss_box = F.smooth_l1_loss(predBox[:, :, :4],
                                trueBox[:, None, :4].expand_as(predBox[:, :, :4]))
    # loss_box = torch.sum(loss_box.mean(2) * objectness_label) / (objectness_label.sum() + 1e-6)

    # totalLoss = loss_objective * cfg.object_weight + \
    #             loss_box * cfg.box_weight + \
    #             loss_cla * cfg.seg_weight + \
    #             loss_vote * cfg.vote_weight
    return loss_box * cfg.box_weight + loss_cla * cfg.cla_weight


if __name__ == "__main__":
    cfg = parse_config()
    pprint.pprint(cfg)
    cfg.device = torch.device(cfg.device)

    if cfg.pretrain is None: model = TwoStreamDynamicGraphTransformer(cfg).to(cfg.device)
    else:                    model = torch.load(cfg.pretrain).to(cfg.device)

    if cfg.optimizer.lower() == "sgd": optim = SGD (model.parameters(), lr=cfg.lr, momentum=0.9)
    else:                              optim = Adam(model.parameters(), lr=cfg.lr, betas=(0.5, 0.999), eps=1e-06)  # weight_decay=cfg.wd

    if cfg.dataset.lower() == "kitti":
        trainData = KITTI_Util(cfg, cfg.train_split)
        validData = KITTI_Util(cfg, cfg.valid_split)
        train = KITTI_Loader(trainData, cfg)
        valid = KITTI_Loader(validData, cfg)

    trainLoader = DataLoader(train, batch_size=cfg.batch_size, num_workers=cfg.workers, pin_memory=True, shuffle=True)
    validLoader = DataLoader(valid, batch_size=1             , num_workers=cfg.workers, pin_memory=True)

    # train
    if cfg.test is False:
        bestAcc = -1e9
        for i_epoch in range(cfg.epoch):
            totalLoss = 0
            SuccessTrain = Success()
            PrecisionTrain = Precision()

            model.train()
            trainBar = tqdm(trainLoader)
            for batch in trainBar:
                res = model(batch)
                loss = criterion(batch, res, cfg.device)

                optim.zero_grad()
                loss.backward()
                optim.step()
                totalLoss += loss.detach().item()

                for i in range(len(batch['trueBox'])):
                    tb = batch['trueBox'][i].cpu().detach().numpy()
                    fb = res['finalBox'][i].cpu().detach().numpy()
                    trueBox = Box((tb[0], tb[1], tb[2]), (tb[3], tb[4], tb[5]), tb[6], Quaternion(axis=[0, 0, 1], degrees=tb[6]))
                    predBox = Box((fb[0], fb[1], fb[2]), (tb[3], tb[4], tb[5]), fb[3], Quaternion(axis=[0, 0, 1], degrees=fb[3]))

                    overlap = estimateOverlap(trueBox, predBox)
                    accuracy = estimateAccuracy(trueBox, predBox)

                    SuccessTrain.add_overlap(overlap)
                    PrecisionTrain.add_accuracy(accuracy)

                trainBar.set_description(
                    f"train {i_epoch}/{cfg.epoch}: [loss: {totalLoss / len(trainLoader):.3f}] "
                    f"[S/P: {SuccessTrain.average:.1f}/{PrecisionTrain.average:.1f}]"
                )

            SuccessTrain.reset()
            PrecisionTrain.reset()

            if cfg.save_last:
                torch.save(model, os.path.join(cfg.checkpoint, f"last-model-{i_epoch}.pt"))

            if i_epoch % cfg.check_val_every_n_epoch != 0:
                continue

            # valid
            SuccessValid = Success()
            PrecisionValid = Precision()

            model.eval()
            validLoss = 0
            validBar = tqdm(validLoader)
            for batch in validBar:
                with torch.no_grad():
                    res = model(batch)
                    loss = criterion(batch, res, cfg.device)
                    validLoss += loss.detach().item()

                for i in range(len(batch['trueBox'])):
                    tb = batch['trueBox'][i].cpu().detach().numpy()
                    fb = res['finalBox'][i].cpu().detach().numpy()
                    trueBox = Box((tb[0], tb[1], tb[2]), (tb[3], tb[4], tb[5]), tb[6], Quaternion(axis=[0, 0, 1], degrees=tb[6]))
                    predBox = Box((fb[0], fb[1], fb[2]), (tb[3], tb[4], tb[5]), fb[3], Quaternion(axis=[0, 0, 1], degrees=fb[3]))

                    overlap = estimateOverlap(trueBox, predBox)
                    accuracy = estimateAccuracy(trueBox, predBox)

                    SuccessValid.add_overlap(overlap)
                    PrecisionValid.add_accuracy(accuracy)

                validBar.set_description(
                    f"valid {i_epoch}/{cfg.epoch}: [loss: {validLoss / len(validLoader):.3f}] "
                    f"[S/P: {SuccessValid.average:.1f}/{PrecisionValid.average:.1f}]"
                )

            if bestAcc < PrecisionValid.average:
                bestAcc = PrecisionValid.average
                torch.save(model, os.path.join(cfg.checkpoint, f"best_model-{i_epoch}-{SuccessValid.average}-{bestAcc}.pt"))

            SuccessValid.reset()
            PrecisionValid.reset()
        else:
            ...
