"""
This file is only used for testing parts of the code and will be removed when
the project is finished.
"""
import hydra
from omegaconf import OmegaConf

import lightning.pytorch as pl
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize
from models import get_model
from models.unet import UNetPolyp, UNetSD
from models.unet_lightning import UNetLightning
from models.losses import dice_loss_fun
from data import load_polypgen_centralized, load_polypgen_federated, load_config
from monai.losses import DiceLoss
from utils import extract_data_config
from collections import OrderedDict
import torchmetrics
import sys
import os
import time
import pandas as pd
import wandb
from skimage import io

import numpy as np
import cv2

from models.missformer import MISSFormer
from models.utnet import UTNet, ChannelAttention
from models.sdnet import SDNet
from models.sd_utnet import SD_UTNet
from optimizers.myadamw import MyAdamW
from torch.optim import AdamW


# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.image import show_cam_on_image

from matplotlib import pyplot as plt


# pl.seed_everything(0)

# cfg_name = "sdnet_exp5"
# config = load_config(cfg_name)
# data_config = extract_data_config(config)

# dataloaders, num_examples = load_polypgen_federated(None, data_config)
# print(num_examples)

# for x, y in dataloaders["out_test"]:
#     print(torch.min(x), torch.max(x))
#     break

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # net = get_model(config, use_lightning=False, local=False).to(device)
# # criterion = local_model.get_criterion(train=True)
# # optimizer = torch.optim.AdamW(local_model.parameters(), lr=0.001)

# # print(torch.cuda.memory_allocated())
# # print(torch.cuda.memory_reserved())

# # # # global_model = get_model(config, use_lightning=False, local=False)

# # print(sum(p.numel() for p in local_model.parameters()))
# # # # print(sum(p.numel() for p in global_model.parameters()))

# img_size = 128
# x = torch.ones(2, 3, img_size, img_size).to(device)
# y = (torch.rand((1, 1, 512, 512)) > 0.5).float().to(device)


# # net = ChannelAttention(in_width=32, in_height=32, depth=512).to(device)
# # net(x)
# net = SD_UTNet(width=img_size, height=img_size).to(device)
# net(x)
# print(sum(p.numel() for p in net.parameters()))


# for name, param in net.named_parameters():
#     print(name, param.shape)
# print(torch.min(x), torch.max(x))

# x2 = (torch.rand(1, 8, 512, 512).to("cuda") > 0.5).float()


# regress_loss = nn.L1Loss()
# reco_loss = nn.L1Loss()
# anam_loss = nn.L1Loss()
# dice_loss = DiceLoss()
# print(dice_loss_fun(torch.sigmoid(x), torch.sigmoid(x2)))
# regress_w = 1.0
# reco_w = 1.0
# aug_w = 1.0

# print(local_model.content_encoder.unet.encoder_block1[0].weight)
# # print(local_model.m_encoder.block1[0].weight)
# optimizer.zero_grad()

# start_time = time.time()

# a_out = local_model.content_encoder(x)
# seg_pred = local_model.content_segmentor(a_out)

# for name, param in local_model.named_parameters():
#     if "content" not in name:
#         param.requires_grad = False

# org_state_dict = OrderedDict()
# org_state_dict["m_encoder.block1.0.weight"] = local_model.state_dict()["m_encoder.block1.0.weight"].clone()
# org_state_dict["m_encoder.block1.0.bias"] = local_model.state_dict()["m_encoder.block1.0.bias"].clone()

# centers = [1]
# total_aug_loss = 0
# for aug_center_nr in centers:
#     state_dict = OrderedDict()
#     state_dict["m_encoder.block1.0.weight"] = torch.ones((16, 11, 3, 3)) * 100
#     state_dict["m_encoder.block1.0.bias"] = torch.ones((16)) * 100

#     local_model.load_state_dict(state_dict, strict=False)

#     aug_z_out, aug_mu_out, aug_logvar_out = local_model.m_encoder(a_out, x)
#     aug_reco = local_model.decoder(a_out, aug_z_out)
#     aug_reco = aug_reco.detach()
#     aug_a_out_tilde = local_model.content_encoder(aug_reco)

#     aug_loss = anam_loss(aug_a_out_tilde, a_out) * aug_w * (1/len(centers)) * 1000
#     print(aug_loss)
#     # aug_loss.backward(retain_graph=True)
#     total_aug_loss += aug_loss.item()

# for name, param in local_model.named_parameters():
#     if "content" not in name:
#         param.requires_grad = True


# local_model.load_state_dict(org_state_dict, strict=False)

# z_out, mu_out, logvar_out = local_model.m_encoder(a_out, x)
# reco = local_model.decoder(a_out, z_out)
# a_out_tilde = local_model.content_encoder(reco)
# _, mu_out_tilde, _ = local_model.m_encoder(a_out_tilde, reco)

# loss_dict = criterion.forward_aug(y, seg_pred, reco, z_out, mu_out_tilde, mu_out, logvar_out, aug_loss, x)
# loss = loss_dict['train_loss'] + aug_loss
# loss.backward()

# optimizer.step()

# end_time = time.time() - start_time

# print(local_model.content_encoder.unet.encoder_block1[0].weight)
# # print(local_model.m_encoder.block1[0].weight)

# print(f"execution time: {end_time}")

from client import FlowerClientGrad, get_client_fn

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def hydra_test(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_size = 256
    x = torch.ones(10, 3, img_size, img_size).to(device)
    # y = (torch.rand((2, 3, 512, 512)) > 0.5).float().to(device)

    net = UNetPolyp()

    print(sum(p.numel() for p in net.parameters()))

    net = UNetSD()

    print(sum(p.numel() for p in net.parameters()))




if __name__ == "__main__":
    hydra_test()