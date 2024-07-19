import lightning.pytorch as pl
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

import torch
from models import get_model
from data import load_polypgen_centralized, load_polypgen_federated, load_config
import torchmetrics
import os
import wandb
import numpy as np
import cv2
from collections import defaultdict
from torchvision.io import read_image
from torchvision.transforms import Resize
from torchvision.utils import make_grid
from models.sdnet import SDNet, SDNetGlobal
from models.sdnet_lightning import SDNetLightning

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

CKPTS = [
            "sdnet_exp1",
            "sd_utnet_exp1",
            "federated_sdnet_exp1_best.pt",
            "federated_sd_utnet_exp2_best.pt",
        ]

class SegmentationTarget:
    def __init__(self, mask, category=1):
        self.category = category
        self.mask = mask
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()
        
    def __call__(self, model_output):
        return (model_output[0, self.category, :, : ] * self.mask).sum()

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataloaders, num_examples = load_polypgen_federated(4, data_config={
                                                    "batch_size": 4,
                                                    "target_size": 512,
                                                    "out_center": 6,
                                                    "splits": "per_patient",
                                                    "seed": 1,
                                                    "load_in_ram": False})
    print(num_examples)

    wandb.init(project=f"plot_images",
               config={'ckpts': CKPTS})
    
    n_rows = 3
    n_cols = 8

    for ckpt_name in CKPTS:
        if ckpt_name.startswith("federated"):
            config = load_config('_'.join(ckpt_name.split('_')[1:-1]))
            path = f"ckpts/{ckpt_name}"
            net = get_model(config, use_lightning=False, local=False)
            ckpt = torch.load(path)
            net.load_state_dict(ckpt)
        else:
            config = load_config(ckpt_name)
            path = os.path.join(f"ckpts/{ckpt_name}", sorted(os.listdir(f"ckpts/{ckpt_name}"))[0])
            net = get_model(config, local=False)
            ckpt = torch.load(path)
            net.load_state_dict(ckpt['state_dict'])

        imgs = torch.zeros((n_rows*n_cols, 3, 256, 256))

        net = net.to(device)
        gradcam_images(net, imgs, 0, dataloaders["val"], device, n_cols)
        mask_images(net, imgs, n_cols, dataloaders["val"], device, n_cols)

        imgs_grid = make_grid(imgs, nrow=n_cols)
        imgs_wandb = wandb.Image(imgs_grid)
        wandb.log({f"{ckpt_name.split('_best.pt')[0]}/gradcams": imgs_wandb})


def mask_images(net, imgs, start_idx, dataloader, device, n_cols=10):
    net.eval()
    resize_transform = Resize((256, 256)) # reduce dimensions to decrease wandb logged data
    n_images = 0

    with torch.no_grad():
        for i, sample_data in enumerate(dataloader):
            images, y_true, img_path, mask_path = sample_data
            img_path, mask_path = img_path[0], mask_path[0]
        
            if n_images >= n_cols:
                break

            images, y_true = images.to(device), y_true.to(device)
        
            out = net(images)
            out = out if type(out) is tuple else tuple([out])

            seg_pred = out[0][:, 1, :, :].detach().cpu().float() *255 # seg_pred has two channels
            seg_true = y_true[:, 0, :, :].detach().cpu().float() *255 # y_true has one channel

            seg_pred_rgb = cv2.cvtColor(seg_pred.squeeze().numpy(), cv2.COLOR_GRAY2RGB)
            seg_true_rgb = cv2.cvtColor(seg_true.squeeze().numpy(), cv2.COLOR_GRAY2RGB)

            imgs[start_idx+i] = resize_transform(torch.from_numpy(seg_pred_rgb).permute(2,0,1))
            imgs[start_idx+i+n_cols] = resize_transform(torch.from_numpy(seg_true_rgb).permute(2,0,1))

            n_images += 1

def gradcam_images(net, imgs, start_idx, dataloader, device, n_cols=10):
    net.train()
    org_resize_transform = Resize((512, 512))
    resize_transform = Resize((256, 256)) # reduce dimensions to decrease wandb logged data
    n_images = 0

    if isinstance(net, SDNetGlobal):
        target_layers = [net.encoder.unet.decoder_upconv4[-1], net.encoder.unet.decoder_block4[-1]]
    elif isinstance(net, SDNetLightning):
        if isinstance(net.model, SDNet):
            target_layers = [net.model.a_encoder.unet.decoder_upconv4[-1], net.model.a_encoder.unet.decoder_block4[-1]]
        else:
            target_layers = [net.model.a_encoder.utnet.up4.conv[-1]]
    elif isinstance(net, SDNet):
        target_layers = [net.a_encoder.unet.decoder_upconv4[-1], net.a_encoder.unet.decoder_block4[-1]]
    else:
        target_layers = [net.a_encoder.utnet.up4.conv[-1]]

    with GradCAM(model=net, target_layers=target_layers) as cam:
        for i, sample_data in enumerate(dataloader):
            images, y_true, img_path, mask_path = sample_data
            img_path, mask_path = img_path[0], mask_path[0]
        
            if n_images >= n_cols:
                break

            images, y_true = images.to(device), y_true.to(device)
            targets = [SegmentationTarget(y_true)]

            grayscale_cam = cam(input_tensor=images, targets=targets)[0, :]
            org_img = org_resize_transform(read_image(img_path))
            cam_image = show_cam_on_image(org_img.permute(1,2,0).numpy()/255, grayscale_cam, use_rgb=True)
            imgs[start_idx+i] = resize_transform(torch.from_numpy(cam_image).permute(2,0,1))

            n_images += 1


if __name__ == "__main__":
    main()