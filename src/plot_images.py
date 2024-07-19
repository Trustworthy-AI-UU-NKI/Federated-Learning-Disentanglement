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
from collections import defaultdict
from skimage import io
from torchvision.transforms import Resize
from torchvision.utils import make_grid
from models.sdnet import SDNet, SDNetGlobal
from models.sdnet_lightning import SDNetLightning
import cv2

CKPTS = [
            "federated_sd_utnet_no_p_factor_exp2_best.pt",
            "federated_sdnet_no_p_factor_exp1_best.pt",
        ]

model_names =   [
                    "sd_utnet", "sdnet", "unet", "utnet"
                ]


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_config = { "batch_size": 4,
                    "target_size": 512,
                    "out_center": 6,
                    "splits": "per_patient",
                    "seed": 1,
                    "load_in_ram": False,
                    "load_names": False,
                    "fold_nr": None}
    
    dataset = "polypgen"

    wandb.init(project=f"plot_images",
               config={'ckpts': CKPTS})
    
    if dataset == "polypgen":
        in_channels = 3
        dataloaders, num_examples = load_polypgen_federated(None, data_config=data_config)
    else:
        in_channels = 1
        # dataloaders, num_examples = load_lits_federated(N, data_config=data_config)

    

    print(num_examples)

    for ckpt_name in CKPTS:
        for name in model_names:
            if name in ckpt_name:
                model_name = name
                break

        print(ckpt_name, model_name)

        if ckpt_name.startswith("federated"):
            path = f"ckpts/{ckpt_name}"
            net = get_model(model_name, in_channels=in_channels, use_lightning=False, local=False)
            ckpt = torch.load(path)
            net.load_state_dict(ckpt)
        else:
            path = os.path.join(f"ckpts/{ckpt_name}", [name for name in os.listdir(f"ckpts/{ckpt_name}") if name.startswith('epoch')][0])
            net = get_model(model_name, in_channels=in_channels, local=False)
            ckpt = torch.load(path)
            net.load_state_dict(ckpt['state_dict'])

        print(ckpt_name, model_name)
        net = net.to(device)
        imgs_in = test_images(net, dataloaders["in_test"], device, max_images=4)
        imgs_out = (test_images(net, dataloaders["out_test"], device, max_images=1))
        imgs = torch.cat((imgs_in, imgs_out), dim=0)
        imgs_grid = make_grid(imgs, nrow=11, padding=5)
        imgs_wandb = wandb.Image(imgs_grid, mode="rgb")
        wandb.log({f"{ckpt_name.split('_best.pt')[0]}/anatomy_channels": imgs_wandb})

def test_images(net, dataloader, device, max_images=10):
    net.eval()
    resize_transform = Resize((128, 128)) # reduce dimensions to decrease wandb logged data
    n_images = 0
    imgs = torch.zeros((11*max_images, 3, 128, 128))
    # images_idx = [0, ]
    image_names = [
        "data/data_C4/images_C4/18_endocv2021_positive_1492.jpg",
        "data/data_C4/images_C4/11_endocv2021_positive_900.jpg",
        "data/data_C5/images_C5/EndoCV2021_C5_50000279.jpg",
        "data/data_C5/images_C5/EndoCV2021_C5_50000062.jpg",
        "data/data_C6/images_C6/EndoCV2021_C6_0100045.jpg"

    ]

    with torch.no_grad():
        for i, sample_data in enumerate(dataloader):
            images, y_true, img_path, mask_path = sample_data
            img_path, mask_path = img_path[0], mask_path[0]
     
            if n_images >= max_images:
                break

            if img_path in image_names:
                images, y_true = images.to(device), y_true.to(device)
                if isinstance(net, SDNetGlobal):
                    a_out = net.encoder(images)
                elif isinstance(net, SDNetLightning):
                    a_out = net.model.a_encoder(images)
                else:
                    a_out = net.a_encoder(images)

                seg_pred = net(images)[0]

                dice_score = torchmetrics.functional.dice(seg_pred, y_true, zero_division=1, num_classes=2, ignore_index=0)

                row_idx = image_names.index(img_path) if max_images > 1 else 0

                print(f"{row_idx}) {img_path}: {dice_score}")

                imgs[row_idx*11+0] = resize_transform((torch.from_numpy(io.imread(img_path)).permute(2, 0, 1)/255).unsqueeze(dim=0)).squeeze(dim=0)
                imgs[row_idx*11+1] = torch.repeat_interleave(resize_transform(y_true.cpu()).squeeze(dim=0), 3, dim=0)
                imgs[row_idx*11+2] = torch.repeat_interleave(resize_transform(seg_pred[:, 1, :, :].cpu().unsqueeze(dim=0)).squeeze(dim=0), 3, dim=0)

                for idx in range(a_out.shape[1]):
                    img = a_out[0][idx][:][:]
                    img = img.cpu().unsqueeze(dim=0).unsqueeze(dim=0)
                    img_resized = (torch.repeat_interleave(resize_transform(img).squeeze(dim=0), 3, dim=0)+1)/2
                    imgs[row_idx*11+3+idx] = img_resized
                
                n_images += 1

    return imgs


if __name__ == "__main__":
    main()