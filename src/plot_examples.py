import matplotlib.pyplot as plt
from data import load_results
import numpy as np
import torch
import os
import math
from skimage import io
from models import get_model


CKPTS = [
            "federated_sd_utnet_no_p_factor_exp2_best.pt",
            "federated_sd_utnet_no_p_factor_exp2_last.pt",
            "federated_sd_utnet_no_p_factor_exp2_best_small.pt",
            "federated_sd_utnet_p_factor_exp5_best.pt",
            "federated_sd_utnet_p_factor_exp5_last.pt",
            "federated_sd_utnet_p_factor_exp5_best_small.pt",
        ]

name = "50000215" #C5

img = io.imread(f"data/data_C5/images_C5/EndoCV2021_C5_{name}.jpg")
mask = io.imread(f"data/data_C5/masks_C5/EndoCV2021_C5_{name}_mask.jpg")

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


transform = A.Compose([
            A.Resize(512, 512),
            A.Normalize(),
            ToTensorV2()
        ])

resize = A.Compose([A.Resize(512, 512), ToTensorV2()])

fig, ax = plt.subplots(nrows=2, ncols=3, layout='constrained')

res = transform(image=img, mask=mask)
img_t = res['image'].unsqueeze(dim=0).to(device)
mask_t = res['mask']
net = get_model("sd_utnet", use_lightning=False, local=False).to(device)

for i, ckpt_path in enumerate(CKPTS):
    ckpt = torch.load("ckpts/" + ckpt_path)
    net.eval()
    net.load_state_dict(ckpt)

    out = net(img_t)
    ax[i//3][i%3].imshow(out[0][0, 1, :, :].detach().cpu().numpy(), cmap='gray')
    ax[i//3][i%3].set_title('_'.join(ckpt_path.split('_')[-3:]))

    if ckpt_path.endswith("best.pt"):
        plt.imsave(f"plots/{ckpt_path[:-2]}.png", out[0][0, 1, :, :].detach().cpu().numpy(), cmap='gray')

plt.figure()
plt.imshow(mask_t.cpu().numpy(), cmap='gray')
plt.imsave("plots/gt.png", mask_t.cpu().numpy(), cmap='gray')
plt.imsave("plots/input.png", resize(image=img)['image'].permute(1,2,0).numpy())

plt.show()