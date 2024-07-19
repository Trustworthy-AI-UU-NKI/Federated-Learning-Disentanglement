"""
This file is only used for testing parts of the code and will be removed when
the project is finished.
"""

import torch
from models import get_model
from data import load_polypgen_centralized, load_polypgen_federated, load_config, get_polyp_size, check_small_polyp
from data_lits import load_lits_federated
import torchmetrics
import os
import wandb
import numpy as np
from collections import defaultdict
from skimage import io
import albumentations as A
import pandas as pd

CKPTS = [
            "federated_utnet_no_p_factor_exp2_best.pt",
            "federated_sd_utnet_no_p_factor_exp2_best.pt",
            "federated_unet_no_p_factor_exp1_best.pt",
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

    if dataset == "polypgen":
        in_channels = 3
        dataloaders, num_examples = load_polypgen_federated(None, data_config=data_config)
    else:
        in_channels = 1
        dataloaders, num_examples = load_lits_federated(None, data_config=data_config)

    print(num_examples)

    wandb.init(project=f"test_centers",
               config={'ckpts': CKPTS, 'dataset': dataset})

    columns = ["center", "avg_dsc_enhanced", "avg_dsc", "ratio"]
    print(columns)
    tables = dict()

    for ckpt_name in CKPTS:
        for name in model_names:
            if name in ckpt_name:
                model_name = name
                break

        tables[ckpt_name] = wandb.Table(columns=columns)

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

        net = net.to(device)
        centers_scores = test(net, dataloaders['in_test'], device)

        for center_nr, scores in centers_scores.items():
            avg_dsc_enhanced = f"{np.mean(scores['avg_dsc_enhanced'])} (std. {np.std(scores['avg_dsc_enhanced'])})"
            avg_dsc = f"{np.mean(scores['avg_dsc'])} (std. {np.std(scores['avg_dsc'])})"
            ratio = len(scores['avg_dsc']) / len(dataloaders['in_test'])

            tables[ckpt_name].add_data(center_nr, avg_dsc_enhanced, avg_dsc, ratio)

        wandb.log({f"{ckpt_name}": tables[ckpt_name]})


def test(net, dataloader, device):
    net.eval()
    
    dice_scores_centers = defaultdict(lambda: {'avg_dsc': list(), 'avg_dsc_enhanced': list()})
   
    with torch.no_grad():
        for images, y_true, img_path, _ in dataloader:
            images, y_true = images.to(device), y_true.to(device)

            center_nr = int(img_path[0].split('/')[1][6])

            out = net(images)
            out = out if type(out) is tuple else tuple([out])

            dsc = torchmetrics.functional.dice(out[0], y_true, zero_division=1, num_classes=2, ignore_index=0)
            dsc = dsc.detach().cpu().numpy()
            
            dice_scores_centers[center_nr]["avg_dsc"].append(dsc)

            if y_true.shape[0] > 1:
                print("!!! Batch size should be 1 !!!")

            if torch.sum(y_true) > 0:
                dice_scores_centers[center_nr]["avg_dsc_enhanced"].append(dsc)

    return dice_scores_centers

if __name__ == "__main__":
    main()
