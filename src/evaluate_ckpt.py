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
            "federated_sd_utnet_no_p_factor_best.pt",
            "federated_sd_utnet_fed_exp1_best.pt",
            "federated_sd_utnet_fed_exp2_best.pt",
            "federated_sd_utnet_fed_baseline_best.pt"
        ]

model_names =   [
                    "sd_utnet_fed", "sd_utnet", "sdnet", "unet", "utnet"
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

    wandb.init(project=f"test_checkpoint",
               config={'ckpts': CKPTS, 'dataset': dataset})

    columns = ["ckpt_name", "avg_dsc_enhanced", "avg_dsc"]
    print(columns)
    table_val = wandb.Table(columns=columns)
    table_in = wandb.Table(columns=columns)
    table_out = wandb.Table(columns=columns)

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

        net = net.to(device)
        res_val = test(net, dataloaders['val'], device, dataset)
        table_val.add_data(ckpt_name, *res_val)
        res_in = test(net, dataloaders['in_test'], device, dataset)
        table_in.add_data(ckpt_name, *res_in)
        res_out = test(net, dataloaders['out_test'], device, dataset)
        table_out.add_data(ckpt_name, *res_out)

    wandb.log({f"Table validation": table_val})
    wandb.log({f"Table in_test": table_in})
    wandb.log({f"Table out_test": table_out})

def test(net, dataloader, device, dataset):
    net.eval()
    
    dice_scores = defaultdict(lambda: list())

    with torch.no_grad():
        for images, y_true, img_path, _ in dataloader:
            images, y_true = images.to(device), y_true.to(device)

            out = net(images)
            out = out if type(out) is tuple else tuple([out])

            # y_pred = out[0][:, 1, :, :].unsqueeze(dim=1)
            # y_pred_binary = (y_pred > 0.5).float()

            dsc = torchmetrics.functional.dice(out[0], y_true, zero_division=1, num_classes=2, ignore_index=0)
            dsc = dsc.detach().cpu().numpy()
            
            dice_scores["avg_dsc"].append(dsc)

            if y_true.shape[0] > 1:
                print("!!! Batch size should be 1 !!!")

            if torch.sum(y_true) > 0:
                dice_scores["avg_dsc_enhanced"].append(dsc)

    avg_dsc_enhanced = f"{np.mean(dice_scores['avg_dsc_enhanced'])}\n(std. {np.std(dice_scores['avg_dsc_enhanced'])})\n(ratio {len(dice_scores['avg_dsc_enhanced']) / len(dice_scores['avg_dsc'])})"
    avg_dsc = f"{np.mean(dice_scores['avg_dsc'])}\n(std. {np.std(dice_scores['avg_dsc'])})"

    return avg_dsc_enhanced, avg_dsc

if __name__ == "__main__":
    main()
