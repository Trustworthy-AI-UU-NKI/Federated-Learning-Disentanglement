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
            "federated_sd_utnet_lits_no_p_factor_best.pt",
            "federated_sd_utnet_lits_no_p_factor_best_small.pt",
            "federated_sd_utnet_lits_p_factor_best.pt",
            "federated_sd_utnet_lits_p_factor_best_small.pt",
            "federated_utnet_lits_no_p_factor_best.pt",
            "federated_utnet_lits_no_p_factor_best_small.pt",
            "federated_utnet_lits_p_factor_best.pt",
            "federated_utnet_lits_p_factor_best_small.pt"
        ]

model_names =   [
                    "sd_utnet", "sdnet", "utnet"
                ]


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_config = { "batch_size": 4,
                    "target_size": 512,
                    "out_center": 5,
                    "splits": "per_patient",
                    "seed": 1,
                    "load_in_ram": False,
                    "load_names": False,
                    "fold_nr": None}
    
    dataset = "lits"

    if dataset == "polypgen":
        in_channels = 3
        dataloaders, num_examples = load_polypgen_federated(None, data_config=data_config)
    else:
        in_channels = 1
        dataloaders, num_examples = load_lits_federated(None, data_config=data_config)

    print(num_examples)

    wandb.init(project=f"test_polyp_sizes",
               config={'ckpts': CKPTS, 'dataset': dataset})

    columns = ["ckpt_name", "small_dsc", "large_dsc", "avg_dsc_enhanced", "avg_dsc"]
    # columns = ["ckpt_name", "small_dsc", "med_dsc", "large_dsc", "avg_dsc_enhanced", "avg_dsc"]
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
            path = os.path.join(f"ckpts/{ckpt_name}", sorted(os.listdir(f"ckpts/{ckpt_name}"))[0])
            net = get_model(model_name, in_channels=in_channels, local=False)
            ckpt = torch.load(path)
            net.load_state_dict(ckpt['state_dict'])

        net = net.to(device)
        res_val, val_scores = test(net, dataloaders['val'], device, dataset)
        table_val.add_data(ckpt_name, *res_val)

        if dataset == "lits":
            res_in, in_scores = test_multi_dataloaders(net, [dataloaders['in_test'], dataloaders['out_test']], device, dataset)
            table_in.add_data(ckpt_name, *res_in)
            out_scores = []
        else:
            res_in, in_scores = test(net, dataloaders['in_test'], device, dataset)
            table_in.add_data(ckpt_name, *res_in)
            res_out, out_scores = test(net, dataloaders['out_test'], device, dataset)
            table_out.add_data(ckpt_name, *res_out)

        save_individual_scores(ckpt_name, {'in_scores': in_scores, 'out_scores': out_scores})


    wandb.log({f"Table validation": table_val})
    wandb.log({f"Table in_test": table_in})
    wandb.log({f"Table out_test": table_out})


def save_individual_scores(ckpt_name, ckpt_scores):
    exp_idx = ckpt_name.find('exp')
    if exp_idx >= 0:
        ckpt_save_name = ckpt_name.replace(ckpt_name[exp_idx:exp_idx+4], '')
    else:
        ckpt_save_name = ckpt_name
    for scores_set in ["in_scores", "out_scores"]:
        df_rows = []
        for score in ckpt_scores[scores_set]:
                df_rows.append([score[0], score[1], score[2], score[3]])
        df = pd.DataFrame(df_rows, columns=["img_path", "dsc", "small_polyp", "inv_area"])
        df.to_csv(os.path.join("fold_results", f"{ckpt_save_name}_{scores_set}.csv"))

def test(net, dataloader, device, dataset):
    net.eval()
    
    dice_scores = defaultdict(lambda: list())
    img_scores = []

    with torch.no_grad():
        for images, y_true, img_path, _ in dataloader:
            images, y_true = images.to(device), y_true.to(device)

            small_polyp, inv_area = check_small_polyp(img_path[0], True, dataset)
            out = net(images)
            out = out if type(out) is tuple else tuple([out])

            dsc = torchmetrics.functional.dice(out[0], y_true, zero_division=1, num_classes=2, ignore_index=0)
            dsc = dsc.detach().cpu().numpy()
            
            dice_scores["avg_dsc"].append(dsc)

            if inv_area > 0:
                dice_scores["avg_dsc_enhanced"].append(dsc)

                if small_polyp:
                    dice_scores["small_dsc"].append(dsc)
                else:
                    dice_scores["large_dsc"].append(dsc)

            img_scores.append([img_path[0], dsc, small_polyp, inv_area])

    small_dsc = f"{np.mean(dice_scores['small_dsc'])}\n(std. {np.std(dice_scores['small_dsc'])})\n(ratio {len(dice_scores['small_dsc']) / len(dice_scores['avg_dsc_enhanced'])})"
    # med_dsc = f"{np.mean(dice_scores['med_dsc'])}\n(std. {np.std(dice_scores['med_dsc'])})\n(ratio {len(dice_scores['med_dsc']) / len(dice_scores['avg_dsc_enhanced'])})"
    large_dsc = f"{np.mean(dice_scores['large_dsc'])}\n(std. {np.std(dice_scores['large_dsc'])})\n(ratio {len(dice_scores['large_dsc']) / len(dice_scores['avg_dsc_enhanced'])})"
    avg_dsc_enhanced = f"{np.mean(dice_scores['avg_dsc_enhanced'])}\n(std. {np.std(dice_scores['avg_dsc_enhanced'])})\n(ratio {len(dice_scores['avg_dsc_enhanced']) / len(dice_scores['avg_dsc'])})"
    avg_dsc = f"{np.mean(dice_scores['avg_dsc'])}\n(std. {np.std(dice_scores['avg_dsc'])})"

    return [small_dsc, large_dsc, avg_dsc_enhanced, avg_dsc], img_scores
    # return small_dsc, med_dsc, large_dsc, avg_dsc_enhanced, avg_dsc

def test_multi_dataloaders(net, dataloaders, device, dataset):
    net.eval()
    
    dice_scores = defaultdict(lambda: list())
    img_scores = []

    with torch.no_grad():
        for dataloader in dataloaders:
            for images, y_true, img_path, _ in dataloader:
                images, y_true = images.to(device), y_true.to(device)

                small_polyp, inv_area = check_small_polyp(img_path[0], True, dataset)
                out = net(images)
                out = out if type(out) is tuple else tuple([out])

                dsc = torchmetrics.functional.dice(out[0], y_true, zero_division=1, num_classes=2, ignore_index=0)
                dsc = dsc.detach().cpu().numpy()
                
                dice_scores["avg_dsc"].append(dsc)

                if inv_area > 0:
                    dice_scores["avg_dsc_enhanced"].append(dsc)

                    if small_polyp:
                        dice_scores["small_dsc"].append(dsc)
                    else:
                        dice_scores["large_dsc"].append(dsc)

                img_scores.append([img_path[0], dsc, small_polyp, inv_area])

    small_dsc = f"{np.mean(dice_scores['small_dsc'])}\n(std. {np.std(dice_scores['small_dsc'])})\n(ratio {len(dice_scores['small_dsc']) / len(dice_scores['avg_dsc_enhanced'])})"
    large_dsc = f"{np.mean(dice_scores['large_dsc'])}\n(std. {np.std(dice_scores['large_dsc'])})\n(ratio {len(dice_scores['large_dsc']) / len(dice_scores['avg_dsc_enhanced'])})"
    avg_dsc_enhanced = f"{np.mean(dice_scores['avg_dsc_enhanced'])}\n(std. {np.std(dice_scores['avg_dsc_enhanced'])})\n(ratio {len(dice_scores['avg_dsc_enhanced']) / len(dice_scores['avg_dsc'])})"
    avg_dsc = f"{np.mean(dice_scores['avg_dsc'])}\n(std. {np.std(dice_scores['avg_dsc'])})"

    return [small_dsc, large_dsc, avg_dsc_enhanced, avg_dsc], img_scores

if __name__ == "__main__":
    main()
