import torch
from models import get_model
from data import load_polypgen_centralized, load_polypgen_federated, load_config, get_polyp_size, check_small_polyp
import torchmetrics
import os
import wandb
import numpy as np
from collections import defaultdict
from skimage import io
import albumentations as A
import pandas as pd

from scipy.stats import t

CKPTS = [
            "federated_sd_utnet_no_p_factor",
            "federated_sd_utnet_p_factor",
            "federated_utnet_no_p_factor",
            "federated_utnet_p_factor"
        ]

model_names =   [
                    "sd_utnet", "sdnet", "utnet"
                ]


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    wandb.init(project=f"test_folds",
               config={'ckpts': CKPTS})
    
    metric_names = ["dice_s", "dice_t", "dice_l", "dice_n", "dice"]

    columns = ["ckpt_name"] + metric_names
    print(columns)
    table_val = wandb.Table(columns=columns)
    table_in = wandb.Table(columns=columns)
    table_out = wandb.Table(columns=columns)

    dataset = "polypgen"

    dict_keys = ["in_test_" + metric for metric in metric_names]
    dict_keys.extend(["out_test_" + metric for metric in metric_names])
    df_dict = {dict_key: dict() for dict_key in dict_keys}

    best_score_dict = {dict_key: 0 for dict_key in dict_keys}
    best_name_dict = {dict_key: None for dict_key in dict_keys}
    ckpt_type_names = []

    for ckpt_name in CKPTS:
        for name in model_names:
            if name in ckpt_name:
                model_name = name
                break

        net = get_model(model_name, in_channels=3, use_lightning=False, local=False)
        print(ckpt_name, model_name)

        best_fold_dict = {fold_i: {} for fold_i in range(1,6)}
        small_fold_dict = {fold_i: {} for fold_i in range(1,6)}

        for fold_i in range(1, 6):
            ckpt_save_name = f"{ckpt_name}_fold{fold_i}"
            best_ckpt = torch.load(f"ckpts/folds/{ckpt_save_name}_best.pt")
            small_ckpt = torch.load(f"ckpts/folds/{ckpt_save_name}_best_small.pt")

            dataloaders, num_examples = load_polypgen_federated(None, data_config={
                                                "batch_size": 4,
                                                "target_size": 512,
                                                "out_center": 6,
                                                "splits": "per_fold",
                                                "seed": 1,
                                                "load_in_ram": False,
                                                "load_names": False,
                                                "fold_nr": fold_i})
            
            for ckpt, dict_ckpt in [(best_ckpt, best_fold_dict), (small_ckpt, small_fold_dict)]:
                net.load_state_dict(ckpt)
                net = net.to(device)

                dict_ckpt[fold_i]['val'], dict_ckpt[fold_i]['val_scores'], dict_ckpt[fold_i]['val_ratio'] = test(net, dataloaders['val'], device, dataset)
                dict_ckpt[fold_i]['in'], dict_ckpt[fold_i]['in_scores'], dict_ckpt[fold_i]['in_ratio'] = test(net, dataloaders['in_test'], device, dataset)
                dict_ckpt[fold_i]['out'], dict_ckpt[fold_i]['out_scores'], dict_ckpt[fold_i]['out_ratio'] = test(net, dataloaders['out_test'], device, dataset)
        
        ckpt_i = 0
        # Average over all folds
        for ckpt_type, dict_ckpt in [("best", best_fold_dict), ("best_small", small_fold_dict)]:
            ckpt_type_name = ckpt_name + "_" + ckpt_type
            ckpt_type_names.append(ckpt_type_name)
            n_col = len(metric_names)
            val_total, in_total, out_total = np.zeros((5, n_col)), np.zeros((5, n_col)), np.zeros((5, n_col))

            for i, fold_i in enumerate(range(1, 6)):
                val_total[i] = dict_ckpt[fold_i]['val']
                in_total[i] = dict_ckpt[fold_i]['in']
                out_total[i] = dict_ckpt[fold_i]['out']

            for i, metric in enumerate(metric_names):
                in_m = "in_test_" + metric
                out_m = "out_test_" + metric
                df_dict[in_m][ckpt_type_name] = np.array([dict_ckpt[fold_i]['in'][i] for fold_i in range(1, 6)])
                df_dict[out_m][ckpt_type_name] = np.array([dict_ckpt[fold_i]['out'][i] for fold_i in range(1, 6)])

            res_val = (np.mean(val_total, axis=0)).tolist()
            res_val = [f"{res_val[i]}, std. {np.std(val_total[:, i])}" for i in range(len(res_val))]

            res_in = (np.mean(in_total, axis=0)).tolist()
            res_in_str = [f"{res_in[i]}, std. {np.std(in_total[:, i])}" for i in range(len(res_in))]

            res_out = (np.mean(out_total, axis=0)).tolist()
            res_out_str = [f"{res_out[i]}, std. {np.std(out_total[:, i])}" for i in range(len(res_out))]

            table_val.add_data(ckpt_type_name, *res_val)
            table_in.add_data(ckpt_type_name, *res_in_str)
            table_out.add_data(ckpt_type_name, *res_out_str)

            for i, metric in enumerate(metric_names):
                in_test_m = "in_test_" + metric
                out_test_m = "out_test_" + metric

                if res_in[i] > best_score_dict[in_test_m]:
                    best_score_dict[in_test_m] = res_in[i]
                    best_name_dict[in_test_m] = ckpt_type_name

                if res_out[i] > best_score_dict[out_test_m]:
                    best_score_dict[out_test_m] = res_out[i]
                    best_name_dict[out_test_m] =  ckpt_type_name

            ckpt_i += 1

            # Save individual scores to csv
            save_individual_scores(ckpt_type_name, dict_ckpt)

    wandb.log({f"Table validation": table_val})
    wandb.log({f"Table in_test": table_in})
    wandb.log({f"Table out_test": table_out})

    table_ratios = wandb.Table(columns=['fold', 'val', 'in-test', 'out-test'])
    for fold_i in range(1, 6):
        table_ratios.add_data(fold_i, dict_ckpt[fold_i]['val_ratio'], dict_ckpt[fold_i]['in_ratio'], dict_ckpt[fold_i]['out_ratio'])

    wandb.log({f"Table ratios": table_ratios})

    print()

def save_individual_scores(ckpt_type_name, dict_ckpt):
    for scores_set in ["in_scores", "out_scores"]:
        df_rows = []
        for fold_i in range(1, 6):
            fold_scores = dict_ckpt[fold_i][scores_set]
            for score in fold_scores:
                df_rows.append([fold_i, score[0], score[1], score[2], score[3]])
        df = pd.DataFrame(df_rows, columns=["fold_nr", "img_path", "dsc", "small_polyp", "inv_area"])
        df.to_csv(os.path.join("fold_results", f"{ckpt_type_name}_{scores_set}.csv"))
    


def test(net, dataloader, device, dataset):
    net.eval()
    threshold = 150 if dataset == "polypgen" else 1000
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
            
            dice_scores["dice"].append(dsc)

            if inv_area > 0:
                dice_scores["dice_n"].append(dsc)

                if small_polyp:
                    dice_scores["dice_s"].append(dsc)
                    if inv_area >= threshold:
                        dice_scores["dice_t"].append(dsc)
                else:
                    dice_scores["dice_l"].append(dsc)

            img_scores.append([img_path[0], dsc, small_polyp, inv_area])

    dice_s = np.mean(dice_scores['dice_s'])
    dice_t = np.mean(dice_scores['dice_t'])
    dice_l = np.mean(dice_scores['dice_l'])
    dice_n = np.mean(dice_scores['dice_n'])
    dice = np.mean(dice_scores['dice'])

    arr = np.array([dice_s, dice_t, dice_l, dice_n, dice])
    ratios = f"dice_s: {len(dice_scores['dice_s']) / len(dice_scores['dice_n'])} | " + \
             f"dice_t: {len(dice_scores['dice_t']) / len(dice_scores['dice_s'])} | " + \
             f"dice_l: {len(dice_scores['dice_l']) / len(dice_scores['dice_n'])} | " + \
             f"dice_n: {len(dice_scores['dice_n']) / len(dice_scores['dice'])}"

    return arr, img_scores, ratios

if __name__ == "__main__":
    main()
