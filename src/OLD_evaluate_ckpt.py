"""
This file is only used for testing parts of the code and will be removed when
the project is finished.
"""

import torch
from models import get_model
from data import load_polypgen_centralized, load_polypgen_federated, load_config
import torchmetrics
import os
import wandb
import numpy as np
from collections import defaultdict
from skimage import io
import albumentations as A

CKPTS = [
            "sd_utnet_lits_exp2",
        ]


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataloaders, num_examples = load_polypgen_federated(None, data_config={
                                                    "batch_size": 4,
                                                    "target_size": 512,
                                                    "out_center": 6,
                                                    "splits": "per_patient",
                                                    "seed": 1,
                                                    "load_names": True,
                                                    "load_in_ram": False})
    print(num_examples)

    wandb.init(project=f"seg_images",
               config={'ckpts': CKPTS})

    columns = ["img_name"]
    columns.extend([f"dsc_{name.split('_best.pt')[0]}" for name in CKPTS])
    columns.append("image")
    print(columns)
    table_in = wandb.Table(columns=columns)
    table_out = wandb.Table(columns=columns)
    table_avg = wandb.Table(columns=["dist", "ckpt", "dsc", "dsc_enhanced"])

    results = {}
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

        net = net.to(device)
        res_in = test_images(net, dataloaders['in_test'], device)
        res_out = test_images(net, dataloaders['out_test'], device)
        results[ckpt_name] = {'in': res_in, 'out': res_out}

    for test_name, res, table in [("in", res_in, table_in), ("out", res_out, table_out)]:
    # for test_name, res, table in [("in", res_in, table_in)]:
        avg_dsc = np.zeros(len(CKPTS))
        avg_dsc_enhanced = np.zeros(len(CKPTS))
        true_positives = 0

        for key in res.keys():
            org_img = my_transform(io.imread(key))
            org_mask = io.imread(results[CKPTS[0]][test_name][key]['mask_path'])
            org_mask = (np.where(org_mask > 128, 1, 0)).astype(np.float64)

            masks = {name.split('_best.pt')[0]: {"mask_data": my_transform(results[name][test_name][key]['y_pred'], i=i),
                                                 "class_labels": {i*2: f"{name.split('_best.pt')[0]}_0",
                                                                  i*2+1: f"{name.split('_best.pt')[0]}_1"}}
                        for i, name in enumerate(CKPTS, start=1)}
            masks["ground_truth"] = {"mask_data": my_transform(org_mask), "class_labels": {0: "true_0", 1: "true_1"}}
            img = wandb.Image(org_img, masks=masks)
            dsc_list = np.array([results[ckpt_name][test_name][key]['dsc'] for ckpt_name in CKPTS]).flatten()

            
            avg_dsc += dsc_list
            if np.sum(org_mask) > 0:
                avg_dsc_enhanced += dsc_list
                true_positives += 1

        for i, ckpt in enumerate(CKPTS):
            table_avg.add_data(test_name, ckpt, avg_dsc[i]/len(res.keys()), avg_dsc_enhanced[i]/true_positives)


    wandb.log({"Table avg dice scores": table_avg})

def my_transform(img, i=None):
    resize_transform = A.Resize(256, 256) # reduce dimensions to decrease wandb logged data

    if i is not None:
        img = np.where(img >= 0.99, i*2+1, i*2).astype(np.float64)

    return resize_transform.apply(img)

def test_images(net, dataloader, device, max_images=150):
    net.eval()

    preds = {}
    n_images = 0 
    images_idx = [i for i in range(0, len(dataloader))]

    with torch.no_grad():
        for i, sample_data in enumerate(dataloader):
            images, y_true, img_path, mask_path = sample_data
            img_path, mask_path = img_path[0], mask_path[0]
     
            if n_images >= max_images:
                break

            if i in images_idx:
                images, y_true = images.to(device), y_true.to(device)
                out = net(images)
                out = out if type(out) is tuple else tuple([out])

                dsc = torchmetrics.functional.dice(out[0], y_true, zero_division=1, num_classes=2, ignore_index=0)
                y_pred = (out[0][:, 1, :, :] > 0.5).float()

                preds[img_path] = {'mask_path': mask_path, 'dsc': dsc.cpu().numpy(), 'y_pred': y_pred.squeeze().cpu().numpy()}
                n_images += 1
    
    return preds

def test(net, dataloader, device):
    net.eval()

    criterion = net.get_criterion(train=False)
    correct, total_samples, loss_total, dice_total = 0, 0, 0.0, 0.0
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 4:
                images, y_true, img_path, mask_path = batch
            else:
                images, y_true = batch

            images, y_true = images.to(device), y_true.to(device)

            out = net(images)
            out = out if type(out) is tuple else tuple([out])

            y_pred = out[0][:, 1, :, :].unsqueeze(dim=1)
            y_pred_binary = (y_pred > 0.5).float()

            losses_dict = criterion(y_true, *out)
            loss = losses_dict['dice_loss']
            loss_total += loss.item()
            dice_total += torchmetrics.functional.dice(out[0], y_true, zero_division=1, num_classes=2, ignore_index=0)
            correct += torch.sum(y_pred_binary == y_true)
            total_samples += y_true.shape[0]

    loss_avg = loss_total / len(dataloader)
    dice_avg = dice_total / len(dataloader)
    accuracy = correct / (total_samples * y_true.shape[2] * y_true.shape[3])

    return loss_avg, accuracy, 1-loss_avg, dice_avg

if __name__ == "__main__":
    main()
