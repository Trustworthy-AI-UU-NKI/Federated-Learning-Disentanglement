import torch
import torch.nn as nn
from tqdm import tqdm
from data import save_model, check_small_polyp, POLYP_CENTERS
import matplotlib.pyplot as plt
from collections import defaultdict
import torchmetrics
import wandb
import numpy as np
import pandas as pd

def calculate_global_epoch(epoch, aug_epochs, train_rounds, epochs, completed_rounds):
    if aug_epochs is None:
        return completed_rounds*epochs + epoch
    else:
        completed_train_rounds = min(completed_rounds, train_rounds)
        aug_rounds = max(0, completed_rounds - completed_train_rounds)
        return completed_train_rounds*epochs + aug_rounds*aug_epochs + epoch

def train(net, optimizer, dataloader, device, center_nr, logger, **cfg):
    net.train()

    criterion = net.get_criterion(train=True)
    loss_per_epoch = torch.zeros(cfg['epochs'])
    p_factors = []
    for epoch in range(cfg['epochs']):
        epoch_losses_dict = defaultdict(lambda: 0)

        for batch in dataloader:
            if len(batch) == 4:
                images, y_true, img_path, mask_path = batch
            else:
                images, y_true = batch
                img_path, mask_path = None, None

            images, y_true = images.to(device), y_true.to(device)
            optimizer.zero_grad()

            out = net(images)
            out = out if type(out) is tuple else tuple([out])
            losses_dict = criterion(y_true, *out)
            loss = losses_dict['train_loss']
            loss.backward()

            p_factor = calculate_p_factor(img_path, y_true.cpu().numpy(), cfg['my_details'], cfg['dataset']) if cfg['p_factor_enabled'] else None
            if p_factor is not None and p_factor > 1:
                p_factors.append(p_factor)
                optimizer.step(p_factor=p_factor)
            else:
                optimizer.step()

            loss_per_epoch[epoch] += loss.item()
            epoch_losses_dict['dice_loss'] += losses_dict['dice_loss'].item()


        loss_per_epoch[epoch] /= len(dataloader)
        epoch_losses_dict['dice_loss'] /= len(dataloader)

        logger.log_metrics({f"{center_nr}/train_loss": loss_per_epoch[epoch],
                            f"{center_nr}/dice_loss": epoch_losses_dict["dice_loss"],
                            f"{center_nr}/epoch": calculate_global_epoch(epoch, cfg['aug_epochs'], cfg['train_rounds'], cfg['epochs'], cfg['server_round']-1)})
    
    if cfg['p_factor_enabled']:
        print(f"[client-{center_nr}] batch p_factor: {np.mean(p_factors)}, len p_factors {len(p_factors)} (out of {len(dataloader)*cfg['epochs']} iters)")

    return loss_per_epoch

def calculate_p_factor(img_path, y_true, my_details, dataset):
    total_area = (y_true.shape[2] * y_true.shape[3])
    small_factors = []

    for i, path in enumerate(img_path):
        small_polyp, inv_area = check_small_polyp(path, my_details, dataset)

        if small_polyp:
            inv_area = total_area / np.sum(y_true[i])
            log_base = 100 if dataset == "polypgen" else 1000
            small_factor = np.tanh(np.emath.logn(log_base, inv_area)**2)
            small_factors.append(small_factor)

    if len(small_factors) == 0:
        return None
    else:
        p_factor = sum(small_factors) * (2 / y_true.shape[0])
        
        return 1 + p_factor

def train_aug(net, optimizer, dataloader, device, save_path, center_nr, logger, **cfg):
    net.train()

    criterion = net.get_criterion(train=True)
    loss_per_epoch = torch.zeros(cfg['aug_epochs'])

    aug_centers = [1,2,3,4,5]
    aug_centers.remove(center_nr)
    aug_center_idx = ((center_nr-1) % len(aug_centers) + (cfg['server_round']-1)) % len(aug_centers)
    aug_center_nr = aug_centers[aug_center_idx]
    print(f"[client-{center_nr}] Using aug_center_nr {aug_center_nr}")


    for epoch in range(cfg['aug_epochs']):
        epoch_losses_dict = defaultdict(lambda: 0)

        for batch in dataloader:
            if len(batch) == 4:
                images, y_true, img_path, mask_path = batch
            else:
                images, y_true = batch
                img_path, mask_path = None, None

            images, y_true = images.to(device), y_true.to(device)
            optimizer.zero_grad()

            out = net.forward_aug(images, center_nr, cfg['server_round'], [aug_center_nr], save_path)
            out = out if type(out) is tuple else tuple([out])
            losses_dict = criterion.forward_aug(y_true, *out)
            loss = losses_dict['train_loss']
            loss.backward()
            optimizer.step()

            loss_per_epoch[epoch] += loss.item()
            for key in losses_dict.keys():
                if isinstance(losses_dict[key], float):
                    epoch_losses_dict[key] += losses_dict[key]
                else:
                    epoch_losses_dict[key] += losses_dict[key].item()

        loss_per_epoch[epoch] /= len(dataloader)
        for key in epoch_losses_dict.keys():
            epoch_losses_dict[key] /= len(dataloader)
        
        logger.log_metrics({f"{center_nr}/train_loss": loss_per_epoch[epoch],
                            f"{center_nr}/dice_loss": epoch_losses_dict["dice_loss"],
                            f"{center_nr}/total_aug_loss": epoch_losses_dict["aug_loss"],
                            f"{center_nr}/epoch": calculate_global_epoch(epoch, cfg['aug_epochs'], cfg['train_rounds'], cfg['epochs'], cfg['server_round']-1)})
        
    return loss_per_epoch

def test(net, dataloader, device, center_nr, logger, **cfg):
    net.eval()

    criterion = net.get_criterion(train=False)
    correct, total_samples, loss_total, dice_total, dice_small, small_samples= 0, 0, 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 4:
                images, y_true, img_path, mask_path = batch
            else:
                images, y_true = batch
                img_path, mask_path = None, None

            images, y_true = images.to(device), y_true.to(device)

            out = net(images)
            out = out if type(out) is tuple else tuple([out])

            y_pred = out[0][:, 1, :, :].unsqueeze(dim=1)
            y_pred_binary = (y_pred > 0.5).float()

            losses_dict = criterion(y_true, *out)
            loss = losses_dict['dice_loss']
            loss_total += loss.item()
            dice_score = torchmetrics.functional.dice(out[0], y_true, zero_division=1, num_classes=2, ignore_index=0)
            dice_total += dice_score
            correct += torch.sum(y_pred_binary == y_true)
            total_samples += y_true.shape[0]

            if img_path is not None:
                if len(img_path) > 1:
                    print("!!! Validation batch_size should be 1 !!!")
                if check_small_polyp(img_path[0], cfg['my_details'], cfg['dataset'])[0]:
                    dice_small += dice_score
                    small_samples += 1

    loss_avg = loss_total / len(dataloader)
    dice_avg = dice_total / len(dataloader)
    dice_small_avg = (dice_small / small_samples) if small_samples > 0 else 0.0
    accuracy = correct / (total_samples * y_true.shape[2] * y_true.shape[3])

    logger.log_metrics({f"{center_nr}/val_loss": loss_avg,
                f"{center_nr}/val_dice": dice_avg,
                f"{center_nr}/val_dice_small": dice_small_avg,
                f"{center_nr}/val_accuracy": accuracy,
                f"{center_nr}/epoch": max(0, calculate_global_epoch(0, cfg['aug_epochs'], cfg['train_rounds'], cfg['epochs'], cfg['server_round'])-1)})

    return loss_avg, accuracy, dice_avg, dice_small_avg

def train_centralized(net, dataloaders, device, epochs, lr, cfg_name, eval_per_epochs=1):
    criterion = net.get_criterion(train=True).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    loss_per_epoch = torch.zeros(epochs)
    eval_results = []
    best_score = 0.0
    best_epoch = 0

    for epoch in tqdm(range(epochs), desc="Epochs"):
        net.train()
        epoch_losses_dict = defaultdict(lambda: 0)

        for images, y_true in dataloaders["train"]:
            images, y_true = images.to(device), y_true.to(device)
            optimizer.zero_grad()

            out = net(images)
            losses_dict = criterion(y_true, *out)
            loss = losses_dict['train_loss']
            loss.backward()
            optimizer.step()

            loss_per_epoch[epoch] += loss.item()
            for key in losses_dict.keys():
                epoch_losses_dict[key] += losses_dict[key].item()

        loss_per_epoch[epoch] /= len(dataloaders["train"])
        for key in epoch_losses_dict.keys():
            epoch_losses_dict[key] /= len(dataloaders['train'])
        

        print(f"[Epoch {epoch}] average loss, dice loss: {loss_per_epoch[epoch]}, {epoch_losses_dict['dice_loss']}")

        wandb.log(epoch_losses_dict, commit=False)

        if (epoch+1) % eval_per_epochs == 0:
            eval_results.append(test(net, dataloaders["val"], device))

            if eval_results[-1][-1] > best_score:
                best_score = eval_results[-1][-1]
                best_epoch = epoch
                save_model(net, cfg_name)
                print(f"[Epoch {epoch}] [CKPT SAVED] eval results: {eval_results[-1]}")
            else:
                print(f"[Epoch {epoch}] eval results: {eval_results[-1]}")

        wandb.log({'epoch': epoch})

    save_model(net, cfg_name, last=True)
    print(f"Model reached highest validation accuracy at epoch {best_epoch}...")
    wandb.log({'best_epoch': best_epoch})

    return loss_per_epoch, eval_results
