import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from monai.losses import DiceLoss, FocalLoss

def KL_divergence(logvar, mu):
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    return kld.mean()

def dice_loss_fun(pred, target):
    smooth = 0.1

    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    loss = ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)).mean()

    return 1 - loss

class UNetLoss(nn.Module):
    def __init__(self, train, dice_w=1.0, bce_w=0.0):
        super().__init__()
        self.train_mode = train

        self.dice_loss = dice_loss_fun
        # self.bce_loss = nn.BCELoss()

        self.dice_w = dice_w
        self.bce_w = bce_w

    def forward(self, y_true, y_pred):
        seg_pred = F.softmax(y_pred, dim=1)
        d_loss = self.dice_loss(seg_pred[:, 1, :, :], y_true)

        if self.train_mode:
            # b_loss = self.bce_loss(y_pred, y_true)
            loss = d_loss

            return {'train_loss': loss, 'dice_loss': d_loss}
        else:
            return {'dice_loss': d_loss}


class SDNetLoss(nn.Module):
    def __init__(self, train, k1_w=0.01, regress_w=1.0, focal_w=0.0, dice_w=10.0, reco_w=1.0, aug_w=1.0):
        super().__init__()
        self.train_mode = train

        self.k1_loss = KL_divergence
        self.regress_loss = nn.L1Loss()
        self.dice_loss = dice_loss_fun
        self.reco_loss = nn.L1Loss()

        self.k1_w = k1_w
        self.regress_w = regress_w
        self.focal_w = focal_w
        self.dice_w = dice_w
        self.reco_w = reco_w

    def forward(self, y_true, seg_pred, reco, z_out, z_out_tilde, mu, logvar, images):
        d_loss = self.dice_loss(seg_pred[:, 1, :, :], y_true)

        if self.train_mode:
            k_loss = self.k1_loss(logvar, mu)
            r1_loss = self.regress_loss(z_out_tilde, z_out)
            r2_loss = self.reco_loss(reco, images)

            loss = d_loss * self.dice_w + \
                k_loss * self.k1_w + \
                r1_loss * self.regress_w + \
                r2_loss * self.reco_w 

            return {'train_loss': loss, 'dice_loss': d_loss, 'k1_loss': k_loss, 'regress_loss': r1_loss, 'reco_loss': r2_loss}
        else:
            return {'dice_loss': d_loss}
        
    def forward_aug(self, y_true, seg_pred, reco, z_out, mu_tilde, mu, logvar, aug_loss, images):
        d_loss = self.dice_loss(seg_pred[:, 1, :, :], y_true)

        if self.train_mode:
            k_loss = self.k1_loss(logvar, mu)
            r1_loss = self.regress_loss(mu_tilde, z_out)
            r2_loss = self.reco_loss(reco, images)
                
            loss = d_loss * self.dice_w + \
                k_loss * self.k1_w + \
                r1_loss * self.regress_w + \
                r2_loss * self.reco_w 

            return {'train_loss': loss, 'dice_loss': d_loss, 'k1_loss': k_loss, 'regress_loss': r1_loss, 'reco_loss': r2_loss, 'aug_loss': aug_loss}
        else:
            return {'dice_loss': d_loss}

class FedDisLoss(nn.Module):
    def __init__(self, train, alpha=0.2, beta=0.5, dice_w=5.0):
        super().__init__()
        self.train_mode = train

        self.reco_loss = nn.L1Loss()
        # self.scl_loss = nn.CosineSimilarity()
        self.lol_loss = nn.CosineSimilarity()
        self.dice_loss = dice_loss_fun

        self.alpha = alpha
        self.beta = beta
        self.dice_w = dice_w

    def forward(self, y_true, seg, x_, z_s, z_a, z_s_shift, z_s_proj, images):
        d_loss = self.dice_loss(seg[:, 1, :, :], y_true)

        if self.train_mode:
            r_loss = self.reco_loss(x_, images)
            # s_loss = 1 - self.scl_loss(z_s.flatten(start_dim=1), z_s_shift.flatten(start_dim=1))
            l_loss = self.lol_loss(z_a.flatten(start_dim=1), z_s_proj.flatten(start_dim=1))
            l_loss = torch.mean(l_loss.flatten())
            # lcl_loss = l_loss

            loss =  r_loss * self.alpha + \
                    l_loss * (1-self.alpha) + \
                    d_loss * self.dice_w
            
            return {'train_loss': loss, 'dice_loss': d_loss, 'reco_loss': r_loss, 'lol_loss': l_loss}
        else:
            return {'dice_loss': d_loss}