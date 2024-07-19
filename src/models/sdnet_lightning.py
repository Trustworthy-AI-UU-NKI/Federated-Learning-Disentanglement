import torch
import lightning.pytorch as pl
import torch.nn as nn
import torch.optim as optim
import torchmetrics

from monai.losses import DiceLoss
from .losses import dice_loss_fun, KL_divergence


class SDNetLightning(pl.LightningModule):
    def __init__(self, model, config, k1_w=0.01, regress_w=1.0, dice_w=10.0, reco_w=1.0):
        super().__init__()
        self.model = model
        self.config = config

        self.k1_loss = KL_divergence
        self.regress_loss = nn.L1Loss()
        self.dice_loss = dice_loss_fun
        self.reco_loss = nn.L1Loss()

        self.k1_w = k1_w
        self.regress_w = regress_w
        self.dice_w = dice_w
        self.reco_w = reco_w

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return optim.AdamW(self.model.parameters(), lr=self.config.client.lr)

    def training_step(self, batch, batch_idx):
        if len(batch) == 4:
            images, y_true, img_path, mask_path = batch
        else:
            images, y_true = batch
            img_path, mask_path = None, None

        seg_pred, reco, z_out, z_out_tilde, mu, logvar, _ = self.model(images)

        d_loss = self.dice_loss(seg_pred[:, 1, :, :], y_true)
      
        k_loss = self.k1_loss(logvar, mu)
        r1_loss = self.regress_loss(z_out_tilde, z_out)
        r2_loss = self.reco_loss(reco, images)

        loss = d_loss * self.dice_w + \
               k_loss * self.k1_w + \
               r1_loss * self.regress_w + \
               r2_loss * self.reco_w

        self.log("total_loss", loss, on_step=False, on_epoch=True)
        self.log("dice_loss", d_loss, on_step=False, on_epoch=True)
        self.log("kl_loss", k_loss, on_step=False, on_epoch=True)
        self.log("regression_loss", r1_loss, on_step=False, on_epoch=True)
        self.log("reconstruction_loss", r2_loss, on_step=False, on_epoch=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        if len(batch) == 4:
            images, y_true, img_path, mask_path = batch
        else:
            images, y_true = batch
            img_path, mask_path = None, None        
        
        seg_pred, reco, z_out, z_out_tilde, mu, logvar, _ = self.model(images)

        d_loss = self.dice_loss(seg_pred[:, 1, :, :], y_true)
        dice_score = torchmetrics.functional.dice(
            seg_pred, y_true, zero_division=1, num_classes=2, ignore_index=0
        )
        self.log_dict({"val_dice": dice_score, "val_loss": d_loss})

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        if len(batch) == 4:
            images, y_true, img_path, mask_path = batch
        else:
            images, y_true = batch
            img_path, mask_path = None, None
            
        seg_pred, reco, z_out, z_out_tilde, mu, logvar, _ = self.model(images)

        test_name = "in_test" if dataloader_idx == 0 else "out_test"

        # compute dice score
        dice_score = torchmetrics.functional.dice(
            seg_pred, y_true, zero_division=1, num_classes=2, ignore_index=0
        )
        recall_score = torchmetrics.functional.classification.binary_recall(
            torch.argmax(seg_pred, dim=1, keepdim=True),
            y_true,
            threshold=0.5,
            multidim_average="global",
            ignore_index=0,
            validate_args=True,
        )
        accuracy_score = torchmetrics.functional.classification.binary_accuracy(
            torch.argmax(seg_pred, dim=1, keepdim=True),
            y_true,
            threshold=0.5,
            multidim_average="global",
            ignore_index=None,
            validate_args=True,
        )

        self.log_dict(
            {
                f"{test_name}_test_dice": dice_score,
                f"{test_name}_recall": recall_score,
                f"{test_name}_accuracy": accuracy_score,
            }
        )