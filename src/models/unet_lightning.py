import torch
import lightning.pytorch as pl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics

from .losses import dice_loss_fun


class UNetLightning(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        self.dice_loss = dice_loss_fun

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
        
        y_pred = self.model(images)
        seg_pred = F.softmax(y_pred, dim=1)

        d_loss = self.dice_loss(seg_pred[:, 1, :, :], y_true)
        dice_score = torchmetrics.functional.dice(
            seg_pred, y_true, zero_division=1, num_classes=2, ignore_index=0
        )
        self.log('train_loss', d_loss, on_step=False, on_epoch=True)
        self.log('train_dice', dice_score, on_step=False, on_epoch=True)

        return d_loss
    
    def validation_step(self, batch, batch_idx):
        if len(batch) == 4:
            images, y_true, img_path, mask_path = batch
        else:
            images, y_true = batch
            img_path, mask_path = None, None   
            
        y_pred = self.model(images)
        seg_pred = F.softmax(y_pred, dim=1)

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
            
        y_pred = self.model(images)
        seg_pred = F.softmax(y_pred, dim=1)

        test_name = "in_test" if dataloader_idx == 0 else "out_test"

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
     

    

    