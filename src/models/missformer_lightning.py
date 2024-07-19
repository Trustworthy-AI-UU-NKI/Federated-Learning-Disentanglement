import torch
import lightning.pytorch as pl
import torch.nn as nn
import torch.optim as optim
import torchmetrics

from monai.losses import DiceLoss
from .losses import dice_loss_fun, KL_divergence


class MISSFormerLightning(pl.LightningModule):
    def __init__(self, model, config, dice_w=1.0, ce_w=0.4):
        super().__init__()
        self.model = model
        self.config = config

        self.dice_loss = dice_loss_fun
        self.ce_loss = nn.CrossEntropyLoss()

        self.dice_w = dice_w
        self.ce_w = ce_w

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return optim.AdamW(self.model.parameters(), lr=self.config.client.lr)

    def training_step(self, batch, batch_idx):
        images, y_true = batch
        seg = self.model(images)

        d_loss = self.dice_loss(seg, y_true)
        # c_loss = self.ce_loss(seg, y_true)

        loss = d_loss * self.dice_w #+ \
            #    c_loss * self.ce_w

        self.log("total_loss", loss, on_step=False, on_epoch=True)
        self.log("dice_loss", d_loss, on_step=False, on_epoch=True)
        # self.log("ce_loss", c_loss, on_step=False, on_epoch=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        images, y_true = batch
        seg = self.model(images)

        d_loss = self.dice_loss(seg, y_true)
        dice_score = torchmetrics.functional.dice(
            seg, y_true, zero_division=1
        )
        self.log_dict({"val_dice": dice_score, "val_loss": d_loss})

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        images, y_true = batch
        seg = self.model(images)

        test_name = "in_test" if dataloader_idx == 0 else "out_test"

        # compute dice score
        dice_score = torchmetrics.functional.dice(
            seg, y_true, zero_division=1
        )
        # recall_score = torchmetrics.functional.classification.binary_recall(
        #     torch.argmax(seg, dim=1, keepdim=True),
        #     y_true,
        #     threshold=0.5,
        #     multidim_average="global",
        #     ignore_index=0,
        #     validate_args=True,
        # )
        # accuracy_score = torchmetrics.functional.classification.binary_accuracy(
        #     torch.argmax(seg, dim=1, keepdim=True),
        #     y_true,
        #     threshold=0.5,
        #     multidim_average="global",
        #     ignore_index=None,
        #     validate_args=True,
        # )

        self.log_dict(
            {
                f"{test_name}_test_dice": dice_score,
                # f"{test_name}_recall": recall_score,
                # f"{test_name}_accuracy": accuracy_score,
            }
        )