import torch
import lightning.pytorch as pl
import torch.nn as nn
import torch.optim as optim
import torchmetrics

from .losses import dice_loss_fun


class FedDisLightning(pl.LightningModule):
    def __init__(self, model, config, alpha=0.2, beta=0.5, dice_w=1.0):
        super().__init__()
        self.model = model
        self.config = config

        self.reco_loss = nn.L1Loss()
        self.scl_loss = nn.CosineEmbeddingLoss()
        self.lol_loss = nn.CosineEmbeddingLoss()
        self.dice_loss = dice_loss_fun

        self.alpha = alpha
        self.beta = beta
        self.dice_w = dice_w

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return optim.AdamW(self.model.parameters(), lr=self.config.client.lr)

    def training_step(self, batch, batch_idx):
        images, y_true = batch
        seg, x_, z_s, z_a, z_s_shift, z_s_proj, images = self.model(images)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        d_loss = self.dice_loss(seg[:, 1, :, :], y_true)
        r_loss = self.reco_loss(x_, images)
        # s_loss = self.scl_loss(z_s.flatten(start_dim=1), z_s_shift.flatten(start_dim=1), torch.ones(z_s.shape[0]).to(device))
        l_loss = self.lol_loss(z_a.flatten(start_dim=1), z_s_proj.flatten(start_dim=1), -1*torch.ones(z_a.shape[0]).to(device))
        lcl_loss = l_loss

        loss =  r_loss * self.alpha + \
                lcl_loss * (1-self.alpha) + \
                d_loss * self.dice_w
                
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('dice_loss', d_loss, on_step=False, on_epoch=True)
        self.log('reco_loss', r_loss, on_step=False, on_epoch=True)
        # self.log('scl_loss', s_loss, on_step=False, on_epoch=True)
        self.log('lol_loss', l_loss, on_step=False, on_epoch=True)
        # self.log('lcl_loss', lcl_loss, on_step=False, on_epoch=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        images, y_true = batch
        seg, x_, z_s, z_a, z_s_shift, z_s_proj, images = self.model(images)

        loss = self.dice_loss(seg[:, 1, :, :], y_true)
        dice_score = torchmetrics.functional.dice(
            seg, y_true, zero_division=1, num_classes=2, ignore_index=0
        )
        self.log_dict({"val_dice": dice_score, "val_loss": loss})

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        images, y_true = batch
        seg, x_, z_s, z_a, z_s_shift, z_s_proj, images = self.model(images)

        test_name = "in_test" if dataloader_idx == 0 else "out_test"

        # compute dice score
        dice_score = torchmetrics.functional.dice(
            seg, y_true, zero_division=1, num_classes=2, ignore_index=0
        )
        recall_score = torchmetrics.functional.classification.binary_recall(
            torch.argmax(seg, dim=1, keepdim=True),
            y_true,
            threshold=0.5,
            multidim_average="global",
            ignore_index=0,
            validate_args=True,
        )
        accuracy_score = torchmetrics.functional.classification.binary_accuracy(
            torch.argmax(seg, dim=1, keepdim=True),
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
     

    

    