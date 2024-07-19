import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import SDNetLoss, dice_loss_fun, KL_divergence
from .layers.blocks import *
from .sdnet import Segmentor, Decoder
from .utnet import UTNet

import os

class AnatomyEncoder(nn.Module):
    def __init__(self, width, height, ndf, in_channels, num_output_channels, norm, upsample):
        super().__init__()
        """
        UNet encoder for the anatomy factors of the image
        num_output_channels: number of spatial (anatomy) factors to encode
        """
        self.width = width 
        self.height = height
        self.ndf = ndf
        self.in_channels = in_channels
        self.num_output_channels = num_output_channels
        self.norm = norm
        self.upsample = upsample

        self.utnet = UTNet(self.in_channels, self.ndf, self.num_output_channels)

    def forward(self, x):
        out = self.utnet(x)
        out = torch.tanh(out)

        return out 
    
class ModalityEncoder(nn.Module):
    def __init__(self, z_length, in_channels, target_size):
        super().__init__()
        """
        VAE encoder to extract intensity (modality) information from the image
        z_length: length of the output vector
        """
        self.z_length = z_length

        start_channels = in_channels

        # self.block1 = conv_bn_lrelu(9, 16, 3, 2, 1)
        self.block1 = conv_bn_lrelu(start_channels, 16, 3, 2, 1)
        self.block2 = conv_bn_lrelu(16, 32, 3, 2, 1)
        self.block3 = conv_bn_lrelu(32, 64, 3, 2, 1)
        self.block4 = conv_bn_lrelu(64, 128, 3, 2, 1)
        # self.fc = nn.Linear(25088, 32)
        self.fc = nn.Linear((target_size**2)//2, 32)
        self.norm = nn.BatchNorm1d(32)
        self.activ = nn.LeakyReLU(0.03, inplace=True)
        self.mu = nn.Linear(32, self.z_length)
        self.logvar = nn.Linear(32, self.z_length)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        
        return mu + eps*std

    def encode(self, x):
        return self.mu(x), self.logvar(x)

    def forward(self, x):
        out = x
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.fc(out.view(-1, out.shape[1] * out.shape[2] * out.shape[3]))
        out = self.norm(out) 
        out = self.activ(out)

        mu, logvar = self.encode(out)
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar

class SD_UTNet(nn.Module):
    def __init__(self, width=512, height=512, num_classes=1, ndf=32, in_channels=3,
                 z_length=8, norm="batchnorm", upsample="nearest",
                 anatomy_out_channels=8, num_mask_channels=8):
        """
        Args:
            width: input width
            height: input height
            upsample: upsampling type (nearest | bilateral)
            num_classes: number of semantic segmentation classes
            z_length: number of modality factors
            anatomy_out_channels: number of anatomy factors
            norm: feature normalization method (BatchNorm)
            ndf: number of feature channels
        """
        super().__init__()
        self.h = height
        self.w = width
        self.ndf = ndf
        self.in_channels = in_channels
        self.z_length = z_length
        self.anatomy_out_channels = anatomy_out_channels
        self.norm = norm
        self.upsample = upsample
        self.num_classes = num_classes
        self.num_mask_channels = num_mask_channels

        self.m_encoder = ModalityEncoder(self.z_length, self.in_channels, self.h)
        self.a_encoder = AnatomyEncoder(self.h, self.w, self.ndf, self.in_channels,
                                        self.anatomy_out_channels, self.norm, self.upsample)
        self.segmentor = Segmentor(self.anatomy_out_channels, self.num_classes)
        self.decoder = Decoder(self.anatomy_out_channels, self.z_length, self.in_channels)

    def forward(self, x):
        a_out = self.a_encoder(x)
        seg_pred = self.segmentor(a_out)
        z_out, mu_out, logvar_out = self.m_encoder(x)
        reco = self.decoder(a_out, z_out)

        if self.training:
            z_out_tilde, _ , _ = self.m_encoder(reco)
        else:
            # dummy assignments, not needed during validation
            z_out_tilde = z_out

        return seg_pred, reco, z_out, z_out_tilde, mu_out, logvar_out, x
    
    def get_criterion(self, train=True):
        criterion = SDNetLoss(train)

        return criterion
    
class SD_UTNetLocal(nn.Module):
    def __init__(self, width=512, height=512, num_classes=1, ndf=32, in_channels=3,
                 z_length=8, norm="batchnorm", upsample="nearest",
                 anatomy_out_channels=8, num_mask_channels=8):
        super().__init__()
        self.h = height
        self.w = width
        self.ndf = ndf
        self.in_channels = in_channels
        self.z_length = z_length
        self.anatomy_out_channels = anatomy_out_channels
        self.norm = norm
        self.upsample = upsample
        self.num_classes = num_classes
        self.num_mask_channels = num_mask_channels

        self.m_encoder = ModalityEncoder(self.z_length, self.in_channels, self.h)
        self.content_encoder = AnatomyEncoder(self.h, self.w, self.ndf, self.in_channels,
                                        self.anatomy_out_channels, self.norm, self.upsample)
        self.content_segmentor = Segmentor(self.anatomy_out_channels, self.num_classes)
        self.decoder = Decoder(self.anatomy_out_channels, self.z_length, self.in_channels)

    def forward(self, x):
        a_out = self.content_encoder(x)
        seg_pred = self.content_segmentor(a_out)
        z_out, mu_out, logvar_out = self.m_encoder(x)
        reco = self.decoder(a_out, z_out)

        if self.training:
            z_out_tilde, _ , _ = self.m_encoder(reco)
        else:
            # dummy assignments, not needed during validation
            z_out_tilde = z_out

        return seg_pred, reco, z_out, z_out_tilde, mu_out, logvar_out, x
    
    def forward_aug(self, x, center_nr, server_round, aug_center_nrs, save_path):
        self.k1_loss = KL_divergence
        self.regress_loss = nn.L1Loss()
        self.reco_loss = nn.L1Loss()
        self.anam_loss = nn.L1Loss()
        self.k1_w = 0.01
        self.regress_w = 1.0
        self.reco_w = 1.0
        self.aug_w = 1.0

        a_out = self.content_encoder(x)
        seg_pred = self.content_segmentor(a_out)

        for name, param in self.named_parameters():
            if "content" not in name:
                param.requires_grad = False

        total_aug_loss = 0
        for aug_center_nr in aug_center_nrs:
            aug_save_path = save_path.replace(f'client{center_nr}', f'client{aug_center_nr}')

            style_ckpt = torch.load(f"{aug_save_path}_style_round{server_round-1}.pt")
            self.load_state_dict(style_ckpt, strict=False)

            del style_ckpt
            torch.cuda.empty_cache() 

            aug_z_out, aug_mu_out, aug_logvar_out = self.m_encoder(x)
            aug_reco = self.decoder(a_out, aug_z_out)
            aug_reco = aug_reco.detach()
            aug_a_out_tilde = self.content_encoder(aug_reco)

            aug_loss = self.anam_loss(aug_a_out_tilde, a_out) * self.aug_w * (1/len(aug_center_nrs))
            aug_loss.backward(retain_graph=True)
            total_aug_loss += aug_loss.item()

        for name, param in self.named_parameters():
            if "content" not in name:
                param.requires_grad = True
        
        # save_path = os.path.join("ckpts/", f"temp/client{center_nr}")
        style_ckpt = torch.load(f"{save_path}_style_round{server_round-1}.pt")
        self.load_state_dict(style_ckpt, strict=False)

        z_out, mu_out, logvar_out = self.m_encoder(x)
        reco = self.decoder(a_out, z_out)
        _, mu_out_tilde, _ = self.m_encoder(reco)

        return seg_pred, reco, z_out, mu_out_tilde, mu_out, logvar_out, total_aug_loss, x
    
    def get_criterion(self, train=True):
        criterion = SDNetLoss(train)

        return criterion

    
class SD_UTNetGlobal(nn.Module):
    def __init__(self, width=512, height=512, num_classes=1, ndf=32, in_channels=3,
                 z_length=8, norm="batchnorm", upsample="nearest",
                 anatomy_out_channels=8, num_mask_channels=8):
        super().__init__()
        self.h = height
        self.w = width
        self.ndf = ndf
        self.in_channels = in_channels
        self.z_length = z_length
        self.anatomy_out_channels = anatomy_out_channels
        self.norm = norm
        self.upsample = upsample
        self.num_classes = num_classes
        self.num_mask_channels = num_mask_channels

        self.encoder = AnatomyEncoder(self.h, self.w, self.ndf, self.in_channels,
                                        self.anatomy_out_channels, self.norm, self.upsample)
        self.segmentor = Segmentor(self.anatomy_out_channels, self.num_classes)

    def forward(self, x):
        a_out = self.encoder(x)
        seg_pred = self.segmentor(a_out)

        return seg_pred, None, None, None, None, None, x
    
    def get_criterion(self, train=True):
        criterion = SDNetLoss(train)

        return criterion