from .unet import UNetMONAI, UNetSD, UNetPolyp
from .sdnet import SDNet, SDNetGlobal, SDNetLocal
from .unet_lightning import UNetLightning
from .sdnet_lightning import SDNetLightning
from .feddis_lightning import FedDisLightning
from .feddis import GlobalFedDis, LocalFedDis
from .missformer import MISSFormer
from .missformer_lightning import MISSFormerLightning
from .utnet import UTNet
from .utnet_lightning import UTNetLightning
from .sd_utnet import SD_UTNet, SD_UTNetLocal, SD_UTNetGlobal

import torch
import sys

def get_model(config, in_channels=3, use_lightning=True, local=True):
    model = None

    if isinstance(config, str):
        model_name = config
        lightning = None
        init = "xavier"
        target_size = 512
        in_channels = in_channels
    else:
        model_name = config.client.model
        lightning = None
        init = config.client.weight_init
        target_size = config.data.target_size
        in_channels = 3 if config.data.dataset == "polypgen" else 1
    
    if model_name == "unet":
        model = UNetSD(width=target_size, height=target_size, in_channels=in_channels)
        lightning = UNetLightning
    elif model_name == "unet_polyp":
        model = UNetPolyp(n_channels=in_channels)
        lightning = UNetLightning
    elif model_name == "unet_monai":
        model = UNetMONAI()
    elif model_name == "sdnet":
        model = SDNet(width=target_size, height=target_size, in_channels=in_channels)
        lightning = SDNetLightning
    elif model_name == "feddis":
        model = LocalFedDis() if local else GlobalFedDis()
        lightning = FedDisLightning
    elif model_name == "sdnet_fed":
        model = SDNetLocal() if local else SDNetGlobal()
    elif model_name == "missformer":
        model = MISSFormer(image_size=target_size)
        lightning = MISSFormerLightning
    elif model_name == "utnet":
        model = UTNet(in_chan=in_channels, num_classes=2)
        lightning = UTNetLightning
    elif model_name == "sd_utnet":
        model = SD_UTNet(in_channels=in_channels)
        lightning = SDNetLightning
    elif model_name == "sd_utnet_fed":
        model = SD_UTNetLocal() if local else SD_UTNetGlobal()
    else:
        print(f"Model {model_name} has not been implemented!")

    if model is not None and init is not None:
        initialize_weights(model, init)
    else:
        print("Skipping model weight initialization...")

    if lightning is None or not use_lightning:
        return model

    return lightning(model, config)

def initialize_weights(model, init):    
    init_func = None
    if init == "xavier":
        init_func = torch.nn.init.xavier_normal_
    elif init == "kaiming":
        init_func = torch.nn.init.kaiming_normal_
    elif init == "gaussian" or init == "normal":
        init_func = torch.nn.init.normal_
      
    if init_func is not None:
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                init_func(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, torch.nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()       
    else:
        print(f"Weight initialization {init} is not supported!")