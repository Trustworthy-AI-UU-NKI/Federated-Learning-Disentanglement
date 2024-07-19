import hydra
from omegaconf import OmegaConf
import lightning.pytorch as pl
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

import torch
from models import get_model
from models.unet_lightning import UNetLightning
from data import load_polypgen_centralized, load_config
from data_lits import load_lits_centralized
from utils import extract_data_config
import sys


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config):
    data_config = extract_data_config(config)
    epochs = config.client.epochs*config.server.num_rounds

    pl.seed_everything(config.data.seed)

    print(f"Using the following configuration ({config.name})...\n{OmegaConf.to_yaml(config)}")

    wandb_logger = WandbLogger(project=f"{config.client.model}_tests",
                               config=OmegaConf.to_container(config),
                               name=config.name)
    
    save_path = config.name
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"ckpts/{save_path}",
        save_top_k=1,
        save_last=True,
        monitor="val_dice",
        mode="max",
    )
 
    model = get_model(config)

    trainer = Trainer(
        max_epochs=epochs,
        num_sanity_val_steps=1,
        logger=wandb_logger,
        callbacks=checkpoint_callback,
        enable_progress_bar=False
    )


    if config.data.dataset == "polypgen":
        dataloaders, num_examples = load_polypgen_centralized(center_nr=None, data_config=data_config)
    else:
        dataloaders, num_examples = load_lits_centralized(center_nr=None, data_config=data_config)

    print(f"Succesfully created dataloaders for dataset {config.data.dataset}...\n{num_examples}")

    trainer.fit(model, dataloaders['train'], dataloaders['val'])
    trainer.test(model, dataloaders=[dataloaders['in_test'], dataloaders['out_test']], ckpt_path="best")

if __name__ == "__main__":
    main()