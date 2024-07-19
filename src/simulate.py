import flwr as fl
import hydra
import torch
import pandas as pd
import os
import shutil
import random
import numpy as np

from omegaconf import OmegaConf
from server import get_strategy
from client import get_client_fn
from data import save_results, load_polypgen_federated, load_model_checkpoint, delete_centers_textfile, POLYP_CENTERS, CKPTS_ROOT
from data_lits import load_lits_federated, LITS_CENTERS
from utils import extract_data_config
from models import get_model
from lightning.pytorch.loggers import WandbLogger
from OLD_evaluate_ckpt import test

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config):
    data_config = extract_data_config(config)
    
    # Fix seed
    os.environ["PL_GLOBAL_SEED"] = str(config.data.seed)
    random.seed(config.data.seed)
    np.random.seed(config.data.seed)
    torch.manual_seed(config.data.seed)
    torch.cuda.manual_seed(config.data.seed)

    # Fixes Ray-related errors on Snellius
    ray_init_args = {
        "ignore_reinit_error": True,
        "include_dashboard": False,
        "num_cpus": 18,
        "num_gpus": 1,
        # "_temp_dir": "/home/pschutte/tmp"
    }

    print(f"Using the following configuration ({config.name})...\n{OmegaConf.to_yaml(config)}")

    load_dataset = load_polypgen_federated if config.data.dataset == "polypgen" else load_lits_federated
    centers = POLYP_CENTERS if config.data.dataset == "polypgen" else LITS_CENTERS
    data_collection = {center_nr: load_dataset(center_nr, data_config) for center_nr in centers
                       if center_nr != config.data.out_center}
    data_collection["total_train_size"] = sum([data_collection[key][1]['train'] for key in data_collection])

    server_dataloaders, num_examples = load_dataset(None, data_config)
    print(f"Succesfully created all dataloaders...\nServer num_examples: {num_examples}")

    wandb_logger = WandbLogger(project=f"federated_{config.client.model}",
                               config=OmegaConf.to_container(config),
                               name=config.name)

    # Directory with saved style parameters of clients in between rounds
    if os.path.isdir(os.path.join(CKPTS_ROOT, f"temp_{config.name}")):
        shutil.rmtree(os.path.join(CKPTS_ROOT, f"temp_{config.name}"))
    os.mkdir(os.path.join(CKPTS_ROOT, f"temp_{config.name}"))

    history = fl.simulation.start_simulation(
        client_fn=get_client_fn(config, data_collection, wandb_logger),
        num_clients=config.server.num_clients,
        client_resources={'num_cpus': config.client.num_cpus,
                          'num_gpus': config.client.num_gpus},
        config=fl.server.ServerConfig(num_rounds=config.server.num_rounds),
        strategy=get_strategy(config, server_dataloaders['val'], wandb_logger),
        ray_init_args=ray_init_args
    )

    save_results(history, config)

    delete_centers_textfile(config.name)

    ### TESTING ###

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = get_model(config, use_lightning=False, local=False).to(device)

    df = pd.DataFrame({
        "in/out-test": [], "ckpt": [], "loss": [], "accuracy": [], "1-loss": [], "dsc": []
    })

    for suffix in ["best", "last"]:
        print(f"{suffix} checkpoint results:")
        load_model_checkpoint(net, f"federated_{config.name}", suffix)
        res = test(net, server_dataloaders['in_test'], device)
        print("[in_test]", res)
        df.loc[len(df)] = {"in/out-test": "in-test", "ckpt": suffix, "loss": res[0], "accuracy": res[1].cpu(), "1-loss": res[2], "dsc": res[3].cpu()}

        res = test(net, server_dataloaders['out_test'], device)
        print("[out_test]", res)
        print()
        df.loc[len(df)] = {"in/out-test": "out-test", "ckpt": suffix, "loss": res[0], "accuracy": res[1].cpu(), "1-loss": res[2], "dsc": res[3].cpu()}

    wandb_logger.log_table(key="test_metrics", dataframe=df)

    shutil.rmtree(os.path.join(CKPTS_ROOT, f"temp_{config.name}"))

if __name__ == "__main__":
    main()