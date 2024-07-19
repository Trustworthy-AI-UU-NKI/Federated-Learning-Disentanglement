import flwr as fl
import numpy as np
import torch
from collections import OrderedDict
from data import load_model_parameters, POLYP_CENTERS, CKPTS_ROOT
from strategies.myfedavg import MyFedAvg
from strategies.myfedgs import MyFedGS
from train_test import train, test
from models import get_model
import os

def create_fit_config_fn(config):
    def fit_config_fn(server_round):
        """
        Function that returns config that is used by clients during training,
        i.e. this config is passed to client's fit() method.
        """
        return {
            'epochs': config.client.epochs,
            'aug_epochs': config.two_stage.aug_epochs,
            'train_rounds': config.two_stage.train_rounds,
            'lr': config.client.lr,
            'server_round': server_round,
            'p_factor_enabled': config.client.p_factor_enabled,
            'my_details': config.data.my_details,
            'dataset': config.data.dataset
        }

    return fit_config_fn

def create_evaluate_config_fn(config):
    def evaluate_config_fn(server_round):
        """
        Function that returns config that is used by clients during training,
        i.e. this config is passed to client's evaluate() method.
        """
        return {
            'epochs': config.client.epochs,
            'aug_epochs': config.two_stage.aug_epochs,
            'train_rounds': config.two_stage.train_rounds,
            'server_round': server_round,
            'my_details': config.data.my_details,
            'dataset': config.data.dataset

        }

    return evaluate_config_fn

def create_evaluate_fn(dataloader, my_config, logger):
    def evaluate(server_round, parameters, config):
        """
        Function that returns config that is used by clients during training,
        i.e. this config is passed to client's evaluate() method.
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = get_model(my_config, use_lightning=False, local=False).to(device)
        config = {'epochs': my_config.client.epochs,
                  'aug_epochs': my_config.two_stage.aug_epochs,
                  'train_rounds': my_config.two_stage.train_rounds,
                  'server_round': server_round,
                  'my_details': my_config.data.my_details,
                  'dataset': my_config.data.dataset}
     
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        loss, accuracy, dsc, dsc_small = test(model, dataloader, device, "server", logger, **config)

        return loss, {"dsc": dsc, "dsc_small": dsc_small, "accuracy": accuracy}

    return evaluate

def get_strategy(config, dataloader, logger):
    def get_initial_params():
        if config.server.resume_cfg_name is None:
            model = get_model(config, use_lightning=False, local=False)
            state_dict = model.state_dict()
        elif config.server.resume_from_round is not None:
            path = f"federated_{config.server.resume_cfg_name}_{config.server.resume_from_round}.pt"
            state_dict = torch.load(os.path.join(CKPTS_ROOT, path))
            print(f"Loaded resume checkpoint from path: {path}")

            # Clients start with augmentation so they need style checkpoints of other clients
            if config.two_stage.train_rounds == 0 and config.two_stage.aug_epochs is not None:
                centers = POLYP_CENTERS.copy()
                centers.remove(config.data.out_center)
                for center in centers:
                    path = f"federated_{config.server.resume_cfg_name}_{config.server.resume_from_round}_client{center}.pt"
                    style_ckpt = torch.load(os.path.join(CKPTS_ROOT, path))
                    center_path = f"temp/client{center}_style_round0.pt"
                    torch.save(style_ckpt, os.path.join(CKPTS_ROOT, center_path)) 
        else:
            path = f"federated_{config.server.resume_cfg_name}_last.pt"
            state_dict = torch.load(os.path.join(CKPTS_ROOT, path))
            print(f"Loaded resume checkpoint from path: {path}")

        params = [val.cpu().numpy() for _, val in state_dict.items()]
        return fl.common.ndarrays_to_parameters(params)
    
    if config.server.gradient_mode:
        strategy = MyFedGS
    else:
        strategy = MyFedAvg

    print(f"Using FL strategy {strategy}")

    return strategy(
        config=config,
        logger=logger,
        min_available_clients=config.server.num_clients,
        min_fit_clients=config.server.num_clients,
        min_evaluate_clients=config.server.num_clients,
        evaluate_fn=create_evaluate_fn(dataloader, config, logger),
        on_fit_config_fn=create_fit_config_fn(config),
        on_evaluate_config_fn=create_evaluate_config_fn(config),
        fit_metrics_aggregation_fn=fit_weighted_average,
        evaluate_metrics_aggregation_fn=eval_weighted_average,
        accept_failures=False,
        initial_parameters=get_initial_params()
    )

def fit_weighted_average(metrics):
    """
    Aggregation function for (federated) training loss for every epoch, i.e.
    those returned by the client's fit() method.
    """
    loss_per_client_per_epoch = np.array([num_examples * np.array(m["losses"]) \
                                          for num_examples, m in metrics])
    loss_per_epoch = np.sum(loss_per_client_per_epoch, axis=0)
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"loss_per_epoch": loss_per_epoch / sum(examples)}

def eval_weighted_average(metrics):
    """
    Aggregation function for (federated) evaluation metrics, i.e. those returned
    by the client's evaluate() method.
    """
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    dscs = [num_examples * m["dsc"] for num_examples, m in metrics]

    if "dsc_small" in metrics[0][1]:
        dscs_small = [num_examples * m["dsc_small"] for num_examples, m in metrics]
    else:
        dscs_small = []

    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {
        "accuracy": sum(accuracies) / sum(examples),
        "dsc": sum(dscs) / sum(examples),
        "dsc_small": sum(dscs_small) / sum(examples)
    }

if __name__ == "__main__":
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=get_strategy()
    )