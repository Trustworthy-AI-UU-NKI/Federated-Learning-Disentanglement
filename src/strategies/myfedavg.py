import flwr as fl
import torch
from models import get_model
from collections import OrderedDict
import shutil
import os
from train_test import calculate_global_epoch
from data import create_centers_textfile

class MyFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, config, logger, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.model_name = config.client.model
        self.logger = logger
        self.best_dsc = 0.0

        create_centers_textfile(self.config, self.config.data.out_center)

    def aggregate_fit(self, server_round, results, failures):
        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} model parameters...")

            loss_per_epoch = aggregated_metrics['loss_per_epoch']
            
            for epoch, loss_epoch in enumerate(loss_per_epoch):
                self.logger.log_metrics({"aggregated/train_loss": loss_epoch,
                                         "aggregated/epoch": calculate_global_epoch(epoch, self.config.two_stage.aug_epochs,
                                                                                    self.config.two_stage.train_rounds,
                                                                                    self.config.client.epochs, server_round-1)})

            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays =  fl.common.parameters_to_ndarrays(aggregated_parameters)

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            net = get_model(self.config, use_lightning=False, local=False).to(device)

            # Save aggregated_ndarrays
            params_dict = zip(net.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

            torch.save(net.state_dict(), f"ckpts/federated_{self.config.name}_last.pt")

            if server_round == self.config.two_stage.train_rounds:
                torch.save(net.state_dict(), f"ckpts/federated_{self.config.name}_{server_round}.pt")
                print(f"Saved aggregated content parameters as resume checkpoint for round {server_round}")

        create_centers_textfile(self.config, self.config.data.out_center)

        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(self, server_round, results, failures):
        # Call aggregate_evaluate from base class (FedAvg) to aggregate parameters and metrics
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)

        self.logger.log_metrics({f"aggregated/val_loss": loss_aggregated,
                   f"aggregated/val_dice": metrics_aggregated["dsc"],
                   f"aggregated/val_accuracy": metrics_aggregated["accuracy"],
                   f"aggregated/epoch": calculate_global_epoch(0, self.config.two_stage.aug_epochs,
                                                               self.config.two_stage.train_rounds,
                                                               self.config.client.epochs, server_round)-1})

        if metrics_aggregated is not None and metrics_aggregated['dsc'] >= self.best_dsc:
            self.best_dsc = metrics_aggregated['dsc']
            print(f"Aggregated model of round {server_round} has highest dsc so far ({self.best_dsc})...")

            src = os.path.join(os.getcwd(), f"ckpts/federated_{self.config.name}_last.pt")
            dst = os.path.join(os.getcwd(), f"ckpts/federated_{self.config.name}_best.pt")
            shutil.copy(src, dst)

        create_centers_textfile(self.config, self.config.data.out_center)

        return loss_aggregated, metrics_aggregated