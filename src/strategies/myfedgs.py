"""
Code is based on Flower baseline https://github.com/adap/flower/tree/main/baselines/fednova
"""
import flwr as fl
from flwr.server.strategy.aggregate import aggregate
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters, NDArrays
from functools import reduce
from typing import Any, Callable, List, Tuple

import torch
from models import get_model
from collections import OrderedDict
import shutil
import os
import numpy as np
from train_test import calculate_global_epoch
from data import create_centers_textfile

class MyFedGS(fl.server.strategy.FedAvg):
    def __init__(self, config, logger, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.model_name = config.client.model
        self.logger = logger
        self.best_dsc = 0.0
        self.best_dsc_small = 0.0

        self.global_momentum_buffer = []
        if self.initial_parameters is not None:
            # Remove state_dict buffers from parameters
            ndarrays = parameters_to_ndarrays(self.initial_parameters)
            net = get_model(self.config, use_lightning=False, local=False)

            param_names_set = {name for name, _ in net.named_parameters()}
            params_dict = zip(net.state_dict().keys(), ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

            self.global_parameters = [val.cpu().numpy() for name, val in state_dict.items() if name in param_names_set]

        self.lr = config.client.lr

        # momentum parameter for the server/strategy side momentum buffer
        self.gmf = config.server.gmf

        create_centers_textfile(self.config, self.config.data.out_center)

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate the results from the clients."""
        if not results:
            return None, {}

        if not self.accept_failures and failures:
            return None, {}

        # Compute tau_effective from summation of local client tau: Eqn-6: Section 4.1
        local_tau = [res.metrics["tau"] for _, res in results]
        tau_eff = np.sum(local_tau)

        aggregate_parameters = []
        aggregate_buffers = []

        net = get_model(self.config, use_lightning=False, local=False)
        buffers_idx = len(list(net.parameters()))

        total_norm = np.sum([res.metrics["local_norm"] for _, res in results])

        for _client, res in results:
            ndarrays = parameters_to_ndarrays(res.parameters)
            params = ndarrays[:buffers_idx]
            buffers = ndarrays[buffers_idx:]

            scale = float(res.metrics["local_norm"]) / total_norm
       
            params_scale = scale

            aggregate_parameters.append((params, params_scale))
            aggregate_buffers.append((buffers, scale))

        agg_cum_gradient = my_aggregate(aggregate_parameters)
        agg_buffers = my_aggregate(aggregate_buffers)

        self.update_global_parameters(agg_cum_gradient)

        # Merge updated global parameters and aggregated buffers into one complete state_dict
        param_names_set = {name for name, _ in net.named_parameters()}
        new_state_dict = OrderedDict()
        i, j = 0, 0

        for key in net.state_dict().keys():
            if key in param_names_set:
                new_state_dict[key] = torch.tensor(self.global_parameters[i])
                i += 1
            else:
                new_state_dict[key] = torch.tensor(agg_buffers[j])
                j += 1

        aggregated_metrics = self.aggregate_fit_metrics(server_round, results, net, new_state_dict)

        all_ndarrays = [val.cpu().numpy() for _, val in new_state_dict.items()]

        create_centers_textfile(self.config, self.config.data.out_center)

        return ndarrays_to_parameters(all_ndarrays), aggregated_metrics

    def update_global_parameters(self, agg_cum_grad):
        """Update the global server parameters by aggregating client gradients."""

        if self.gmf != 0:
            print("Server momentum!")

        for i, layer_cum_grad in enumerate(agg_cum_grad):
            if self.gmf != 0:
                # if first round of aggregation, initialize the global momentum buffer
                if len(self.global_momentum_buffer) < len(agg_cum_grad):
                    buf = layer_cum_grad / self.lr
                    self.global_momentum_buffer.append(buf)
                else:
                    # momentum updates using the global accumulated weights buffer for each layer of network
                    self.global_momentum_buffer[i] *= self.gmf
                    self.global_momentum_buffer[i] += layer_cum_grad / self.lr

                self.global_parameters[i] -= self.global_momentum_buffer[i] * self.lr
            else:
                # weight updated eqn: x_new = x_old - gradient
                # the layer_cum_grad already has all the learning rate multiple
                self.global_parameters[i] -= layer_cum_grad

    def aggregate_fit_metrics(self, server_round, results, net, new_state_dict):
        aggregated_metrics = {}

        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            aggregated_metrics = self.fit_metrics_aggregation_fn(fit_metrics)

            loss_per_epoch = aggregated_metrics['loss_per_epoch']

            for epoch, loss_epoch in enumerate(loss_per_epoch):
                self.logger.log_metrics({"aggregated/train_loss": loss_epoch,
                                        "aggregated/epoch": calculate_global_epoch(epoch, self.config.two_stage.aug_epochs,
                                                                                    self.config.two_stage.train_rounds,
                                                                                    self.config.client.epochs, server_round-1)})

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            net.load_state_dict(new_state_dict, strict=True)

            print(f"Saving round {server_round} model parameters...")

            torch.save(net.state_dict(), f"ckpts/federated_{self.config.name}_last.pt")

            if server_round == self.config.two_stage.train_rounds:
                torch.save(net.state_dict(), f"ckpts/federated_{self.config.name}_{server_round}.pt")
                print(f"Saved aggregated content parameters as resume checkpoint for round {server_round}")

        return aggregated_metrics

    def aggregate_evaluate(self, server_round, results, failures):
        # Call aggregate_evaluate from base class (FedAvg) to aggregate parameters and metrics
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)

        self.logger.log_metrics({f"aggregated/val_loss": loss_aggregated,
                   f"aggregated/val_dice": metrics_aggregated["dsc"],
                   f"aggregated/val_dice_small": metrics_aggregated["dsc_small"],
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

        if metrics_aggregated is not None and self.config.data.load_names and metrics_aggregated['dsc_small'] >= self.best_dsc_small:
            self.best_dsc_small = metrics_aggregated['dsc_small']
            print(f"Aggregated model of round {server_round} has highest dsc_small so far ({self.best_dsc_small})...")

            src = os.path.join(os.getcwd(), f"ckpts/federated_{self.config.name}_last.pt")
            dst = os.path.join(os.getcwd(), f"ckpts/federated_{self.config.name}_best_small.pt")
            shutil.copy(src, dst)

        create_centers_textfile(self.config, self.config.data.out_center)

        return loss_aggregated, metrics_aggregated

def my_aggregate(results: List[Tuple[NDArrays, float]]) -> NDArrays:
    """Compute weighted average with scales that sum to 1."""

    # Create a list of weights, each multiplied by the related scales
    weighted_weights = [
        [layer * scale for layer in weights] for weights, scale in results
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates)
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime