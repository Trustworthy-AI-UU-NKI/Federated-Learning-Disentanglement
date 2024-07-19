import torch
import flwr as fl
from collections import OrderedDict
from data import POLYP_CENTERS, CKPTS_ROOT, get_center_from_textfile
from utils import extract_data_config
from train_test import train, test, train_aug
from models import get_model
from optimizers.myadamw import MyAdamW

import sys
import os

# models which have content-style parameter disentanglement
DISENTANGLED = ["feddis", "sdnet_fed", "sd_utnet_fed"]
TRAIN_ROUND, AUG_ROUND = 0, 1

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, center_nr, config, data, logger):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.center_nr = center_nr
        self.net = get_model(config, use_lightning=False).to(self.device)
        self.disentangled = config.client.model in DISENTANGLED
        self.resume_from_round = config.server.resume_from_round
        self.save_resume_enabled = config.server.save_resume_enabled
        self.config_name = config.name

        self.two_stage = config.two_stage.enabled
        self.train_rounds = config.two_stage.train_rounds

        dataloaders, num_examples = data
        self.dataloaders = dataloaders
        self.num_examples = num_examples
        self.logger = logger
        self.save_path = os.path.join(CKPTS_ROOT, f"temp_{self.config_name}/client{self.center_nr}")

        self.save_resume_ckpt_path = os.path.join(CKPTS_ROOT, f"federated_{self.config_name}_{self.train_rounds}_client{self.center_nr}.pt")

        if self.resume_from_round is not None:
            self.load_resume_ckpt_path = os.path.join(CKPTS_ROOT, f"federated_{config.server.resume_cfg_name}_{self.resume_from_round}_client{self.center_nr}.pt")
        else:
            self.load_resume_ckpt_path = None

    def get_parameters(self, config):
        if self.disentangled:
            params = [val.cpu().numpy() for name, val in self.net.state_dict().items()
                      if "content" in name]
        else:
            params = [val.cpu().numpy() for _, val in self.net.state_dict().items()]

        return params

    def save_state_dict(self, server_round, param_type="style", round_type=TRAIN_ROUND):
        if param_type == "style" or param_type == "both":
            style_state_dict = {k: self.net.state_dict()[k] for k in self.net.state_dict().keys()
                                if "content" not in k}
            torch.save(style_state_dict, f"{self.save_path}_style_round{server_round}.pt")

            if server_round == self.train_rounds and self.save_resume_enabled:
                torch.save(style_state_dict, self.save_resume_ckpt_path)
                self.print_client(f"Saved style parameters as resume checkpoint for round {server_round}")

            if server_round > 1:
                # os.remove(f"{self.save_path}_style_round{server_round-1}.pt")
                old_style_path = f"{self.save_path}_style_round{server_round-2}.pt"
                if os.path.exists(old_style_path):
                    os.remove(old_style_path)

        if param_type == "content" or param_type == "both":
            content_state_dict = {k: self.net.state_dict()[k] for k in self.net.state_dict().keys()
                                if "content" in k}
            torch.save(content_state_dict, f"{self.save_path}_content_round{server_round}.pt")

            if server_round > 2:
                os.remove(f"{self.save_path}_content_round{server_round-2}.pt")

    def set_parameters_one_stage(self, parameters, server_round, mode="fit"):
        if self.disentangled:
            # Content parameters
            content_keys = [key for key in self.net.state_dict().keys() if "content" in key]
            params_dict = zip(content_keys, parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

            # Style parameters
            if server_round < 1 or (server_round == 1 and mode == "fit"):
                if self.resume_from_round is not None:
                    style_ckpt = torch.load(self.load_resume_ckpt_path)
                    for key in style_ckpt.keys():
                        state_dict[key] = style_ckpt[key]
                    self.net.load_state_dict(state_dict, strict=True)
                else:
                    # At start of training there are no style params saved yet
                    self.net.load_state_dict(state_dict, strict=False)
            else:
                load_round = server_round-1 if mode == "fit" else server_round
                style_ckpt = torch.load(f"{self.save_path}_style_round{load_round}.pt")

                for key in style_ckpt.keys():
                    state_dict[key] = style_ckpt[key]

                self.net.load_state_dict(state_dict, strict=True)
        else:
            params_dict = zip(self.net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.net.load_state_dict(state_dict, strict=True)

    def set_parameters_two_stage(self, parameters, server_round, mode="fit", round_type=TRAIN_ROUND):
        if self.disentangled:
            ### CONTENT PARAMETERS ###
            # if server_round < 1 or (round_type == TRAIN_ROUND and mode == "fit") or (round_type == AUG_ROUND and mode == "eval"):
            content_keys = [key for key in self.net.state_dict().keys() if "content" in key]
            params_dict = zip(content_keys, parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            debug_content = "use aggregated"

            ### STYLE PARAMETERS ###
            if server_round < 1 or (server_round == 1 and mode == "fit"):
                if self.resume_from_round is not None:
                    style_ckpt = torch.load(self.load_resume_ckpt_path)
                    for key in style_ckpt.keys():
                        state_dict[key] = style_ckpt[key]
                    self.net.load_state_dict(state_dict, strict=True)
                    debug_style = f"loaded resume ckpt round{self.resume_from_round}"
                else:
                    # At start of training there are no style params saved yet
                    self.net.load_state_dict(state_dict, strict=False)
                    debug_style = f"no style ckpt available"
            else:
                load_round = server_round-1 if mode == "fit" else server_round
                style_ckpt = torch.load(f"{self.save_path}_style_round{load_round}.pt")
                debug_style = f"loaded ckpt round{load_round}"

                for key in self.net.state_dict().keys():
                    if "content" not in key:
                        state_dict[key] = style_ckpt[key]

                self.net.load_state_dict(state_dict, strict=True)

            self.print_client(f"[ROUND {server_round}] CONTENT: {debug_content}, STYLE: {debug_style}")
        else:
            params_dict = zip(self.net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        if self.two_stage:
            return self.fit_two_stage(parameters, config)
        else:
            return self.fit_one_stage(parameters, config)

    def fit_one_stage(self, parameters, config):
        self.set_parameters_one_stage(parameters, config['server_round'], "fit")
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=config["lr"])

        loss_per_epoch = train(self.net, optimizer, self.dataloaders['train'],
                               self.device, self.center_nr, self.logger, **config)

        self.save_state_dict(config["server_round"], param_type="style")

        return self.get_parameters(config={}), self.num_examples['train'], \
               {"center_nr": self.center_nr,
                "losses": loss_per_epoch}

    def fit_two_stage(self, parameters, config):
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=config["lr"])

        ### TRAINING ROUND ###
        if self.get_two_stage_round(config['server_round']) == TRAIN_ROUND:
            self.set_parameters_two_stage(parameters, config['server_round'], "fit", round_type=TRAIN_ROUND)

            loss_per_epoch = train(self.net, optimizer, self.dataloaders['train'],
                                   self.device, self.center_nr, self.logger, **config)

            self.save_state_dict(config["server_round"], param_type="style", round_type=TRAIN_ROUND)

            return self.get_parameters(config={}), self.num_examples['train'], \
               {"center_nr": self.center_nr,
                "losses": loss_per_epoch}
        ### AUGMENTATION ROUND ###
        else:
            self.set_parameters_two_stage(parameters, config['server_round'], "fit", round_type=AUG_ROUND)
            loss_per_epoch = train_aug(self.net, optimizer, self.dataloaders['train'], self.device,
                                       self.save_path, self.center_nr, self.logger, **config)
            self.save_state_dict(config["server_round"], param_type="style", round_type=AUG_ROUND)

            return self.get_parameters(config={}), self.num_examples['train'], \
               {"center_nr": self.center_nr,
                "losses": loss_per_epoch}

    def get_two_stage_round(self, server_round):
        if server_round <= self.train_rounds:
            return TRAIN_ROUND
        else:
            if server_round == self.train_rounds+1:
                self.print_client("Starting augmentation rounds...")

            return AUG_ROUND


    def evaluate(self, parameters, config):
        if self.two_stage:
            self.set_parameters_two_stage(parameters, config['server_round'], "eval",
                                          round_type=self.get_two_stage_round(config['server_round']))
        else:
            self.set_parameters_one_stage(parameters, config['server_round'], "eval")

        loss, accuracy, dsc, _dsc_small = test(self.net, self.dataloaders['val'], self.device,
                                   self.center_nr, self.logger, **config)

        return float(loss), self.num_examples['val'], \
               {"center_nr": self.center_nr,
                "accuracy": float(accuracy),
                "dsc": float(dsc)}

    def print_client(self, msg):
        print(f"[client-{self.center_nr}] {msg}")


class FlowerClientGrad(fl.client.NumPyClient):
    def __init__(self, center_nr, config, data, total_train_size, logger):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.center_nr = center_nr
        self.net = get_model(config, use_lightning=False).to(self.device)

        self.config_name = config.name
        self.train_rounds = config.two_stage.train_rounds

        dataloaders, num_examples = data
        self.num_examples = num_examples
        self.dataloaders = dataloaders
        self.ratio = num_examples['train'] / total_train_size
        self.optimizer = MyAdamW(self.net.parameters(), self.ratio, lr=config.client.lr)
        self.logger = logger

    def get_parameters(self, config):
        params = [val["cum_grad"].cpu().numpy() for _, val in
                  self.optimizer.state_dict()["state"].items()]

        param_names_set = {name for name, _ in self.net.named_parameters()}

        state_dict_buffers = [val.cpu().numpy() for name, val in self.net.state_dict().items()
                              if name not in param_names_set]

        params.extend(state_dict_buffers)

        return params

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)
        # self.optimizer.set_model_parameters(parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        loss_per_epoch = train(self.net, self.optimizer, self.dataloaders['train'],
                               self.device, self.center_nr, self.logger, **config)

        metrics = {"center_nr": self.center_nr, "losses": loss_per_epoch}
        metrics.update(self.optimizer.get_gradient_scaling())

        return self.get_parameters(config={}), self.num_examples['train'], metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        loss, accuracy, dsc, dsc_small = test(self.net, self.dataloaders['val'], self.device,
                                   self.center_nr, self.logger, **config)

        return float(loss), self.num_examples['val'], \
               {"center_nr": self.center_nr,
                "accuracy": float(accuracy),
                "dsc": float(dsc),
                "dsc_small": float(dsc_small)}

    def print_client(self, msg):
        print(f"[client-{self.center_nr}] {msg}")


def get_client_fn(config, data_collection, logger):
    def client_fn(cid):
        """Function to create client, passed as arg to server simulation."""
        center_nr = get_center_from_textfile(config.name)
        if config.server.gradient_mode:
            total_train_size = data_collection['total_train_size']
            return FlowerClientGrad(center_nr, config, data_collection[center_nr], total_train_size, logger)

        if not config.server.gradient_mode:
            return FlowerClient(center_nr, config, data_collection[center_nr], logger)

    return client_fn

if __name__ == "__main__":
    cid = sys.argv[0] if len(sys.argv) > 0 else 0
    client = FlowerClient(cid)
    fl.client.start_numpy_client(server_address="[::]:8080", client=client)
