# Federated Learning with Disentangled Representations for Heterogeneous Medical Image Segmentation
This repository contains the code of the thesis "Federated Learning with Disentangled Representations for Heterogeneous Medical Image Segmentation" and the paper "FedGS: Federated Gradient Scaling for
Heterogenous Medical Image Segmentation" by Philip Schutte, Valentina Corbetta and Wilson Silva.

The FL implementation of this project created with [Flower](https://github.com/adap/flower/tree/main), an open-source FL framework. The foundation of the FL code was created with the help of the code examples provided in their GitHub repository.

## Requirements

The Python requirements can be installed as a Conda environment and activated as follows:

```
conda env create -f environment.yml
conda activate env_fl
```

## Datasets
The [PolypGen](https://www.nature.com/articles/s41597-023-01981-y) dataset can be downloaded at https://www.synapse.org/Synapse:syn45200214 \
The [LiTS](https://www.sciencedirect.com/science/article/pii/S1361841522003085) dataset can be downloaded at https://competitions.codalab.org/competitions/17094

## Training

With configuration file `{config}`, the training process is initiated as follows:
#### Centralized Training
```
python src/centralized.py experiment={config}
```
#### Federated Training
```
python src/simulate.py experiment={config}
```
The configuration files can be found in [`conf/experiment`](conf/experiment).

Model checkpoints can be downloaded from Hugging Face https://huggingface.co/trustworthy-ai/Federated-Learning-Disentanglement and should be put in [`ckpts`](ckpts).

> **_NOTE:_** Since the training is logged with Weights and Biases (wandb), a wandb account is required. To disable the logging, the `WandbLogger` instantiation in `centralized.py` and `simulate.py` must be replaced with `WandbLogger(mode="disabled")`.
