import matplotlib.pyplot as plt
from data import load_results
import numpy as np
import torch
import os
import math

def get_centralized_results(filename):
    results = load_results(filename)

    eval_results = results["eval_results"]
    eval_loss = [m[0] for m in eval_results]
    eval_accuracy = [m[1].cpu().detach().item() for m in eval_results]
    eval_dsc = [m[2] for m in eval_results]

    return {
        "loss_per_epoch":   results["loss_per_epoch"],
        "eval_loss":        eval_loss,
        "eval_accuracy":    eval_accuracy,
        "eval_dsc":         eval_dsc
    }

def get_federated_results(filename):
    results = load_results(filename)
    loss_per_epoch = [loss for m in results.metrics_distributed_fit["loss_per_epoch"] for loss in m[1]]

    eval_loss = [m[1] for m in results.losses_distributed]
    eval_accuracy = [m[1] for m in results.metrics_distributed["accuracy"]]
    eval_dsc = [m[1] for m in results.metrics_distributed["dsc"]]

    return {
        "loss_per_epoch":   loss_per_epoch,
        "eval_loss":        eval_loss,
        "eval_accuracy":    eval_accuracy,
        "eval_dsc":         eval_dsc
    }

def plot_results(results, filename):
    names = ["loss_per_epoch", "eval_loss", "eval_accuracy", "eval_dsc"]
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(filename)
    
    for i in range(len(names)):
        ax = axs[i//2][i%2]
        y = results[names[i]]
        x = np.arange(1, len(y)+1)


        ax.set_xticks(x)
        magnitude = 10**math.floor(math.log(math.ceil(max(y)), 10))
        ax.set_yticks(np.arange(0, math.ceil(max(y))+0.1*magnitude, 0.1*magnitude))
        ax.set_ylim(0, math.ceil(max(y))+0.1*magnitude)
        ax.set_title(names[i])

        ax.plot(x, y)

    plt.savefig(f"plots/{filename}.png")
    print(f"Created plot for {filename}...")

if __name__ == "__main__":
    for f in os.listdir("results/"):
        filename = f[:-4] # remove .pkl extension
        if "centralized" in filename:
            results = get_centralized_results(filename)
        else:
            results = get_federated_results(filename)

        plot_results(results, filename)