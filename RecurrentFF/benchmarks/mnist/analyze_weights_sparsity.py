import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_density_function(weights, layer_idx):
    """Plot the density function of weight sparsities for a specific layer."""

    forward_key = f"inner_layers.layers.{layer_idx}.forward_linear.weight"
    recurrent_key = f"inner_layers.layers.{layer_idx}.lateral_linear.weight"
    backward_key = f"inner_layers.layers.{layer_idx}.backward_linear.weight"

    # Extract and flatten weights
    forward_weights = weights[forward_key].detach().cpu().numpy().flatten()
    recurrent_weights = weights[recurrent_key].detach().cpu().numpy().flatten()
    backward_weights = weights[backward_key].detach().cpu().numpy().flatten()

    weight_types = {
        "forward": forward_weights,
        "recurrent": recurrent_weights,
        "backward": backward_weights
    }

    plt.figure(figsize=(10, 6))

    # Create a set of threshold values based on the percentiles of weights
    thresholds = np.linspace(
        0, np.max([np.max(w) for w in weight_types.values()]), 100)

    for name, weight_values in weight_types.items():
        densities = [np.mean(weight_values < t) for t in thresholds]
        plt.plot(thresholds, densities, label=name)

    plt.xlabel('Thresholds')
    plt.ylabel('Percentage of weights below threshold')
    plt.title(f'Layer {layer_idx} Density Function of Weight Sparsities')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(
        f"img/weight_sparsities/layer_{layer_idx}_density_function.png")


# Assuming you're using the same loading code as before
weights = torch.load("weights_9-2.pth")
layers = 3
for i in range(layers):
    plot_density_function(weights, i)
