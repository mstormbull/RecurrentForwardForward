from enum import Enum
import logging

from RecurrentFF.settings import Settings
from torchvision import transforms
from torch import nn
import torch

import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')


def set_logging():
    """
    Must be called after argparse.
    """
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')


def standardize_layer_activations(layer_activations, epsilon):

    # # Compute mean and standard deviation for prev_layer
    # prev_layer_mean = layer_activations.mean(
    #     dim=1, keepdim=True)
    # prev_layer_std = layer_activations.std(
    #     dim=1, keepdim=True)
    # # Apply standardization
    # prev_layer_stdized = (layer_activations - prev_layer_mean) / \
    #     (prev_layer_std + epsilon)
    # return prev_layer_stdized

    squared_activations = layer_activations ** 2
    mean_squared = torch.mean(squared_activations, dim=1, keepdim=True)
    l2_norm = torch.sqrt(mean_squared + epsilon)

    normalized_activations = layer_activations / l2_norm
    return normalized_activations

    # return layer_activations


# input of dims (frames, batch size, input size)

class TrainInputData:
    def __init__(self, pos_input, neg_input):
        self.pos_input = pos_input
        self.neg_input = neg_input

    def __iter__(self):
        yield self.pos_input
        yield self.neg_input

    def move_to_device_inplace(self, device):
        self.pos_input = self.pos_input.to(device)
        self.neg_input = self.neg_input.to(device)

    def jitter_inplace(self):
        # Define a random jitter transformation
        transform = transforms.RandomAffine(
            degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5)

        # Iterate over the batch dimension
        for i in range(self.pos_input.shape[1]):
            # Apply the same transformation to all frames in a batch
            transformed_image = transform(
                self.pos_input[:, i].reshape(-1, 28, 28))
            self.pos_input[:, i] = transformed_image.reshape(-1, 784)

            transformed_image = transform(
                self.neg_input[:, i].reshape(-1, 28, 28))
            self.neg_input[:, i] = transformed_image.reshape(-1, 784)

        # # Debugging visualization
        # num_samples = self.pos_input.shape[1]
        # for i in range(num_samples):
        #     plt.figure(figsize=(10, 5))

        #     # Displaying positive input
        #     plt.subplot(1, 2, 1)
        #     plt.imshow(self.pos_input[0, i].reshape(
        #         28, 28).cpu().numpy(), cmap='gray')
        #     plt.title(f'Pos Sample {i}')

        #     # Displaying negative input
        #     plt.subplot(1, 2, 2)
        #     plt.imshow(self.neg_input[0, i].reshape(
        #         28, 28).cpu().numpy(), cmap='gray')
        #     plt.title(f'Neg Sample {i}')

        #     plt.show()
        #     input()


# input of dims (frames, batch size, num classes)
class TrainLabelData:
    def __init__(self, pos_labels, neg_labels):
        self.pos_labels = pos_labels
        self.neg_labels = neg_labels

    def __iter__(self):
        yield self.pos_labels
        yield self.neg_labels

    def move_to_device_inplace(self, device):
        self.pos_labels = self.pos_labels.to(device)
        self.neg_labels = self.neg_labels.to(device)


class Activations:
    def __init__(self, current, previous):
        self.current = current
        self.previous = previous

    def __iter__(self):
        yield self.current
        yield self.previous

    def advance(self):
        self.previous = self.current


class OutputLayer(nn.Module):
    def __init__(self, prev_size, label_size) -> None:
        super(OutputLayer, self).__init__()

        self.backward_linear = nn.Linear(
            label_size, prev_size)


class ForwardMode(Enum):
    PositiveData = 1
    NegativeData = 2
    PredictData = 3


def layer_activations_to_badness(layer_activations):
    """
    Computes the 'badness' of activations for a given layer in a neural network
    by taking the mean of the squared values.

    'Badness' in this context refers to the average squared activation value.
    This function is designed to work with PyTorch tensors, which represent the
    layer's activations.

    Args:
        layer_activations (torch.Tensor): A tensor representing activations from
        one layer of a neural network. The tensor has shape (batch_size,
        num_activations), where batch_size is the number of samples processed
        together, and num_activations is the number of neurons in the layer.

    Returns:
        torch.Tensor: A tensor corresponding to the 'badness' (mean of the
        squared activations) of the given layer. The output tensor has shape
        (batch_size,), since the mean is taken over the activation values for
        each sample in the batch.
    """
    badness_for_layer = torch.mean(
        torch.square(layer_activations), dim=1)

    return badness_for_layer


class LatentAverager:
    """
    This class is used for tracking and averaging tensors of the same shape.
    It's useful for collapsing latents in a series of computations.
    """

    def __init__(self):
        """
        Initialize the LatentAverager with an empty sum_tensor and a zero count.
        """
        self.sum_tensor = None
        self.count = 0

    def track_collapsed_latents(self, tensor: torch.Tensor):
        """
        Add the given tensor to the tracked sum.

        :param tensor: A tensor to be tracked.
        :type tensor: torch.Tensor
        :raises AssertionError: If the shape of the tensor does not match the shape of the sum_tensor.
        """
        if self.sum_tensor is None:
            self.sum_tensor = tensor
            self.count = 1
        else:
            assert tensor.shape == self.sum_tensor.shape, "Shape mismatch"
            self.sum_tensor += tensor
            self.count += 1

    def retrieve(self) -> torch.Tensor:
        """
        Retrieve the averaged tensor.

        :return: The averaged tensor.
        :rtype: torch.Tensor
        :raises ValueError: If no tensors have been tracked.
        """
        if self.count == 0:
            raise ValueError("No tensors have been tracked")
        return self.sum_tensor / self.count
