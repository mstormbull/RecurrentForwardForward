from enum import Enum
import logging

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
import torchviz
import wandb

from RecurrentFF.model.model import RecurrentFFNet, InputData, LabelData, TestData
from RecurrentFF.model.constants import EPOCHS, LEARNING_RATE, ITERATIONS, THRESHOLD, DAMPING_FACTOR, EPSILON, DEVICE

NUM_CLASSES = 10
INPUT_SIZE = 784
LAYERS = [200, 200, 200]


def MNIST_loaders(train_batch_size=50000, test_batch_size=10000):
    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))])

    train_loader = DataLoader(
        MNIST('./data/', train=True,
              download=True,
              transform=transform),
        batch_size=train_batch_size, shuffle=True)

    test_loader = DataLoader(
        MNIST('./data/', train=False,
              download=True,
              transform=transform),
        batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Pytorch utils.
    torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(1234)

    wandb.init(
        # set the wandb project where this run will be logged
        project="Recurrent-FF",

        # track hyperparameters and run metadata
        config={
            "architecture": "Recurrent-FF",
            "dataset": "MNIST",
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "layers": str(LAYERS),
            "iterations": ITERATIONS,
            "threshold": THRESHOLD,
            "damping_factor": DAMPING_FACTOR,
            "epsilon": EPSILON,
        }
    )

    # Generate train data.
    train_loader, test_loader = MNIST_loaders()
    x, y_pos = next(iter(train_loader))
    x, y_pos = x.to(DEVICE), y_pos.to(DEVICE)
    train_batch_size = len(x)

    shuffled_labels = torch.randperm(x.size(0))
    y_neg = y_pos[shuffled_labels]

    positive_one_hot_labels = torch.zeros(
        len(y_pos), NUM_CLASSES, device=DEVICE)
    positive_one_hot_labels.scatter_(1, y_pos.unsqueeze(1), 1.0)

    negative_one_hot_labels = torch.zeros(
        len(y_neg), NUM_CLASSES, device=DEVICE)
    negative_one_hot_labels.scatter_(1, y_neg.unsqueeze(1), 1.0)

    input_data = InputData(x, x)
    label_data = LabelData(positive_one_hot_labels, negative_one_hot_labels)

    # Generate test data.
    x, y = next(iter(test_loader))
    x, y = x.to(DEVICE), y.to(DEVICE)
    test_batch_size = len(x)
    labels = y
    test_data = TestData(x, labels)

    # Create and run model.
    model = RecurrentFFNet(train_batch_size, test_batch_size,
                           INPUT_SIZE, LAYERS, NUM_CLASSES).to(DEVICE)

    model.train(input_data, label_data, test_data)

    model.predict(test_data)
