import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from RecurrentFF.benchmarks.mnist.mnist import DATA_SIZE, ITERATIONS, NUM_CLASSES, \
    TRAIN_BATCH_SIZE, MNIST_loaders, DATASET
from RecurrentFF.model.data_scenario.processor import DataScenario
from RecurrentFF.util import set_logging
from RecurrentFF.model.model import RecurrentFFNet
from RecurrentFF.settings import Settings, DataConfig

TEST_BATCH_SIZE = 1
NUM_BATCHES = 1000

TENSOR_PATH = "MNIST-10-pre-15-post-(more layers)-95%.pth"

if __name__ == "__main__":
    settings = Settings.new()

    data_config = {
        "data_size": DATA_SIZE,
        "num_classes": NUM_CLASSES,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "test_batch_size": TEST_BATCH_SIZE,
        "iterations": ITERATIONS,
        "dataset": DATASET}

    if settings.data_config is None:
        settings.data_config = DataConfig(**data_config)

    settings.model.should_log_metrics = False

    set_logging()

    # Pytorch utils.
    torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(1234)

    # Generate train data.
    train_loader, test_loader = MNIST_loaders(
        settings.data_config.train_batch_size, settings.data_config.test_batch_size)

    # Create and run model.
    model = RecurrentFFNet(settings).to(settings.device.device)

    model.load_state_dict(torch.load(
        TENSOR_PATH, map_location=settings.device.device))

    _train_loader, test_loader_tmp = MNIST_loaders(
        settings.data_config.train_batch_size, 1000)
    model.predict(DataScenario.StaticSingleClass,
                  test_loader_tmp, 1, write_activations=False)

    input("Does the accuracy look good?")

    reps = 1
    numImages = 20

    hidden, lbls = model.processor.matt_05_14_2024_predict(test_loader_tmp, numImages = numImages, alpha = 0.5, replicates = reps)
    l2layer = []
    labels = []
    for t in range(len(hidden)):
        l2layer.append(torch.mean(torch.sqrt(hidden[t].detach().cpu()**2), dim = 1))
        labels.append(lbls[t].detach().cpu())

    cmap = cm.Blues(np.linspace(0,1,6))
    l2layerplot = torch.stack(l2layer)
    labelsplot = torch.stack(labels)
    fig, ax = plt.subplots(2,1, figsize = [10,6])
    for i in range(5):
        ax[0].plot(l2layerplot[:,i], color = cmap[i+1,:], lw = 3, label = 'layer ' + str(i+1))
    for r in range(numImages):
        ax[0].plot([(1+r)*reps*10, (1+r)*reps*10], [0, 1.5], '--r')
    ax[0].legend(loc='best')
    ax[1].plot(labelsplot, lw = 3)
    ax[1].set_xlabel('Time')
    ax[0].set_ylabel('L2 activation')
    ax[1].set_ylabel('Label magnitude')
    plt.tight_layout()
    plt.savefig('alpha0pt5_20images_rep1_zero.png')
    plt.show()
    