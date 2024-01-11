import random
from multiprocessing import Process

import torch
import wandb

from RecurrentFF.benchmarks.mnist.mnist import MNIST_loaders
from RecurrentFF.model.model import RecurrentFFNet
from RecurrentFF.settings import DataConfig, Settings
from RecurrentFF.util import set_logging

EPOCHS = 1
DATA_SIZE = 784
NUM_CLASSES = 10
TRAIN_BATCH_SIZE = 500
TEST_BATCH_SIZE = 5000
DATASET = "MNIST"

DEVICE = "mps"


def run(settings: Settings):
    # Needs to be done here as well because multiprocessing.
    set_logging()

    # Pytorch utils.
    torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(1234)

    print(settings.model_dump())

    wandb.init(
        # set the wandb project where this run will be logged
        project="Recurrent-FF",

        # track hyperparameters and run metadata
        config={
            "architecture": "Recurrent-FF",
            "dataset": "MNIST",
            "settings": settings.model_dump(),
        }
    )

    # Generate train data.
    train_loader, test_loader = MNIST_loaders(
        settings.data_config.train_batch_size, settings.data_config.test_batch_size)

    # Create and run model.
    model = RecurrentFFNet(settings).to(settings.device.device)

    try:
        model.train(train_loader, test_loader)
        print("======================FINISHED RUN============================")
    except KeyboardInterrupt:
        print("======================FINISHED RUN============================")
        exit(0)


if __name__ == "__main__":
    set_logging()

    loss_thresholds = [1.5]

    iterations = [15]

    hidden_sizes = [[500, 500, 500]]

    ff_act = ["relu"]

    ff_optimizers = ["rmsprop"]
    classifier_optimizers = ["rmsprop"]

    ff_rmsprop_momentums = [0.0]
    ff_rmsprop_learning_rates = [
        0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005]
    classifier_rmsprop_momentums = [0.0]
    classifier_rmsprop_learning_rates = [0.0001]

    ff_adam_learning_rates = [0.00001]
    classifier_adam_learning_rates = [0.00001]

    ff_adadelta_learning_rates = [0.00001]
    classifier_adadelta_learning_rates = [0.00001]

    # train_batch_sizes = [100, 200, 500, 1000]
    train_batch_sizes = [5000]
    damping_factors = [0.3, 0.4, 0.5, 0.6, 0.7]

    loss_scale_ffs = [1, 3, 5, 8, 13, 21]
    loss_scale_predictives = [1, 3, 5, 8, 13, 21]
    loss_scale_hebbians = [1, 3, 5, 8, 13, 21]
    loss_scale_decorrelatives = [1, 3, 5, 8, 13, 21]

    while True:
        # random hyperparams
        loss_threshold = random.choice(loss_thresholds)
        iterations_ = random.choice(iterations)
        hidden_sizes_ = random.choice(hidden_sizes)
        act = random.choice(ff_act)
        ff_opt = random.choice(ff_optimizers)
        classifier_opt = random.choice(classifier_optimizers)
        ff_rmsprop_momentum = random.choice(ff_rmsprop_momentums)
        ff_rmsprop_learning_rate = random.choice(ff_rmsprop_learning_rates)
        classifier_rmsprop_momentum = random.choice(
            classifier_rmsprop_momentums)
        classifier_rmsprop_learning_rate = random.choice(
            classifier_rmsprop_learning_rates)
        ff_adam_learning_rate = random.choice(ff_adam_learning_rates)
        classifier_adam_learning_rate = random.choice(
            classifier_adam_learning_rates)
        train_batch_size = random.choice(train_batch_sizes)
        ff_adadelta_learning_rate = random.choice(ff_adadelta_learning_rates)
        classifier_adadelta_learning_rate = random.choice(
            classifier_adadelta_learning_rates)
        damping_factor = random.choice(damping_factors)

        loss_scale_ff = random.choice(loss_scale_ffs)
        loss_scale_predictive = random.choice(loss_scale_predictives)
        loss_scale_hebbian = random.choice(loss_scale_hebbians)
        loss_scale_decorrelative = random.choice(loss_scale_decorrelatives)

        # construct settings
        settings = Settings.new()

        settings.device.device = DEVICE

        data_config = {
            "dataset": DATASET,
            "data_size": DATA_SIZE,
            "num_classes": NUM_CLASSES,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "test_batch_size": TEST_BATCH_SIZE,
            "iterations": iterations_}

        if settings.data_config is None:
            settings.data_config = DataConfig(**data_config)

        # mutate settings
        settings.data_config.iterations = iterations_
        settings.data_config.train_batch_size = train_batch_size

        settings.model.epochs = EPOCHS
        settings.model.loss_threshold = loss_threshold
        settings.model.hidden_sizes = hidden_sizes_
        settings.model.ff_activation = act
        settings.model.ff_optimizer = ff_opt
        settings.model.classifier_optimizer = classifier_opt
        settings.model.damping_factor = damping_factor

        settings.model.loss_scale_ff = loss_scale_ff
        settings.model.loss_scale_predictive = loss_scale_predictive
        settings.model.loss_scale_hebbian = loss_scale_hebbian
        settings.model.loss_scale_decorrelative = loss_scale_decorrelative

        if ff_opt == "rmsprop":
            settings.model.ff_rmsprop.momentum = ff_rmsprop_momentum
            settings.model.ff_rmsprop.learning_rate = ff_rmsprop_learning_rate
        elif ff_opt == "adam":
            settings.model.ff_adam.learning_rate = ff_adam_learning_rate
        elif ff_opt == "adadelta":
            settings.model.ff_adadelta.learning_rate = ff_adadelta_learning_rate

        if classifier_opt == "rmsprop":
            settings.model.classifier_rmsprop.momentum = classifier_rmsprop_momentum
            settings.model.classifier_rmsprop.learning_rate = classifier_rmsprop_learning_rate
        elif classifier_opt == "adam":
            settings.model.classifier_adam.learning_rate = classifier_adam_learning_rate
        elif classifier_opt == "adadelta":
            settings.model.classifier_adadelta.learning_rate = classifier_adadelta_learning_rate

        # run hyperparams
        p = Process(target=run, args=(
            settings,
        ))
        p.start()
        p.join()
        print("-----------", str(p.exitcode))
        if p.exitcode != 0:
            exit(1)
