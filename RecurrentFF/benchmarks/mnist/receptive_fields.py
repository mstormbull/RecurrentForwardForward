import torch
import os
import random
import torchvision
import cv2
import matplotlib.pyplot as plt

ACTIVATIONS_DIR = "./artifacts/activations/"
NUM_LAYERS = 3
NUM_NEURONS_PER_LAYER = 2000
NUM_NEURONS_TO_ANALYZE = 100
FILE_LIMIT = 1000


def enhance_and_visualize(image_path):
    # Load the receptive field
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Convert to grayscale for simplicity; you can skip this if you want to keep color
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Enhance contrast using histogram equalization
    equ = cv2.equalizeHist(gray)

    # Convert to heatmap
    heatmap = cv2.applyColorMap(equ, cv2.COLORMAP_JET)

    # Save the image
    cv2.imwrite(f"enhanced_{image_path}", heatmap)
    # cv2.imshow('Enhanced Receptive Field', heatmap)


random_neurons = [(random.randint(0, NUM_LAYERS - 1), random.randint(0, NUM_NEURONS_PER_LAYER - 1))
                  for _ in range(NUM_NEURONS_TO_ANALYZE)]

# Dictionary to store activations
neuron_to_activations = {(layer, neuron): []
                         for layer, neuron in random_neurons}

files = os.listdir(ACTIVATIONS_DIR)

# Load activations and gather data
file_count = 0
for filename in files:

    print(f"Processing file {file_count}")

    print(filename)

    if file_count >= FILE_LIMIT:
        break

    filepath = os.path.join(ACTIVATIONS_DIR, filename)
    data = torch.load(filepath)
    activations = data["correct_activations"].abs()

    for layer, neuron in random_neurons:
        for timestep in range(activations.shape[0]):
            neuron_to_activations[(layer, neuron)].append(
                activations[timestep, layer, neuron].item())

    file_count += 1

# print(neuron_to_activations)
# input()

# Calculate mean and standard deviation for each neuron's activations
neuron_to_mean_std_dev = {(layer, neuron): (torch.mean(torch.tensor(neuron_to_activations[(layer, neuron)])),
                                            torch.std(torch.tensor(neuron_to_activations[(layer, neuron)])))
                          for layer, neuron in random_neurons}

# print(neuron_to_mean_std_dev)
# input()

# Identify images where activation exceeds threshold and average them
neuron_to_images = {(layer, neuron): [] for layer, neuron in random_neurons}
file_count = 0
for filename in files:
    print(filename)

    print(f"Processing file {file_count}")

    if file_count >= FILE_LIMIT:
        break

    filepath = os.path.join(ACTIVATIONS_DIR, filename)
    data = torch.load(filepath)
    activations = data["correct_activations"].abs()

    # CIFAR10
    # image = data["data"].view(3, 32, 32)

    # MNIST
    image = data["data"].view(1, 28, 28)

    for layer, neuron in random_neurons:
        mean, std_dev = neuron_to_mean_std_dev[(layer, neuron)]
        threshold = mean + std_dev * 3
        print(f"threshold: {threshold}")
        threshold = threshold
        for timestep in range(activations.shape[0]):
            if activations[timestep, layer, neuron] > threshold:
                print(
                    f"activations[timestep, layer, neuron]: {activations[timestep, layer, neuron]}")
                neuron_to_images[(layer, neuron)].append(image)
                break  # Once an image is added for a neuron, no need to check other timesteps for the same image

    file_count += 1

# Average images and save
for layer, neuron in random_neurons:

    images = neuron_to_images[(layer, neuron)]
    if len(images) > 0:
        print(f"Saving images for neuron {layer}-{neuron}")

        print(len(images))
        avg_image = torch.stack(images).mean(dim=0)
        # Convert image to be in the [0,1] range for saving
        avg_image = (avg_image - avg_image.min()) / \
            (avg_image.max() - avg_image.min())
        filename = f"{layer}-{neuron}.png"

        avg_image = avg_image.squeeze()  # Remove the singleton channel dimension

        plt.imshow(avg_image.cpu().numpy(), cmap='gray')
        filename = f"{layer}-{neuron}.png"
        plt.imsave(filename, avg_image.cpu().numpy(), cmap='gray')

        # avg_image = avg_image.repeat(3, 1, 1)
        # torchvision.utils.save_image(avg_image, filename)
