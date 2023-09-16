import torch
import pandas as pd
import os

# Define constants
DATASET = "MNIST"
LABEL_SHOW_BOUNDARY = 10
NUMBER_FILES = 1
DIR_PATH = "artifacts/activations"


# Get the first 100 files from the directory
files = [f for f in os.listdir(DIR_PATH) if f.endswith('.pt')]
files = sorted(files)[:NUMBER_FILES]

# Initialize an empty dataframe
df = pd.DataFrame()

for i, file in enumerate(files):
    print("processing file", i, "of", len(files))

    # Load data
    data = torch.load(os.path.join(DIR_PATH, file))

    # Extract data from the loaded file
    for data_type, label in [("correct", 1), ("incorrect", 0)]:
        for activity_type in ["activations", "forward_activations", "backward_activations", "lateral_activations"]:
            activity_data = data[f"{data_type}_{activity_type}"]

            # Determine the activation type based on the activity_type
            if activity_type == "activations":
                activation_type_name = "full"
            else:
                activation_type_name = activity_type.replace(
                    "_activations", "")

            for timestep, timestep_data in enumerate(activity_data):
                is_label_showing = 1 if timestep >= LABEL_SHOW_BOUNDARY else 0

                print("processing timestep ", timestep)

                for layer_index, layer_data in enumerate(timestep_data):
                    for neuron_index, neuron_data in enumerate(layer_data):
                        row = {
                            "image_timestep": timestep,
                            "layer_index": layer_index,
                            "is_label_showing": is_label_showing,
                            "neuron index": neuron_index,
                            "activity": neuron_data.cpu().item(),  # Convert to scalar
                            "image": data['data'][timestep].cpu().numpy(),
                            # static single class means 0 index is safe
                            "label": data['labels'][0].cpu().numpy(),
                            "dataset": DATASET,
                            "is_correct": True if data_type == "correct" else False,
                            "activation_type": activation_type_name
                        }
                        df = pd.concat(
                            [df, pd.DataFrame([row])], ignore_index=True)

# Save dataframe to a csv file
df.to_parquet('converted_data.parquet')


print("Data conversion complete.")
