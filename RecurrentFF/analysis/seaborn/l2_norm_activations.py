import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

SAVE_BASEPATH = "./img/presentation/"

# Load the parquet file
df = pd.read_parquet('converted_data.parquet')


def plot_activations_l2_over_time_from_df(df):
    # Filter dataframe for just the "full" activation data
    df_activations = df[df["activation_type"] == "full"].copy()

    # Compute the squared activity for each neuron
    df_activations['squared_activity'] = df_activations['activity']**2

    # Calculate L2 norm for each image-label pair
    df_activations['image'] = df_activations['image'].astype(str)
    df_activations['label'] = df_activations['label'].astype(str)
    df_activations['l2_norm'] = df_activations.groupby(['image', 'label', 'layer_index', 'is_correct'])[
        'squared_activity'].transform(lambda x: np.sqrt(x.sum()))

    # Now, average the L2 norms for every `image_timestep`, `layer_index`, and `is_correct` combination
    df_avg = df_activations.groupby(['image_timestep', 'layer_index', 'is_correct'])[
        'l2_norm'].mean().reset_index()

    # Plot settings
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("whitegrid")

    # Color palette
    palette = sns.color_palette("colorblind")

    # Split data into positive and negative
    df_positive = df_avg[df_avg['is_correct']]
    df_negative = df_avg[~df_avg['is_correct']]

    for label, data in [("Negative Data", df_negative), ("Positive Data", df_positive)]:
        g = sns.relplot(data=data, x='image_timestep', y='l2_norm', hue='layer_index',
                        kind='line', height=4, aspect=2, palette=palette, legend="full")

        # Enhancements
        g.set_axis_labels("Time", "L2 Norm")
        g.set_titles(label)
        g.fig.suptitle(f'L2 Norm of Activations Over Time ({label})')
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.savefig(
            f'{SAVE_BASEPATH}/l2norm_over_time_{label.replace(" ", "_").lower()}.pdf', format='pdf', bbox_inches='tight')
        plt.savefig(
            f'{SAVE_BASEPATH}/l2norm_over_time_{label.replace(" ", "_").lower()}.png', format='png', bbox_inches='tight')
        # plt.show()


plot_activations_l2_over_time_from_df(df)