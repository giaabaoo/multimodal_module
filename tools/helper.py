import argparse
from pathlib import Path
import os
import pandas as pd
import pdb
import matplotlib.pyplot as plt
import numpy as np

def get_args_parser():
    parser = argparse.ArgumentParser("Parsing arguments", add_help=False)
    parser.add_argument("--config", default="./configs/default.yaml", type=str)
    parser.add_argument("--batch_idx", default=-1, type=int)

    return parser

def get_batch_df(config):
    if config.network.use_audio_features:
        folder_path = "/home/dhgbao/Research_Monash/code/my_code/unsupervised/multimodal_module/output/batches_audio"
    else:
        folder_path = "/home/dhgbao/Research_Monash/code/my_code/unsupervised/multimodal_module/output/batches"
    output_path = os.path.join(folder_path, f"batch_{config.batch_idx}")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    csv_path = "/home/dhgbao/Research_Monash/code/my_code/unsupervised/multimodal_module/data/batches"
    df = pd.read_csv(os.path.join(csv_path, f"batch_{config.batch_idx}.csv"))
    
    return df, output_path

def draw_result_graph(config, score_cp_matrix, score_cp_matrix_ts):
    # assign a color to each track for plotting
    # Set up color map for tracks
    num_tracks = len(score_cp_matrix)
    color_map = plt.get_cmap('Set1')
    colors = [color_map(i) for i in np.linspace(0, 1, num_tracks)]
    
    folder_path = f"{os.path.dirname(config.single_output_path)}/figures"
    Path(folder_path).mkdir(exist_ok=True, parents=True)

    # Plot frame-level predictions
    plt.figure(figsize=(16, 8))
    for i in range(num_tracks):
        plt.plot(score_cp_matrix[i], color=colors[i], label='Track {}'.format(i))
    plt.title('Frame-level predictions')
    plt.xlabel('Frame')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig(f'{folder_path}/frame_level_predictions.png')
    plt.show()

    # Plot timestamp-level predictions
    plt.figure(figsize=(16, 8))
    for i in range(num_tracks):
        plt.plot(score_cp_matrix_ts[i], color=colors[i], label='Track {}'.format(i))
    plt.title('Timestamp-level predictions')
    plt.xlabel('Second')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig(f'{folder_path}/timestamp_level_predictions.png')
    plt.show()