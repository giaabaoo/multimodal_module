import argparse
import yaml
from pathlib import Path
import pandas as pd
from facenet_pytorch import MTCNN
from hsemotion.facial_emotions import HSEmotionRecognizer
from dotmap import DotMap
import os 
from main_modules.ES_extractor.ES_visual.visual_feat import VisualES
from main_modules.ES_extractor.ES_audio.audio_feat import AudioES

import argparse
from pathlib import Path
import os
import pdb
import matplotlib.pyplot as plt
import numpy as np

def get_args_parser():
    parser = argparse.ArgumentParser("Parsing arguments", add_help=False)
    parser.add_argument("--config", default="./configs/default.yaml", type=str)
    parser.add_argument("--batch_idx", default=-1, type=int)
    parser.add_argument("--test_path", default=None, type=int)

    return parser

def get_batch_df(config):
    folder_path = "/home/dhgbao/Research_Monash/code/my_code/unsupervised_approach/multimodal_module/" + config.pipeline.output_path
    output_path = os.path.join(folder_path, f"batch_{config.batch_idx}")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    csv_path = "/home/dhgbao/Research_Monash/code/my_code/unsupervised_approach/multimodal_module/" + config.dataset.input_path
    df = pd.read_csv(os.path.join(csv_path, f"batch_{config.batch_idx}.csv"))
    
    return df, output_path

def prepare_configs(args):
    # Load configs
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    print(config_dict)
    config = DotMap(config_dict)
    config.batch_idx = args.batch_idx
    
    if config.batch_idx != -1:
        config.batch_idx = config.batch_idx
        df, output_path = get_batch_df(config)
    else:
        df = pd.read_csv(config.dataset.input_path)
        output_path = config.pipeline.output_path
        
    config.df = df
    config.output_path = output_path
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    if config.network.use_visual_features:
        print("Using visual features...")
    if config.network.use_audio_features:
        print("Using audio features...")
        
    return config

def prepare_architectures(config):
    """Initialize Model"""  
    ##### Visual ES Extractor #####
    face_detector = MTCNN(keep_all=False, post_process=False, min_face_size=config.network.min_face_size, device=config.pipeline.device)
    emotion_recognizer = HSEmotionRecognizer(model_name=config.network.model_name, device=config.pipeline.device)
    
    ES_extractor = VisualES(config)
    ES_extractor.initialize_model(face_detector, emotion_recognizer)
    
    ##### Audio ES Extractor #####
    AudioES_extractor = AudioES(config)
    
    return ES_extractor, AudioES_extractor

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