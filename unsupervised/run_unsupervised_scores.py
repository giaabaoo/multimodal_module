import os 
import argparse
import sys
sys.path.append("..")

from utils import prepare_architectures, prepare_configs
from inference_functions import get_unsupervised_scores
from tqdm import tqdm
from moviepy.editor import *
from utils import get_args_parser
import pdb

def inference(config, ES_extractor, AudioES_extractor):
    for index, row in tqdm(config.df.iterrows(), mininterval=0.1):
        video_path = row['file_path']
        config.start_second = row['start_second']
        config.end_second = row['end_second']
        config.video_name = row['segment_id']
        config.single_video_path = video_path
        config.single_audio_path = row['audio_path']
        config.single_output_path = os.path.join(config.output_path, video_path.split("/")[-1].replace(".mp4",".json"))
        print(f"Video: {config.video_name}")
        print(f'Video path: {config.single_video_path} \n JSON path: {config.single_output_path} \n')
    
        if os.path.isfile(config.single_output_path) is True:
            print('...Result file exists, skipping...')
            continue
        
        # Timestamp-level scores with shape (number of detected tracks, number of timestamps)
        score_cp_matrix_ts, binary_cp_matrix_ts = get_unsupervised_scores(config, ES_extractor, AudioES_extractor)
        pdb.set_trace()  
        
if __name__ == "__main__":
    ##### Defining arguments #####
    parser = argparse.ArgumentParser(
        "UCP detection inference on multi-modal data", parents=[get_args_parser()])
    args = parser.parse_args()
    
    ##### Preparing configs and architectures #####
    config = prepare_configs(args)
    ES_extractor, AudioES_extractor = prepare_architectures(config)
    
    ##### Perform inference on the video segments #####
    inference(config, ES_extractor, AudioES_extractor)