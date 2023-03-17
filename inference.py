import os 
import argparse
# import sys
# sys.path.append("unsupervised")
from unsupervised.inference_functions import run_pipeline_single_video
from unsupervised.utils import prepare_configs, prepare_architectures, get_args_parser
from tools.debug.debug import *
from tqdm import tqdm
from moviepy.editor import *

def inference(df, output_path, config, ES_extractor):
    for index, row in tqdm(df.iterrows(), mininterval=0.1):
        video_path = row['file_path']
        config.start_second = row['start_second']
        config.end_second = row['end_second']
        config.video_name = row['segment_id']
        config.single_video_path = video_path
        config.single_output_path = os.path.join(output_path, video_path.split("/")[-1].replace(".mp4",".json"))
        print(f"Video: {config.video_name}")
        print(f'Video path: {config.single_video_path} \n JSON path: {config.single_output_path} \n')
    
        if os.path.isfile(config.single_output_path) is True:
            print('...Result file exists, skipping...')
            continue
        
        run_pipeline_single_video(config, ES_extractor)  
        
if __name__ == "__main__":
    ##### Defining arguments #####
    parser = argparse.ArgumentParser(
        "UCP detection inference on multi-modal data", parents=[get_args_parser()])
    args = parser.parse_args()
    
    ##### Preparing configs and architectures #####
    config = prepare_configs(args)
    ES_extractor, df, output_path = prepare_architectures(config)
    
    ##### Perform inference on the video segments #####
    inference(df, output_path, config, ES_extractor)