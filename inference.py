import os 
import argparse
import sys
sys.path.append("unsupervised")
from inference_functions import run_pipeline_single_video
from utils import prepare_configs, prepare_architectures, get_args_parser
from tools.debug.debug import *
from tqdm import tqdm
from moviepy.editor import *
import multiprocessing

def process_video(config, row, ES_extractor, AudioES_extractor):
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
        return
    
    run_pipeline_single_video(config, ES_extractor, AudioES_extractor)  

def inference(config, ES_extractor, AudioES_extractor):
    processes = []
    for index, row in tqdm(config.df.iterrows(), mininterval=0.1):
        multiprocessing.set_start_method('forkserver', force=True)
        p = multiprocessing.Process(target=process_video, args=(config, row, ES_extractor, AudioES_extractor))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()

# def inference(config, ES_extractor, AudioES_extractor):
#     for index, row in tqdm(config.df.iterrows(), mininterval=0.1):
#         video_path = row['file_path']
#         config.start_second = row['start_second']
#         config.end_second = row['end_second']
#         config.video_name = row['segment_id']
#         config.single_video_path = video_path
#         config.single_audio_path = row['audio_path']
#         config.single_output_path = os.path.join(config.output_path, video_path.split("/")[-1].replace(".mp4",".json"))
#         print(f"Video: {config.video_name}")
#         print(f'Video path: {config.single_video_path} \n JSON path: {config.single_output_path} \n')
    
#         if os.path.isfile(config.single_output_path) is True:
#             print('...Result file exists, skipping...')
#             continue
        
#         run_pipeline_single_video(config, ES_extractor, AudioES_extractor)  
        
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