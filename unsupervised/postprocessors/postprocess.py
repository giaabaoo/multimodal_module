import argparse
from pathlib import Path
import os
import glob
from utils import get_args_parser
import pdb

if __name__ == "__main__":
    ##### Defining arguments #####
    parser = argparse.ArgumentParser(
        "UCP detection inference on multi-modal data", parents=[get_args_parser()])
    args = parser.parse_args()
    
    Path("scoring_data").mkdir(parents=True, exist_ok=True)
    output_path = args.output_path
    json_files = glob.glob(os.path.join(output_path, '**/*.json'), recursive=True)
    
    video_dict = {}
    for json_file in json_files:
        segment_id = json_file.split("/")[-1].replace(".json", "")
        video_name = segment_id.split("_")[0]
        try:
            video_dict[video_name].append(json_file)
        except:
            video_dict[video_name] = [json_file]
        
    for video_name in video_dict.keys():
        cc = 0
        