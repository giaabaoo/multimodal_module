import argparse
from pathlib import Path
import os
import glob
from utils import get_args_parser
import pandas as pd
import json
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
    
    column_names = ['file_id', 'timestamp', 'llr']
    for video_name in video_dict.keys():
        df = pd.DataFrame(columns=column_names)
        videos_list = video_dict[video_name].values()
        for video in videos_list:
            segment_id = video.split("/")[-1].replace(".json", "")
            with open(video,"r") as f:
                data = json.load(f)
            
            try:
                llr = data['final_cp_llr'][0][0]
            except:
                llr = 0 # no CP predicted
            
            predicted_ts = data['final_cp_result'][0][0]
            
            df.loc[0] = [segment_id, predicted_ts, llr]
        df.to_csv(f'scoring_data/{video_name}.tab', sep='\t', index=False)
        
    output_index_column_names = ['file_id', 'is_processed', 'message file_path'] 			
    output_index_df = pd.DataFrame(columns=output_index_column_names)
    