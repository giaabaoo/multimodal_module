import os
import pandas as pd
import argparse
from utils import get_args_parser
from pathlib import Path

if __name__ == '__main__':
    ##### Defining arguments #####
    parser = argparse.ArgumentParser(
        "UCP detection inference on multi-modal data", parents=[get_args_parser()])
    args = parser.parse_args()
    
    Path("data").mkdir(parents=True, exist_ok=True)
    
    # Define the folder path containing the segmented video files
    video_folder = args.video_folder

    # Define the path and name of the new CSV file to be created
    new_csv_file = 'data/full_data.csv'    

    # Read in the all_segments.csv file
    all_segments_file = args.all_segments_file
    all_segments_df = pd.read_csv(all_segments_file)
    all_segments_df = all_segments_df[all_segments_df['type'] == 'video']

    # Create a new dataframe with the segment IDs, file paths, start and end timestamps
    segment_path_df = pd.DataFrame({
        'segment_id': all_segments_df['segment_id'],
        'file_path': all_segments_df['segment_id'].apply(lambda x: os.path.join(video_folder, x + '.mp4')),
        'start_second': all_segments_df['start'],
        'end_second': all_segments_df['end']
    })

    # Read in the changepoint_preprocessed.csv file
    changepoints_file = args.changepoints_file
    changepoint_df = pd.read_csv(changepoints_file)

    # Add a "label" column to the segment dataframe with a default value of 0
    segment_path_df['label'] = 0

    # Set label to 1 for segments that appear in the changepoint data
    segment_path_df.loc[segment_path_df['segment_id'].isin(changepoint_df['segment_id']), 'label'] = 1

    # Write the segment IDs, file paths, start and end timestamps, and labels to the new CSV file
    segment_path_df.to_csv(new_csv_file, index=False)

    print("Finish generating annotation csv for UCP")
