import os
import pandas as pd
import argparse
import random
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
    audio_folder = args.audio_folder

    # Define the path and name of the new CSV file to be created
    new_csv_file = 'data/balanced_data.csv'    

    # Read in the all_segments.csv file
    all_segments_file = args.all_segments_file
    all_segments_df = pd.read_csv(all_segments_file)
    all_segments_df = all_segments_df[all_segments_df['type'] == 'video']

    # Create a new dataframe with the segment IDs, file paths, start and end timestamps
    segment_path_df = pd.DataFrame({
        'segment_id': all_segments_df['segment_id'],
        'file_path': all_segments_df['segment_id'].apply(lambda x: os.path.join(video_folder, x + '.mp4')),
        'audio_path': all_segments_df['segment_id'].apply(lambda x: os.path.join(audio_folder, x + '.mp3')),
        'start_second': all_segments_df['start'],
        'end_second': all_segments_df['end']
    })

    # Read in the changepoint_preprocessed.csv file
    changepoints_file = args.changepoints_file
    changepoint_df = pd.read_csv(changepoints_file)

    # Merge the segment dataframe with the changepoint dataframe to get the corresponding timestamp values
    segment_path_df = pd.merge(segment_path_df, changepoint_df[['segment_id', 'timestamp']], on='segment_id', how='left')

    # Set the CP_second column to -1 for segments without changepoints
    segment_path_df['CP_second'] = -1

    # Set the CP_second column to the corresponding timestamp value for segments with changepoints
    segment_path_df.loc[segment_path_df['timestamp'].notnull(), 'CP_second'] = segment_path_df.loc[segment_path_df['timestamp'].notnull(), 'timestamp']

    # Drop the timestamp column
    segment_path_df.drop('timestamp', axis=1, inplace=True)

    # Get the segments that appear in the changepoint data
    changepoint_segments = changepoint_df['segment_id'].tolist()

    # Get the number of segments with label=1
    num_pos_segments = len(changepoint_segments)

    # Get a random sample of segments with label=0
    neg_segments = segment_path_df[~segment_path_df['segment_id'].isin(changepoint_segments)]
    neg_sample = neg_segments.sample(n=num_pos_segments, random_state=42)
    # import pdb
    # pdb.set_trace()
    pos_sample = segment_path_df[segment_path_df['segment_id'].isin(changepoint_segments)]

    # Combine the positive and negative segments
    segment_path_df = pd.concat([neg_sample, pos_sample])

    # Shuffle the dataframe
    segment_path_df = segment_path_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Add a "label" column to the segment dataframe with a default value of 0
    segment_path_df['label'] = 0

    # Set label to 1 for segments that appear in the changepoint data
    segment_path_df.loc[segment_path_df['segment_id'].isin(changepoint_segments), 'label'] = 1

    # Write the segment IDs, file paths, start and end timestamps, and labels to the new CSV file
    segment_path_df.to_csv(new_csv_file, index=False)

    print("Finish generating annotation csv for UCP")
