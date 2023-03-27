import os
import pandas as pd
import argparse
from utils import get_args_parser
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "UCP detection inference on multi-modal data", parents=[get_args_parser()])
    args = parser.parse_args()
    # Define the folder path containing the segmented video files
    video_folder = args.video_folder
    audio_folder = args.audio_folder
    Path("data").mkdir(parents=True, exist_ok=True)
    
    # Define the path to the original changepoints_preprocessed.csv file
    changepoints_file = args.changepoints_file

    # Define the path and name of the new CSV file to be created
    new_csv_file = 'data/positive_data.csv'

    # Read in the original CSV file using pandas and filter out rows with type 'text' or 'audio'
    df = pd.read_csv(changepoints_file)
    df = df[df['type'] == 'video']

    # Read in the all_segments.csv file
    all_segments_file = args.all_segments_file
    all_segments_df = pd.read_csv(all_segments_file)

    # Merge the changepoints dataframe with the all_segments dataframe on the 'file_id' and 'segment_id' columns
    merged_df = pd.merge(df, all_segments_df, on=['file_id', 'segment_id'])

    # Calculate the 'start_second' and 'end_second' columns using the 'start' and 'end' columns in the all_segments dataframe
    merged_df['start_second'] = merged_df['start']
    merged_df['end_second'] = merged_df['end']

    # Create a new dataframe with the segment IDs, file paths, start and end timestamps
    segment_path_df = pd.DataFrame({
        'segment_id': merged_df['segment_id'],
        'file_path': merged_df['segment_id'].apply(lambda x: os.path.join(video_folder, x + '.mp4')),
        'audio_path': merged_df['segment_id'].apply(lambda x: os.path.join(audio_folder, x + '.mp3')),
        'start_second': merged_df['start_second'],
        'end_second': merged_df['end_second'],
        'CP_second' : merged_df['timestamp']
    })
    segment_path_df['label'] = 1

    # Write the segment IDs, file paths, start and end timestamps to the new CSV file
    segment_path_df.to_csv(new_csv_file, index=False)

    print("Finish generating positive annotation csv for UCP")
