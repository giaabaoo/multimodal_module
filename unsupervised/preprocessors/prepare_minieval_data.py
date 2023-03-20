import os
import pandas as pd
import argparse
from utils import get_args_parser
from pathlib import Path
import pdb

if __name__ == "__main__":
    ##### Defining arguments #####
    parser = argparse.ArgumentParser(
        "UCP detection inference on multi-modal data", parents=[get_args_parser()])
    args = parser.parse_args()
    
    Path("data").mkdir(parents=True, exist_ok=True)
    
    # Read in the segments.tab and changepoints_preprocessed.csv files
    segments = pd.read_csv(args.segments_tab_file, sep='\t')
    
    changepoints_file = args.changepoints_file
    changepoints = pd.read_csv(changepoints_file)

    # Create a dictionary mapping segment IDs to labels
    labels = {seg_id: 1 for seg_id in changepoints['segment_id'].unique()}

    # Define the file path for the videos
    video_dir = args.video_folder
    audio_dir = args.audio_folder

    # Initialize an empty list to store the data for each segment
    segment_data = []

    # Loop through the rows of the segments file
    for i, row in segments.iterrows():
        # Extract the relevant information from the row
        segment_id = row['segment_id']
        start = row['start']
        end = row['end']
        
        # Determine if the segment is a changepoint (i.e., has a label)
        if segment_id in labels:
            label = 1
        else:
            label = 0
        
        # Construct the file path for the video
        video_name = row['segment_id'] + ".mp4"
        audio_name = row['segment_id'] + ".mp3"
        file_path = os.path.join(video_dir, video_name)
        audio_path = os.path.join(audio_dir, audio_name)
        
        # Add the segment data to the list
        segment_data.append([segment_id, file_path, audio_path, start, end, label])

    # Create a DataFrame from the segment data
    df = pd.DataFrame(segment_data, columns=['segment_id', 'file_path', 'audio_path', 'start_second', 'end_second', 'label'])
    
    # Extract the segment IDs present in all_segments_df
    valid_segment_ids = [i.replace(".mp4","") for i in os.listdir(video_dir)]

    # Filter df to keep only rows with segment IDs present in valid_segment_ids
    df = df[df['segment_id'].isin(valid_segment_ids)]
    
    # Write the DataFrame to a new CSV file
    df.to_csv('data/minieval_data.csv', index=False)