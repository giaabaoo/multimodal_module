import pandas as pd
import numpy as np
import os
import json
from tqdm import tqdm

if __name__ == '__main__':
    # Read the changepoints_preprocessed.csv file
    df_gt = pd.read_csv('/home/dhgbao/Research_Monash/code/my_code/unsupervised/multimodal_module/data/changepoints_preprocessed.csv')

    # Set the threshold for a true positive to 10 seconds
    threshold = 10.0

    # Initialize variables for overall true positives, false positives, and segments with changepoints
    overall_tp = 0
    overall_fp = 0
    segments_with_changepoints = 0

    # Set the path to the folder containing the predicted changepoints for each segment_id
    pred_folder_path = 'output'

    # Iterate through each file in the predicted changepoints folder
    for file_name in tqdm(os.listdir(pred_folder_path)):
        if file_name.endswith('.json'):
            # Read the predicted changepoints from the final_cp_result field
            with open(os.path.join(pred_folder_path, file_name), 'r') as f:
                pred_data = json.load(f)
                pred_cp = np.array(pred_data['final_cp_result'])
            
            # Get the segment_id from the file name
            segment_id = os.path.splitext(file_name)[0]
            
            # Check if any changepoints were predicted for this segment
            if len(pred_cp) == 0 or len(pred_cp[0]) == 0:
                continue
            
            # Increment the count of segments with changepoints
            segments_with_changepoints += 1
            
            # Loop through each predicted changepoint and compare it with the ground-truth changepoints for this segment
            for pred in pred_cp[0]:
                is_tp = False
                for _, row in df_gt.iterrows():
                    if row['type'] == 'video' and row['segment_id'] == segment_id and abs(pred - row['timestamp']) <= threshold:
                        is_tp = True
                        break
                if is_tp:
                    overall_tp += 1
                else:
                    overall_fp += 1

    # Calculate accuracy for all segments with changepoints
    accuracy = overall_tp / (overall_tp + overall_fp)

    # Print the evaluation metric for all segments with changepoints
    print('Evaluation metric for all segments with changepoints:')
    print(f'Accuracy: {accuracy:.2f}')
