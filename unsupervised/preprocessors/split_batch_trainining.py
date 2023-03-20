import pandas as pd
import math
from pathlib import Path
import argparse
from utils import get_args_parser
import pdb

if __name__ == "__main__":
    ##### Defining arguments #####
    parser = argparse.ArgumentParser(
        "UCP detection inference on multi-modal data", parents=[get_args_parser()])
    args = parser.parse_args()
    
    Path(args.batches_data_csv).mkdir(parents=True, exist_ok=True)
    # Load the segments into a Pandas DataFrame
    try:
        segments = pd.read_csv(args.full_data_csv)
    except:
        print("mini_eval_data.csv might not exist. Generate it using prepare_full_data_mini_eval.py first.")

    # Define the number of segments per batch
    segments_per_batch = 1000

    # Calculate the total number of batches
    total_batches = math.ceil(len(segments) / segments_per_batch)

    # Split the segments into batches
    batches = [segments[i:i+segments_per_batch] for i in range(0, len(segments), segments_per_batch)]

    # Save each batch to a separate CSV file
    for i, batch in enumerate(batches):
        batch.to_csv(f'{args.batches_data_csv}/batch_{i+1}.csv', index=False)