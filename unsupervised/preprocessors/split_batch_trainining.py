import pandas as pd
import math
from pathlib import Path

if __name__ == "__main__":
    Path("data/batches").mkdir(parents=True, exist_ok=True)
    # Load the segments into a Pandas DataFrame
    try:
        segments = pd.read_csv('data/full_data.csv')
    except:
        print("full_data.csv might not exist. Generate it using prepare_full_data_UCP.py first.")

    # Define the number of segments per batch
    segments_per_batch = 1000

    # Calculate the total number of batches
    total_batches = math.ceil(len(segments) / segments_per_batch)

    # Split the segments into batches
    batches = [segments[i:i+segments_per_batch] for i in range(0, len(segments), segments_per_batch)]

    # Save each batch to a separate CSV file
    for i, batch in enumerate(batches):
        batch.to_csv(f'data/batches/batch_{i+1}.csv', index=False)