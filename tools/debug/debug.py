import pandas as pd

def get_video_info(config):
    video_name = config.video_name
    df = pd.read_csv("data/changepoints_preprocessed.csv")
    
    # Use the loc function to filter the dataframe based on the segment ID
    result = df.loc[df['segment_id'] == video_name]
    pd.set_option('display.max_rows', None)

    print(result)