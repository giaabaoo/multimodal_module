import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    # Load data from balanced_data.csv
    data = pd.read_csv('/home/dhgbao/Research_Monash/code/my_code/unsupervised_approach/multimodal_module/data/balanced_data.csv')

    # Define parameters
    n_frames = 4410   # Number of frames per 100ms

    # Initialize tone feature arrays
    cp_tone_features = []
    noncp_tone_features = []

    # Iterate over rows in data
    for index, row in tqdm(data.head(10).iterrows(), total=len(data)):
        audio_file = row['audio_path']

        try:
            # Load audio file
            clip, sr = librosa.load(audio_file, sr=None)
        except:
            continue

        # Define parameters
        n_frames = sr//10   # Number of frames per 100ms

        # Split audio into 15 second segments
        audio_segments = [clip[i*n_frames*150:(i+1)*n_frames*150] for i in range(int(len(clip)/(n_frames*150)))]

        # Compute tone features for each segment
        tone_features = []
        for segment in audio_segments:
            tonnetz = librosa.feature.tonnetz(y=segment, sr=sr)
            tone_features.append(tonnetz.T)

        # Convert list of tone feature arrays to a single numpy array
        try:
            tone_features = np.vstack(tone_features)
        except:
            continue

        # Average tone features across each segment
        tone_features = np.mean(tone_features, axis=0)

        # Determine whether to append to cp_tone_features or noncp_tone_features
        if row['label'] == 1:
            cp_tone_features.append(tone_features)
        else:
            noncp_tone_features.append(tone_features)

    # Convert tone feature lists to numpy arrays
    try:
        cp_tone_features = np.vstack(cp_tone_features)
        noncp_tone_features = np.vstack(noncp_tone_features)
    except:
        pass

    # Create figure with 2 subplots
    fig, axs = plt.subplots(2, figsize=(10, 8))

    # Adjust vertical spacing between subplots
    fig.subplots_adjust(hspace=0.5)

    # Plot CP tone feature bar chart
    x = np.arange(cp_tone_features.shape[1])
    width = 0.35
    axs[0].bar(x, np.mean(cp_tone_features, axis=0), width, label='CP')
    axs[0].set_xticks(x)
    axs[0].set_xlabel('Pitch Class')
    axs[0].set_ylabel('Value')
    axs[0].set_title('Tone Features for CP')
    axs[0].legend()

    # Plot non-CP tone feature bar chart
    x = np.arange(noncp_tone_features.shape[1])
    width = 0.35
    axs[1].bar(x, np.mean(noncp_tone_features, axis=0), width, label='Non-CP')
    axs[1].set_xticks(x)
    axs[1].set_xlabel('Pitch Class')
    axs[1].set_ylabel('Value')
    axs[1].set_title('Tone Features for Non-CP')
    axs[1].legend()

    plt.savefig("output/balanced_data_tone_plot.png")
