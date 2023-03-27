import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import pdb

if __name__ == "__main__":
    # Load data from balanced_data.csv
    data = pd.read_csv('/home/dhgbao/Research_Monash/code/my_code/unsupervised_approach/multimodal_module/data/balanced_data.csv')

    # Define parameters
    n_frames = 4410   # Number of frames per 100ms

    # Initialize tone feature arrays
    cp_tone_features = []
    noncp_tone_features = []

    # Iterate over rows in data
    for index, row in tqdm(data.head(50).iterrows(), total=len(data)):
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
            # chroma = librosa.feature.chroma_stft(y=segment, sr=sr)
            # mfcc = librosa.feature.mfcc(y=segment, sr=sr)
            # spectral_contrast = librosa.feature.spectral_contrast(y=segment, sr=sr)
            tonnetz = librosa.feature.tonnetz(y=segment, sr=sr)
            # Add spectral flux and zero-crossing rate features
            # spectral_flux = librosa.onset.onset_strength(y=segment, sr=sr)
            # zero_crossing_rate = librosa.feature.zero_crossing_rate(y=segment)
            # pitch = librosa.yin(segment, 50, 300, sr=sr) / 100
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

    # Plot CP tone feature bar chart
    cp_mean = np.mean(cp_tone_features, axis=0)
    cp_median = np.median(cp_tone_features, axis=0)
    x = np.arange(len(cp_mean))
    width = 0.35
    axs[0].bar(x - width/2, cp_mean, width, label='Mean')
    axs[0].bar(x + width/2, cp_median, width, label='Median')
    axs[0].set_xticks(x)
    # axs[0].set_xticklabels(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])
    axs[0].set_xlabel('Pitch Class')
    axs[0].set_ylabel('Value')
    axs[0].set_title('Tone Features for CP')
    axs[0].legend()

    # Plot non-CP tone feature bar chart
    noncp_mean = np.mean(noncp_tone_features, axis=0)
    noncp_median = np.median(noncp_tone_features, axis=0)
    axs[1].bar(x - width/2, noncp_mean, width, label='Mean')
    axs[1].bar(x + width/2, noncp_median, width, label='Median')
    axs[1].set_xticks(x)
    # axs[0].set_xticklabels(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])
    axs[1].set_title('Non-CP Tone Features')
    axs[1].set_xlabel('Tone Feature')
    axs[1].set_ylabel('Value')
    axs[1].legend()
    plt.savefig('cp_noncp_tone_plot.png')
