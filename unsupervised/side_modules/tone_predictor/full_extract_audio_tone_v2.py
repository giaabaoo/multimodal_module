import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")

if __name__ == "__main__":
    # Load data from balanced_data.csv
    data = pd.read_csv('/home/dhgbao/Research_Monash/code/my_code/unsupervised_approach/multimodal_module/data/balanced_data.csv')

    # Initialize loudness feature arrays and class labels
    cp_loudness_means = []
    cp_loudness_maxes = []
    cp_loudness_mins = []
    noncp_loudness_means = []
    noncp_loudness_maxes = []
    noncp_loudness_mins = []

    # Iterate over rows in data
    for index, row in tqdm(data.head(100).iterrows(), total=len(data)):
        audio_file = row['audio_path']

        try:
            # Load audio file
            clip, sr = librosa.load(audio_file, sr=None)
        except:
            continue

        # Compute loudness
        # loudness = np.nanmean(librosa.power_to_db(np.abs(librosa.stft(clip)), ref=np.max), axis=0)
        pitches, magnitudes = librosa.piptrack(y=clip, sr=sr)
        pitch = np.nanmean(pitches, axis=0)
        

        if row['label'] == 1:
            cp_loudness_means.append(np.mean(pitch))
            cp_loudness_maxes.append(np.max(pitch))
            cp_loudness_mins.append(np.min(pitch))
        else:
            noncp_loudness_means.append(np.mean(pitch))
            noncp_loudness_maxes.append(np.max(pitch))
            noncp_loudness_mins.append(np.min(pitch))

    # Create boxplots for CP loudness features
    fig, axs1 = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))
    axs1[0].boxplot(cp_loudness_means)
    axs1[0].set_title('Mean')
    axs1[1].boxplot(cp_loudness_maxes)
    axs1[1].set_title('Max')
    axs1[2].boxplot(cp_loudness_mins)
    axs1[2].set_title('Min')
    fig.suptitle('CP Pitch Features')

    # Set axis labels
    for ax in axs1.flat:
        ax.set(xlabel='Pitch')

    # Create boxplots for non-CP loudness features
    fig, axs2 = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))
    axs2[0].boxplot(noncp_loudness_means)
    axs2[0].set_title('Mean')
    axs2[1].boxplot(noncp_loudness_maxes)
    axs2[1].set_title('Max')
    axs2[2].boxplot(noncp_loudness_mins)
    axs2[2].set_title('Min')
    fig.suptitle('Non-CP Pitch Features')

    # Set axis labels
    for ax in axs2.flat:
        ax.set(xlabel='Pitch')

    # Save figures
    plt.savefig("output/cp_pitch_plot.png")
    plt.savefig("output/noncp_pitch_plot.png")
