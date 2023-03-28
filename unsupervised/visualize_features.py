import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import librosa
from dotmap import DotMap
# from main_modules.ES_extractor.ES_audio.audio_feat import AudioES
import numpy as np
from utils import get_args_parser
import argparse

def extract_emotions(config, row):
    config.video_duration = int(row['end_second'] - row['start_second'])
    AudioES_extractor = AudioES(config)
    
    # extract ES signals, all emotion category tracks, and all start-end offset tracks
    audio_emo_pred, audio_emo_feat = AudioES_extractor.extract_audio_features(row['audio_path'],True)
    return audio_emo_pred

def map_emotions(audio_emo_pred):
    # Define the emotions mapping
    emotions_mapping = {
        "emotion0": "fear",
        "emotion1": "anger",
        "emotion2": "joy",
        "emotion3": "sadness",
        "emotion4": "disgust",
        "emotion5": "surprise",
        "emotion6": "trust",
        "emotion7": "anticipation",
        "emotion8": "# none",
    }
    
    # Define the data types for the structured audio_emo_pred
    dtype = [('valence', float), ('arousal', float), ('emotion', 'U10')]

    # Create a new structured numpy audio_emo_pred with the specified data types
    audio_emo_pred_res = np.zeros((audio_emo_pred.shape[0],), dtype=dtype)

    # Iterate over each row in the audio_emo_pred
    for i, line in enumerate(audio_emo_pred):
        # Ignore the first two values in the row (i.e., "valence" and "arousal")
        values_to_check = line[:-2]
        
        # Get the indices of the three highest values in the row
        index = np.argmax(values_to_check)
        
        # Map the indices to the corresponding emotions using the emotions mapping
        emotion = emotions_mapping.get(f"emotion{index}")
        
        # Set the values in the new audio_emo_pred
        audio_emo_pred_res[i] = (line[1], line[0], emotion)
        
    print(audio_emo_pred_res)
    return audio_emo_pred_res

def visualize_audio_signals(config, row, fig, ax):
    audio_emo_pred = extract_emotions(config,row)
    vis_data = map_emotions(audio_emo_pred)
    
    # Find the range of arousal values in vis_data
    arousal_range = (min(label[1] for label in vis_data), max(label[1] for label in vis_data))

    # Set the y-axis limits to the arousal range
    ax.set_ylim(arousal_range)
    ax.set_yticks(np.arange(0, max(arousal_range)+0.1, 0.01))


    # Define the emotions and their colors
    emotions = ["anger", "anticipation", "disgust", "fear", "joy", "none", "sadness", "surprise", "trust"]
    colors = ['red', 'orange', 'brown', 'purple', 'blue', 'gray', 'green', 'pink', 'black']

    for i, label in enumerate(vis_data):
        if label[2] in emotions:
            color = colors[emotions.index(label[2])]
            ax.plot([i, i+1], [label[1], label[1]], c=color, marker='o')

    # Create the legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color=color, label=label, markersize=10)
                    for label, color in zip(emotions, colors)]
    ax.legend(handles=legend_elements, loc='upper left', ncols=3)
    
    return fig, ax

def visualize_audio_acoustics(row, fig, ax):
    audio_file = row['audio_path']

    # Load audio file
    clip, sr = librosa.load(audio_file, sr=None)

    # Define parameters
    n_frames = sr//10   # Number of frames per 100ms

    # Split audio into 15 second segments
    audio_segments = [clip[i*n_frames*150:(i+1)*n_frames*150] for i in range(int(len(clip)/(n_frames*150)))]

    # Compute tone features for each segment
    tone_features = []
    for segment in audio_segments:
        chroma = librosa.feature.chroma_stft(y=segment, sr=sr)
        # mfcc = librosa.feature.mfcc(y=segment, sr=sr)
        # spectral_contrast = librosa.feature.spectral_contrast(y=segment, sr=sr)
        # tonnetz = librosa.feature.tonnetz(y=segment, sr=sr)
        # Add spectral flux and zero-crossing rate features
        # spectral_flux = librosa.onset.onset_strength(y=segment, sr=sr)
        # zero_crossing_rate = librosa.feature.zero_crossing_rate(y=segment)
        # pitch = librosa.yin(segment, 50, 300, sr=sr) / 100
        # features = np.concatenate([chroma, tonnetz], axis=0)
        tone_features.append(chroma.T)

    # Convert list of tone feature arrays to a single numpy array
    tone_features = np.vstack(tone_features)

    # Average tone features across each segment
    tone_features = np.mean(tone_features, axis=0)

    # Create time array
    times = np.linspace(0, len(clip) / sr, num=tone_features.shape[0])

    # Plot tone features
    ax.plot(times, tone_features, label='Tone Features')
    ax.set_ylabel('Tone values')
    ax.legend()
    return fig, ax
    
def visualize_gt(row, fig, ax):
    segment_id = row['segment_id']
    end = row['end_second']
    start = row['start_second']
    duration = int(end - start)
    
    cp = row['CP_second']
    if cp != -1:
        ax.scatter(cp - start, 0, color='gray', marker="x")
        # Add text label to the first marker
        # ax.annotate('CP', xy=(cp - start, 0), xytext=(cp - start, 0.2),
        #             ha='center', va='bottom', fontsize=10,
        #             arrowprops=dict(facecolor='black', shrink=0.1))
    
    label = row['label']
    ax.set_title(segment_id+f' (label {label})')
    ax.set_xlim([0, duration])
    ax.set_xticks(range(0, duration+1, 1))
    ax.set_xlabel('Time (seconds)')
    
    return fig, ax

def visualize_audio_pitch_and_loudness(row, fig, ax):
    # Load audio file
    clip, sr = librosa.load(row['audio_path'], sr=None)

    # Compute pitch and loudness
    pitches, magnitudes = librosa.piptrack(y=clip, sr=sr)
    pitch = np.nanmean(pitches, axis=0)
    loudness = np.nanmean(librosa.power_to_db(magnitudes, ref=np.max), axis=0)

    # Compute time array
    frame_times = librosa.frames_to_time(np.arange(pitch.shape[0]), sr=sr, hop_length=512)
    time_diffs = np.diff(frame_times)
    times = np.cumsum(np.concatenate([[0], time_diffs]))

    # Plot pitch
    ax.plot(times, pitch, label='Pitch', color='blue')
    ax.set_ylabel('Pitch (Hz)')
    ax.legend()

    # ax.plot(times, loudness.T, label='Loudness', color='green')
    # ax.set_ylabel('Loudness (dB)')
    # ax.legend()
    
    return fig, ax

if __name__ == "__main__":
    ##### Defining arguments #####
    parser = argparse.ArgumentParser(
        "UCP detection inference on multi-modal data", parents=[get_args_parser()])
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    print(config_dict)
    config = DotMap(config_dict)
    data = pd.read_csv(config.dataset.input_path)
    
    # Create the output directory if it doesn't already exist
    output_dir = Path('output/pitch_features_visualization')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Iterate over the first 10 rows of data
    for i, row in data.head(50).iterrows():
        # Create a new figure and axis
        fig, ax = plt.subplots()
        fig, ax = visualize_gt(row, fig, ax)
        try:
            # fig, ax = visualize_audio_acoustics(row, fig, ax)
            fig, ax = visualize_audio_pitch_and_loudness(row, fig, ax)
        except:
            continue
        
        # fig, ax = visualize_audio_signals(config, row, fig, ax)
        # ax.set_ylabel('Arousal')
        # Save the figure to a PNG file in the output directory
        segment_id = row['segment_id']
        fig.savefig(output_dir / f'{segment_id}.png')
        
        # Close the figure to free up memory
        plt.close(fig)