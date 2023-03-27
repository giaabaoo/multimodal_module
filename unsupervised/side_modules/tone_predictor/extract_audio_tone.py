import librosa
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    audio_file = '/home/dhgbao/Research_Monash/dataset/ccu-data/CCU_COMBINED_ANNOTATED_VIDEO/segmented_videos_audio/M01000AJ7_0010.mp3'

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
        mfcc = librosa.feature.mfcc(y=segment, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=segment, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=segment, sr=sr)
        features = np.concatenate([chroma, mfcc, spectral_contrast, tonnetz], axis=0)
        tone_features.append(features.T)

    # Convert list of tone feature arrays to a single numpy array
    tone_features = np.vstack(tone_features)

    # Average tone features across each segment
    tone_features = np.mean(tone_features, axis=0)

    # Create time array
    times = np.linspace(0, len(clip) / sr, num=tone_features.shape[0])

    # Plot tone features
    plt.figure(figsize=(20, 10))
    plt.plot(times, tone_features, label='Tone Features')
    plt.title('Tone Features over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig("output/tone_plot.png")
