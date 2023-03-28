import librosa
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    audio_file = '/home/dhgbao/Research_Monash/dataset/ccu-data/CCU_COMBINED_ANNOTATED_VIDEO/segmented_videos_audio/M01000AJ7_0010.mp3'

    # Load audio file
    clip, sr = librosa.load(audio_file, sr=None)

    # Compute pitch and loudness
    pitches, magnitudes = librosa.piptrack(y=clip, sr=sr)
    pitch = np.nanmean(pitches, axis=0)
    loudness = np.nanmean(librosa.power_to_db(magnitudes, ref=np.max), axis=0)
    energy = np.sum(magnitudes, axis=0)

    # Compute time array
    frame_times = librosa.frames_to_time(np.arange(pitch.shape[0]), sr=sr, hop_length=512)
    time_diffs = np.diff(frame_times)
    times = np.cumsum(np.concatenate([[0], time_diffs]))

    # Plot pitch
    plt.figure(figsize=(20, 5))
    plt.plot(times, pitch, label='Pitch', color='blue')
    plt.title('Pitch over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Pitch (Hz)')
    plt.ylim([0, 800])
    plt.legend()
    plt.savefig("output/pitch_plot.png")

    # Plot loudness
    plt.figure(figsize=(20, 5))
    plt.plot(times, loudness.T, label='Loudness', color='orange')
    plt.title('Loudness over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Loudness (dB)')
    plt.ylim([-100, 0])
    plt.legend()
    plt.savefig("output/loudness_plot.png")
    
    # Plot energy
    plt.figure(figsize=(20, 5))
    plt.plot(times, energy, label='Energy', color='green')
    plt.title('Energy over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy')
    plt.legend()
    plt.savefig("output/energy_plot.png")
