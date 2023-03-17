# Extracting audio features representing for emotional signals of individual in conversational video
import librosa
import pydub.silence as silence
import numpy as np
import pdb
import librosa
from pydub import AudioSegment

def get_audio_features(video):
    # Set parameters for audio extraction and diarization
    min_silence_len = 500
    silence_thresh = -40
    n_mfcc = 13

    # Extract audio from the video
    audio = video.audio.to_soundarray()

    audio_segment = AudioSegment(np.array(audio).tobytes(),frame_rate=44100,sample_width=2,channels=2)

    # Perform diarization to separate out different speakers
    speaker_segments = silence.detect_nonsilent(audio_segment, min_silence_len=min_silence_len, silence_thresh=silence_thresh)

    # Initialize empty lists for the speaker segments, their feature vectors, and their start and end frames
    segments = []
    feature_vectors = []
    start_end_frames = []

    # Loop through each speaker segment and extract features and frame information
    for i, segment in enumerate(speaker_segments):
        
        # Extract audio and sample rate for this segment
        segment_audio = audio_segment[segment[0]:segment[1]]
        sr = video.audio.fps
        
        # Extract features (e.g., MFCCs) for this segment
        segment_audio = np.array(segment_audio.ravel(), dtype=np.float32)
        features = librosa.feature.mfcc(y=segment_audio, sr=sr, n_mfcc=n_mfcc)

        
        # Append the features and frame information to the output lists
        segments.append(segment_audio)
        feature_vectors.append(features.T)
        start_end_frames.append([segment[0], segment[1]])

    # Print the results
    return feature_vectors, start_end_frames
