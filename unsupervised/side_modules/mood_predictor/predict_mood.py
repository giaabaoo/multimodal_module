import os
import numpy as np
from pyannote.audio.features import Pretrained
from pyannote.audio.labeling import SequenceLabeling
from pyannote.audio.signal import Binarize
from pyannote.audio.signal import Peak

# Load the audio file
audio_path = "/home/dhgbao/Research_Monash/dataset/ccu-data/CCU_COMBINED_ANNOTATED_VIDEO/segmented_videos_audio/M01000AJ7_0001.mp3"
pretrained = Pretrained(validate_dir=None)
pipeline = SequenceLabeling(model=pretrained, binarizer=Binarize(offset=0.5, onset=0.5))
audio = pipeline({'audio': audio_path})

# Diarize the audio file
diarization = audio['pyannote.labels']

# Extract mood features for each diarized track
mood_features = []
for segment, label in diarization.itertracks(label=True):
    if label == "speaker":
        # Skip speaker segments
        continue
    segment_audio = audio.crop(segment, mode='center', fixed=5.0)
    features = pretrained(segment_audio)
    features = features.data.mean(axis=0)
    mood_features.append(features)

# Convert mood features to output format
output = []
for segment, label in diarization.itertracks(label=True):
    if label == "speaker":
        # Skip speaker segments
        continue
    segment_start = segment.start * audio.sampling_rate
    segment_end = segment.end * audio.sampling_rate
    segment_duration = int(segment_end - segment_start)
    segment_mood = mood_features.pop(0)
    segment_output = np.repeat(segment_mood.reshape(1, -1), segment_duration, axis=0)
    output.append(segment_output)
output = np.concatenate(output, axis=0)

# Draw a graph for the final output
import matplotlib.pyplot as plt
plt.imshow(output.T, aspect='auto', cmap='jet')
plt.xlabel('Time (seconds)')
plt.ylabel('Mood')
plt.show()
