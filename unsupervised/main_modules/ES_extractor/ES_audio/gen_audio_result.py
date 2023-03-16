import pathlib
from collections import OrderedDict

import librosa
import numpy as np
import pandas as pd
import torch
from pydub import AudioSegment
from torch.autograd import Variable
from tqdm import tqdm

from ccu_mini_eval.model.audio_head import get_audio_model, get_trill_model
from ccu_mini_eval.util.consts import DEVICE, SEGMENT_STRIDE, SEGMENT_DURATION, REQUIRED_SAMPLE_RATE
from ccu_mini_eval.util.label_space_mapping import bold_to_main, bold_to_main_valence, bold_to_main_arousal


def get_emotion_features_from_audio(audio_model, audio_segment, original_sample_rate):
    # audio_model = get_audio_model() # this seems inefficient as it has to load again, everytime the function is called
    # features = extract_trill_features(audio, original_sample_rate)

    channel_sounds = audio_segment.split_to_mono()
    _emb = []
    _lay = []
    for s in channel_sounds:
        audio_array = np.array(s.get_array_of_samples())
        embedding, layer19 = extract_trill_features(audio_array, original_sample_rate)
        _emb.append(embedding)
        _lay.append(layer19)
    embedding = torch.stack(_emb)
    layer19 = torch.stack(_lay)

    with torch.no_grad():
        features = Variable(torch.unsqueeze(embedding.mean(axis=0), 0)).to(DEVICE)
        output_dis, output_con, output_feat = audio_model(features.float())

        output_emo = output_dis.cpu().detach().numpy()
        output_con = output_con.cpu().detach().numpy()
        output_valence = output_con[:, 0]
        output_arousal = output_con[:, 1]
        pen_features = output_feat.cpu().detach().numpy()

        return output_valence, output_arousal, output_emo, pen_features, embedding, layer19


def extract_trill_features(audio, original_sample_rate):
    module = get_trill_model()
    float_audio = audio.astype(np.float32) / np.iinfo(np.int16).max
    if original_sample_rate != REQUIRED_SAMPLE_RATE:
        float_audio = librosa.core.resample(
            float_audio.T, orig_sr=original_sample_rate, target_sr=REQUIRED_SAMPLE_RATE,
            res_type='kaiser_best')
    float_audio = float_audio.flatten()
    emb_dict = module(samples=float_audio, sample_rate=16000)
    emb = emb_dict['embedding']
    emb.shape.assert_is_compatible_with([None, 512])
    # the paper says layer 19 gives better results
    l19 = emb_dict['layer19']
    l19.shape.assert_is_compatible_with([None, 12288])

    feat = np.average(emb, axis=0)
    l19 = np.average(l19, axis=0)

    feat = torch.as_tensor(np.array(feat).astype('float'))
    l19 = torch.as_tensor(np.array(l19).astype('float'))

    # add a dimension to act as batch dimension
    # feat = torch.unsqueeze(feat, 0)
    # l19 = torch.unsqueeze(l19, 0)
    return feat, l19


def process_audio_file(file_path: str, result_path: str, stride=None):
    clip = AudioSegment.from_file(file_path)
    orig_sampling_rate = clip.frame_rate

    data = OrderedDict()

    # split the audio into channels
    # not sure if this was done before or how TRILL handles stereo data.
    # get_array_of_samples() flattens the 2 channels
    channel_sounds = clip.split_to_mono()
    raw_audio = [np.array(s.get_array_of_samples()) for s in channel_sounds]
    raw_audio = np.array(raw_audio).T

    audio_model = get_audio_model()
    audio_embedding = []
    audio_feature = []
    if stride is None:
        stride = SEGMENT_STRIDE
    arange_iter = np.arange(0.0, clip.duration_seconds, stride)

    for (n, i) in tqdm(enumerate(arange_iter)):
        start_time = int(i * 1000)
        end_time = int(min(i + SEGMENT_DURATION, clip.duration_seconds) * 1000)

        segment = clip[start_time:end_time]
        # pass audio segment to audio based model
        # get ndarray from AudioSegment object
        # audio_array = np.array(segment.get_array_of_samples())
        # audio_features = get_emotion_features_from_audio(audio_array, orig_sampling_rate)
        # audio_valence, audio_arousal, audio_emotion, _ = audio_features

        audio_features = get_emotion_features_from_audio(audio_model, segment, orig_sampling_rate)
        audio_valence, audio_arousal, audio_emotion, _, embedding, layer19 = audio_features

        audio_embedding.append(embedding.cpu().detach().numpy())
        audio_feature.append(layer19.cpu().detach().numpy())

        # Mapping audio outputs to the main label space
        main_audio_emo_prob = bold_to_main(audio_emotion[0])
        main_audio_valence = bold_to_main_valence(audio_valence[0])
        main_audio_arousal = bold_to_main_arousal(audio_arousal[0])
        result = np.array([main_audio_arousal, main_audio_valence, *main_audio_emo_prob])

        start_time = start_time / 1000
        end_time = end_time / 1000
        mid_time = start_time + stride
        if mid_time < clip.duration_seconds:
            if (start_time, mid_time) in data:
                data[(start_time, mid_time)].append(result)
            else:
                data[(start_time, mid_time)] = [result]

            if (mid_time, end_time) in data:
                data[(mid_time, end_time)].append(result)
            else:
                data[(mid_time, end_time)] = [result]
        else:
            if (start_time, end_time) in data:
                data[(start_time, end_time)].append(result)
            else:
                data[(start_time, end_time)] = [result]

    df = []

    for key, value in data.items():
        value = np.stack(value, axis=0).mean(axis=0)
        value[2:] = value[2:] / value[2:].sum()
        df.append({
            "start": key[0],
            "end": key[1],
            "valence": value[1],
            "arousal": value[0],
            "emotion0": value[2],
            "emotion1": value[3],
            "emotion2": value[4],
            "emotion3": value[5],
            "emotion4": value[6],
            "emotion5": value[7],
            "emotion6": value[8],
            "emotion7": value[9],
            "emotion8": value[10],
        })

    if len(df) == 0:
        return None, None, None, None, None, None

    audio_embedding = np.array(audio_embedding)
    audio_feature = np.array(audio_feature)

    results = pd.DataFrame(df)
    if result_path != "":
        pathlib.Path(result_path).parent.mkdir(exist_ok=True, parents=True)
        results.to_csv(result_path, index=False)
        print(f"[Audio Head] Process {pathlib.Path(file_path).parent.name}")

    return results, audio_embedding, audio_feature, raw_audio, orig_sampling_rate, clip.duration_seconds
