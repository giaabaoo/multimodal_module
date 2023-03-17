import pathlib
from collections import OrderedDict

import librosa
import numpy as np
import pandas as pd
import torch
from pydub import AudioSegment
from torch.autograd import Variable
from tqdm import tqdm

from audio_head import get_audio_model, get_trill_model
from utils.consts import DEVICE, SEGMENT_STRIDE, SEGMENT_DURATION, REQUIRED_SAMPLE_RATE
from utils.label_space_mapping import bold_to_main, bold_to_main_valence, bold_to_main_arousal


