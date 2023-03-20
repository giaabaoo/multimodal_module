import torch
import os

def get_device() -> torch.device:
    """
    Get torch device
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

DEVICE = get_device()
SEGMENT_STRIDE = 7.5
SEGMENT_DURATION = 15.0
REQUIRED_SAMPLE_RATE = 16000
FACE_DETECT_THRESH = 0.9
MAX_BATCH = 512  # maximum number of face images to pass to the visual model at once when using batch mode.

AUDIO_MODEL_PATH = os.path.join("/home/dhgbao/Research_Monash/code/my_code/unsupervised_approach/multimodal_module/checkpoints", "audio_model_trill.pt")
AUDIO_MODEL_URL = "https://github.com/islam-nassar/ccu_mini_eval/releases/download/misc/audio_model_trill.pt"
