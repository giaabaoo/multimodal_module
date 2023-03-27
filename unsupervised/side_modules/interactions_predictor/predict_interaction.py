import torch
import torchaudio
import torchvision.transforms as T
import cv2
import torchvision

if __name__ == '__main__':
    # Load the MuTUAL model
    model_name = 'c2vlm/mutual'
    model = torch.hub.load('facebookresearch/detectron2:main', 'mask_rcnn_R_50_FPN_3x', pretrained=True)

    # Define the input video and audio files
    video_file = '/home/dhgbao/Research_Monash/dataset/ccu-data/CCU_COMBINED_ANNOTATED_VIDEO/segmented_videos/M01000AJ7_0010.mp4'
    audio_file = '/home/dhgbao/Research_Monash/dataset/ccu-data/CCU_COMBINED_ANNOTATED_VIDEO/segmented_videos_audio/M01000AJ7_0010.mp3'

    # Define the video and audio transforms
    video_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    audio_transform = T.Compose([
        T.Resample(orig_freq=44100, new_freq=16000),
        T.DownmixMono(),
        T.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=512, n_mels=128)
    ])

    # Load the input video and audio files
    cap = cv2.VideoCapture(video_file)
    audio, sr = torchaudio.load(audio_file)

    # Initialize a list to store the features for each frame
    frame_features = []

    # Loop through the frames in the input video
    while(cap.isOpened()):
        # Read a frame from the video
        ret, frame = cap.read()

        if ret:
            # Resize and normalize the frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = video_transform(frame)
            
            # Extract the audio features for the corresponding time window
            start_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            end_time = start_time + (1 / sr * audio.shape[1])
            audio_window = audio[:, int(start_time*sr):int(end_time*sr)]
            audio_window = audio_transform(audio_window).squeeze()
            
            # Concatenate the visual and audio features along the time dimension
            features = torch.cat((frame.unsqueeze(0), audio_window.unsqueeze(0)), dim=1)
            frame_features.append(features)
        else:
            break

    # Concatenate the features for all the frames
    features = torch.cat(frame_features, dim=0)

    # Perform inference using the MuTUAL model
    with torch.no_grad():
        outputs = model(features)

    # Convert the output probabilities to interaction labels
    interaction_labels = torch.argmax(outputs, dim=1)

    # Print the predicted interaction labels for each frame
    for i, label in enumerate(interaction_labels):
        print('Frame {}: {}'.format(i, label))
