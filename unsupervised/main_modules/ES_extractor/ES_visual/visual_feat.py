# Extracting visual features representing for emotional signals of individual faces 
import os
import numpy as np
import cv2
import pdb
from facenet_pytorch import MTCNN
from hsemotion.facial_emotions import HSEmotionRecognizer
import bbox_visualizer as bbv
import torch
from dotmap import DotMap
from moviepy.editor import *
from tqdm import tqdm
from pathlib import Path
from deep_sort.deep_sort import DeepSort
import argparse
import yaml
from deep_sort.deep.feature_extractor import Extractor

class VisualES():
    def __init__(self, args):
        self.args = args
        self.extractor = Extractor("./deep_sort/deep/checkpoint/ckpt.t7", use_cuda=True)

    def initialize_model(self, face_detector, emotion_recognizer):
        self.face_detector = face_detector
        self.emotion_recognizer = emotion_recognizer

    
    def update_args(self, new_args):
        self.args = new_args

    def initialize_video_writer(self, clip, output_folder):
        width = int(clip.w)
        height = int(clip.h)
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        # fourcc = cv2.VideoWriter_fourcc(*'MPEG')
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        return cv2.VideoWriter(os.path.join(output_folder, f'{self.args.video_name}.mp4'), fourcc, clip.fps, (width, height), True)

    def convert_bbox_xyxy_to_xywh(self, bbox_list_xyxy):
        bbox_list_xywh = []
        for bbox_xyxy in bbox_list_xyxy:
            x_min, y_min, x_max, y_max = bbox_xyxy
            bbox_w = x_max - x_min
            bbox_h = y_max - y_min
            bbox_xc = x_min + bbox_w / 2
            bbox_yc = y_min + bbox_h / 2
            bbox_xyxy = [bbox_xc, bbox_yc, bbox_w, bbox_h]
            bbox_list_xywh.append(bbox_xyxy)
        return bbox_list_xywh

    def iou(self, bbox1, bbox2):
        """
        Calculate intersection over union (IoU) between two bounding boxes.

        Parameters:
        - bbox1, bbox2: lists or arrays of four elements (x1, y1, x2, y2)

        Returns:
        - iou: IoU score between the two bounding boxes
        """
        # Calculate intersection coordinates
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        # Calculate intersection area
        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        # Calculate union area
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        # Calculate IoU score
        iou = intersection / union if union > 0 else 0

        return iou

    def extract_sequence_frames_custom(self, clip, frame_skip):
        print("===========Finding face tracks==============")
        frames_list = list(clip.iter_frames())
        softmax = torch.nn.Softmax(dim=1)


        all_tracks = {}
        all_emotion_category_tracks = []
        all_es_feat_tracks = []
        all_start_end_offset_track = []
        

        if self.args.network.visualize_debug_face_track:
            output_folder = '/home/dhgbao/Research_Monash/code/my_code/unsupervised_approach/multimodal_module/unsupervised/main_modules/ES_extractor/ES_visual/test_debug_output'
            output = self.initialize_video_writer(clip, output_folder)

        # Loop through each frame
        for idx, frame in tqdm(enumerate(frames_list), total=len(frames_list)):
            if idx % frame_skip != 0:
                continue
            
            draw_face_track_bbox = []
            # Detect faces
            bounding_boxes, probs = self.face_detector.detect(frame)
            if bounding_boxes is not None:
                bounding_boxes = bounding_boxes[probs > self.args.network.threshold_face]
                probs = probs[probs > self.args.network.threshold_face]

            if bounding_boxes is None:
                if self.args.network.visualize_debug_face_track:
                    print("No faces found!")
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 
                    # pdb.set_trace()
                    output.write(frame)
                continue

            # Loop through each detected face
            for bbox_idx, bbox_xyxy in enumerate(bounding_boxes):
                # bbox_xyxy = self.convert_bbox_xyxy_to_xywh([bbox_xyxy])[0]
                x1, y1, x2, y2 = bbox_xyxy
                x1, x2 = min(max(0, x1), frame.shape[1]), min(max(0, x2), frame.shape[1])
                y1, y2 = min(max(0, y1), frame.shape[0]), min(max(0, y2), frame.shape[0])
                # Extract face features
                face_img = frame[int(y1):int(y2), int(x1):int(x2)]
                # try:
                #     face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR) 
                # except:
                #     continue
                # cv2.imwrite(f"face_{idx}.png", face_img)
                try:
                    face_feat = self.extractor([face_img])[0]
                except:
                    pdb.set_trace()
                    continue
                
                emotion_cat, logits = self.emotion_recognizer.predict_emotions(face_img, logits=True)
                scores = softmax(torch.Tensor(np.array([logits])))
                es_feature = scores[0].tolist()
                
                # Compare to existing tracks to find a match
                best_match_track_id = None
                best_match_score = 0
                for track_id, track_data in all_tracks.items():
                    # Calculate IoU between the face and the track's last bounding box
                    last_bbox_xyxy = track_data['bbox'][-1]
                    iou_score = self.iou(bbox_xyxy, last_bbox_xyxy)

                    # Calculate similarity between the face features and the track's feature vectors
                    track_feats = all_es_feat_tracks[track_id-1]
                    sim_score = np.dot(face_feat, track_feats.T).max()

                    # Combine IoU and similarity scores to get a total match score
                    total_score = sim_score * iou_score


                    # Update best match if this track has the highest score so far
                    if total_score > best_match_score:
                        best_match_track_id = track_id
                        best_match_score = total_score

                if best_match_track_id is not None:
                    # pdb.set_trace()
                    # Update existing track with new bounding box, emotion category, and feature vector
                    all_tracks[best_match_track_id]['bbox'].append(bbox_xyxy)
                    all_tracks[best_match_track_id]['frames_appear'].append(idx)

                    new_es_array_track = np.array([face_feat])
                    all_es_feat_tracks[best_match_track_id-1] = np.concatenate([all_es_feat_tracks[best_match_track_id-1], new_es_array_track])
                    
                    all_emotion_category_tracks[best_match_track_id-1] = np.concatenate([all_emotion_category_tracks[best_match_track_id-1], [emotion_cat]])
                    all_start_end_offset_track[best_match_track_id-1][-1] = idx
                    if self.args.network.visualize_debug_face_track:
                        bbox = [x1, y1, x2, y2]
                        draw_face_track_bbox.append([bbox, best_match_track_id])
                else:
                    # Initialize a new track
                    new_track_id = len(all_tracks) + 1
                    new_bbox_list = [bbox_xyxy]
                    new_frames_list = [idx]
                    all_tracks[new_track_id] = {"bbox": new_bbox_list, "frames_appear": new_frames_list}

                    new_es_array_track = np.array([face_feat])
                    all_es_feat_tracks.append(new_es_array_track)
                    new_ec_array_track = np.array([emotion_cat])
                    all_emotion_category_tracks.append(new_ec_array_track)

                    new_start_end_offset_track = [idx, idx]  # [start, end]
                    all_start_end_offset_track.append(new_start_end_offset_track)
                
                    if self.args.network.visualize_debug_face_track:
                        bbox = [x1, y1, x2, y2]
                        draw_face_track_bbox.append([bbox, new_track_id])
                    
            if self.args.network.visualize_debug_face_track:
                all_box = [[int(coord) for coord in l[0]] for l in draw_face_track_bbox]
                all_id = ["ID:"+str(a[1]) for a in draw_face_track_bbox]

                frame = bbv.draw_multiple_rectangles(frame, all_box)
                frame = bbv.add_multiple_labels(frame, all_id, all_box)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)    
                output.write(frame)


            if idx >= self.args.network.max_idx_frame_debug:
                print("Stop!")
                break

        if self.args.network.visualize_debug_face_track:
            output.release()

        return all_tracks, all_es_feat_tracks, all_start_end_offset_track, all_emotion_category_tracks

                    
                


def get_args_parser():
    parser = argparse.ArgumentParser("Parsing arguments", add_help=False)
    parser.add_argument("--config", default="/home/dhgbao/Research_Monash/code/my_code/unsupervised_approach/multimodal_module/configs/default.yaml", type=str)

    return parser

if __name__ == "__main__":
    ##### Defining arguments #####
    parser = argparse.ArgumentParser(
        "UCP detection inference on multi-modal data", parents=[get_args_parser()])
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    print(config_dict)
    config = DotMap(config_dict)
    # path_test_video = "/home/dhgbao/Research_Monash/dataset/ccu-data/CCU_COMBINED_ANNOTATED_VIDEO/all_videos/M0100405W_0013.mp4"
    # path_test_video = "M01003MTK.mp4"
    path_test_video = "/home/dhgbao/Research_Monash/dataset/ccu-data/CCU_COMBINED_ANNOTATED_VIDEO/all_videos/M01003MTK_0002.mp4"
    # path_test_video = "/home/dhgbao/Research_Monash/dataset/ccu-data/CCU_COMBINED_ANNOTATED_VIDEO/all_videos/M01004JM6_0005.mp4"

    config.video_name = path_test_video.split("/")[-1].replace(".mp4", "")
    face_detector = MTCNN(keep_all=False, post_process=True, min_face_size=config.network.min_face_size, device=config.pipeline.device)
    emotion_recognizer = HSEmotionRecognizer(model_name=config.network.model_name, device=config.pipeline.device)
    
    ES_extractor = VisualES(config)
    ES_extractor.initialize_model(face_detector, emotion_recognizer)

    clip = VideoFileClip(path_test_video)
    print("FPS: ", clip.fps)
    all_tracks, all_es_feat_tracks, all_start_end_offset_track, all_emotion_category_tracks = ES_extractor.extract_sequence_frames_custom(clip, frame_skip=config.network.min_frame_per_second)
    
    print("Emotion category list length: ", len(all_emotion_category_tracks))
    for track_id, track in all_tracks.items():        
        print(f"Emotion list for track {track_id}: ", all_emotion_category_tracks[track_id-1])
        print(f"Emotion features shape for track {track_id}: ", all_es_feat_tracks[track_id-1].shape)
        print("\n")
    print("[Start_frame, end_frame]:", all_start_end_offset_track)
    
    fps = clip.fps
    seconds_all_start_end = []
    seconds_all_start_end = [[int(start_frame / clip.fps), int(end_frame / clip.fps)] for start_frame, end_frame in all_start_end_offset_track]
    print("[Start_second, end_second]: ", seconds_all_start_end)
    print("Track ids: ", all_tracks.keys())
    print(all_tracks.items())
        # save feature for later usage
    # with open('test_es_feature.npy', 'wb') as f:
    #     np.save(f, np.array(all_es_feat_tracks))



