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
from .deep_sort.deep_sort import DeepSort
import argparse
import yaml

class VisualES():
    def __init__(self, args):
        self.args = args
    

    def initialize_model(self, face_detector, emotion_recognizer):
        self.face_detector = face_detector
        self.emotion_recognizer = emotion_recognizer

    
    def update_args(self, new_args):
        self.args = new_args

    def interpolate_missing_frames(self, best_match_track_id, idx_frame, all_tracks, all_es_feat_tracks):
        time_interpolate = idx_frame - all_tracks[best_match_track_id]['frames_appear'][-2] - 1
        if time_interpolate > 0:
            old_rep_track = all_es_feat_tracks[best_match_track_id][-1].tolist()
            all_es_feat_tracks[best_match_track_id] = np.append(all_es_feat_tracks[best_match_track_id], [old_rep_track] * time_interpolate, axis=0)
        return all_es_feat_tracks

    def initialize_video_writer(self, clip, output_folder):
        width = int(clip.w)
        height = int(clip.h)
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        # fourcc = cv2.VideoWriter_fourcc(*'MPEG')
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        return cv2.VideoWriter(os.path.join(output_folder, f'{self.args.video_name}.mp4'), fourcc, clip.fps, (width, height), True)
    def cosine_similarity(self, a, b):
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        similarity = dot_product / (norm_a * norm_b)
        return similarity
    def bbox_iou(self, box1, box2):
        # Compute the coordinates of the intersection rectangle
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # Compute the area of the intersection rectangle
        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

        # Compute the area of the union rectangle
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area

        # Compute the IoU
        iou = intersection_area / union_area if union_area > 0 else 0

        return iou


    def update_or_create_new_track(self, track_id, bbox, idx_frame, all_tracks, mark_old_track_idx, all_es_feat_tracks, all_start_end_offset_track, all_emotion_category_tracks, es_feature, emotion_cat):
        best_match_track_id = None
        active_track_indices = [i for i, _ in enumerate(all_tracks) if i not in mark_old_track_idx]

        for idx in active_track_indices:
            each_active_tracks = all_tracks[idx]
            if each_active_tracks['id'] == track_id:
                best_match_track_id = idx
                break

        if best_match_track_id is None:
            # Check if there are any existing tracks that are similar to this one
            similar_track_ids = []
            for idx, each_active_tracks in enumerate(all_tracks):
                if idx in mark_old_track_idx:
                    continue

                # Compute the intersection over union (IoU) between the new bbox and the bbox of each existing track
                iou = self.bbox_iou(bbox, each_active_tracks['bbox'][-1])

                # If the IoU is above a threshold, add the existing track to the list of similar tracks
                if iou > self.args.network.iou_threshold:
                    similar_track_ids.append(idx)

            if len(similar_track_ids) > 0:
                # If there are similar tracks, update the one with the highest score
                best_match_score = -1
                for track_id in similar_track_ids:
                    score = self.compute_track_score(track_id, all_es_feat_tracks, es_feature, self.args.network.similarity_threshold)
                    if score > best_match_score:
                        best_match_score = score
                        best_match_track_id = track_id

            if best_match_track_id is None:
                # Initialize a new track
                new_track = {"bbox": [bbox], "id": track_id, "frames_appear": [idx_frame]}
                all_tracks.append(new_track)

                new_es_array_track = np.array([es_feature])
                new_start_end_offset_track = [idx_frame, idx_frame]  # [start, end]
                new_ec_array_track = np.array([emotion_cat])

                all_emotion_category_tracks.append(new_ec_array_track)
                all_es_feat_tracks.append(new_es_array_track)
                all_start_end_offset_track.append(new_start_end_offset_track)

            else:
                # Update existing track
                all_tracks[best_match_track_id]['bbox'].append(bbox)
                all_tracks[best_match_track_id]['frames_appear'].append(idx_frame)

                all_es_feat_tracks = self.interpolate_missing_frames(best_match_track_id, idx_frame, all_tracks, all_es_feat_tracks)
                all_es_feat_tracks[best_match_track_id] = np.append(all_es_feat_tracks[best_match_track_id], [es_feature], axis=0)  # add more feature for this track
                all_start_end_offset_track[best_match_track_id][-1] = idx_frame  # change index frame

        return all_tracks, all_es_feat_tracks, all_start_end_offset_track, all_emotion_category_tracks

    def compute_track_score(self, track_id, all_es_feat_tracks, es_feature, similarity_threshold):
        last_feature = all_es_feat_tracks[track_id][-1]
        last_feature = np.array(last_feature)  # Convert to numpy array
        
        try:
            sim = self.cosine_similarity(last_feature.reshape(1, -1), np.array(es_feature).reshape(1, -1).T)
        except:
            pdb.set_trace()
        if sim >= similarity_threshold:
            return sim
        else:
            return -1

    def shrink_bbox(self, bbox, shrink_factor=0.4):
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        new_width = width * shrink_factor
        new_height = height * shrink_factor

        center_x = x1 + width / 2
        center_y = y1 + height / 2

        new_x1 = int(center_x - new_width / 2)
        new_y1 = int(center_y - new_height / 2)
        new_x2 = int(center_x + new_width / 2)
        new_y2 = int(center_y + new_height / 2)

        return new_x1, new_y1, new_x2, new_y2
    
    def extract_sequence_frames_DS(self, clip, frame_skip):
        print("===========Finding face tracks==============")
        frames_list = list(clip.iter_frames())

        softmax = torch.nn.Softmax(dim=1)

        all_tracks = []
        mark_old_track_idx = []
        idx_frame = -1

        all_emotion_category_tracks = []
        all_es_feat_tracks = []
        all_start_end_offset_track = []

        if self.args.network.visualize_debug_face_track:
            output_folder = '/home/dhgbao/Research_Monash/code/my_code/unsupervised_approach/multimodal_module/output/test_debug_output'
            output = self.initialize_video_writer(clip, output_folder)

        # Create a new DeepSortTracker object
        tracker = DeepSort(model_path="/home/dhgbao/Research_Monash/code/my_code/unsupervised_approach/multimodal_module/unsupervised/main_modules/ES_extractor/ES_visual/deep_sort/deep/checkpoint/ckpt.t7")

        for idx, frame in tqdm(enumerate(frames_list), total=len(frames_list)):
            if idx % frame_skip != 0:
                continue
            
            idx_frame += 1
            
            # Clear the draw_face_track_bbox list for the current frame
            draw_face_track_bbox = []

            # Detect faces
            bounding_boxes, probs = self.face_detector.detect(frame)
            if bounding_boxes is not None:
                bounding_boxes = bounding_boxes[probs > self.args.network.threshold_face]

            if bounding_boxes is None:
                continue

            # Track the objects using DeepSORT
            try:
                track_bboxes = tracker.update(bounding_boxes, probs, frame)
            except:
                continue
            
            if track_bboxes is None:
                continue

            # Process dying tracks
            for idx, each_active_tracks in enumerate(track_bboxes):
                old_idx_frame = each_active_tracks[-1]
                if idx_frame - old_idx_frame > self.args.network.threshold_dying_track_len:
                    mark_old_track_idx.append(idx)

            # Assign new boxes to remaining active tracks or create a new track if there are active tracks
            for track in tracker.tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                
                track_id = track.track_id
                
                # try:
                #     bbox_detector = bounding_boxes[track_id].astype(int)
                # except:
                #     continue
                bbox_tracker = track.to_x1y1x2y2().astype(int)
                
                [x1, y1, x2, y2] = bbox_tracker
                
                x1, x2 = min(max(0, x1), frame.shape[1]), min(max(0, x2), frame.shape[1])
                y1, y2 = min(max(0, y1), frame.shape[0]), min(max(0, y2), frame.shape[0])
                
                face_imgs = frame[y1:y2, x1:x2]
                
                

                try:
                    emotion, scores = self.emotion_recognizer.predict_emotions(face_imgs, logits=True)
                except:
                    continue

                scores = softmax(torch.Tensor(np.array([scores])))
                es_feature = scores[0].tolist()
                emotion_cat = emotion

                all_tracks, all_es_feat_tracks, all_start_end_offset_track, all_emotion_category_tracks = self.update_or_create_new_track(
                    track_id, bbox_tracker, idx_frame, all_tracks, mark_old_track_idx, all_es_feat_tracks,
                    all_start_end_offset_track, all_emotion_category_tracks, es_feature, emotion_cat
                )
                

                if self.args.network.visualize_debug_face_track:
                    #shrink bbox
                    # x1, y1, x2, y2 = self.shrink_bbox((x1, y1, x2, y2), shrink_factor=0.8)
                    bbox = [x1, y1, x2, y2]
                    
                    draw_face_track_bbox.append([bbox, track_id])

            if self.args.network.visualize_debug_face_track:
                all_box = [l[0] for l in draw_face_track_bbox]
                all_id = ["ID:"+str(a[1]) for a in draw_face_track_bbox]

                frame = bbv.draw_multiple_rectangles(frame, all_box)
                frame = bbv.add_multiple_labels(frame, all_id, all_box)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Create folder for debug frames
                debug_folder = 'debug_frames'
                os.makedirs(debug_folder, exist_ok=True)

                # Inside your loop where you process each frame, add this code to save a debug image:
                # Replace idx with your frame index variable
                debug_frame_path = os.path.join(debug_folder, f'{idx:04d}.png')
                cv2.imwrite(debug_frame_path, frame)
                output.write(frame)

            if self.args.network.max_idx_frame_debug is not None:
                if idx_frame >= self.args.network.max_idx_frame_debug:
                    break

        if self.args.network.visualize_debug_face_track:
            output.release()

        all_es_feat_tracks_filter = []
        all_start_end_offset_track_filter = []
        all_emotion_category_tracks_filter = []
        
        for es_feat_track, se_track, ec_track in zip(all_es_feat_tracks, all_start_end_offset_track, all_emotion_category_tracks):
            length = es_feat_track.shape[-1]
            
            if length >= self.args.network.len_face_tracks:
                all_es_feat_tracks_filter.append(es_feat_track)
                all_start_end_offset_track_filter.append(se_track)
                all_emotion_category_tracks_filter.append(ec_track)

        return all_es_feat_tracks_filter, all_emotion_category_tracks_filter, all_start_end_offset_track_filter

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
    path_test_video = "/home/dhgbao/Research_Monash/dataset/ccu-data/CCU_COMBINED_ANNOTATED_VIDEO/all_videos/M01000AJ7_0008.mp4"
    config.video_name = path_test_video.split("/")[-1].replace(".mp4", "")
    face_detector = MTCNN(keep_all=False, post_process=True, min_face_size=config.network.min_face_size, device=config.pipeline.device)
    emotion_recognizer = HSEmotionRecognizer(model_name=config.network.model_name, device=config.pipeline.device)
    
    ES_extractor = VisualES(config)
    ES_extractor.initialize_model(face_detector, emotion_recognizer)

    clip = VideoFileClip(path_test_video)
    all_es_feat_tracks, all_emotion_category_tracks, all_start_end_offset_track = ES_extractor.extract_sequence_frames_DS(clip, frame_skip=config.network.min_frame_per_second)
    
    print(all_emotion_category_tracks)
    # save feature for later usage
    # with open('test_es_feature.npy', 'wb') as f:
    #     np.save(f, np.array(all_es_feat_tracks))



