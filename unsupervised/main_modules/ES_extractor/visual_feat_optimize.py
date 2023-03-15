# Extracting visual features representing for emotional signals of individual faces 
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pdb
import bbox_visualizer as bbv
import torch

from facenet_pytorch import MTCNN
from hsemotion.facial_emotions import HSEmotionRecognizer
from dotmap import DotMap

def cal_iou(bbox1, bbox2):
    """
    Calculates the intersection-over-union of two bounding boxes.
        Args:
            bbox1 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
            bbox2 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
        Returns:
            int: intersection-over-onion of bbox1, bbox2
    """
    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, x1_1, y1_1) = bbox1
    (x0_2, y0_2, x1_2, y1_2) = bbox2

    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union

class VisualES():
    def __init__(self, args):
        self.args = args
    

    def initialize_model(self, face_detector, emotion_recognizer):
        self.face_detector = face_detector
        self.emotion_recognizer = emotion_recognizer

    
    def update_args(self, new_args):
        self.args = new_args
    

    def extract_single_frame(self, img):
        # detect bounding boxes
        bounding_boxes, probs = self.face_detector.detect(img, landmarks=False)
        if bounding_boxes is not None:
            bounding_boxes = bounding_boxes[probs>self.args.threshold_face] # threshold_face = 0.6


    def filter_noisy_track(self, all_tracks):
        # filter noisy track based on multiple criteria:
        #   1. Track length must exceed ...

        filter_tracks = []
        for each_track in all_tracks:
            length = len(each_track['frames_appear'])
            if length >= self.args.len_face_tracks:
                filter_tracks.append(each_track)

        return filter_tracks

    
    def interpolate_track(self, all_tracks):
        # interpolate gaps between frame with previous frame box
        interpolate_all_tracks = []

        for each_track in all_tracks:
            current_track = {"bbox": [], "id": each_track["id"], "frames_appear": []}

            start_frame = each_track["frames_appear"][0]
            end_frame = each_track["frames_appear"][1]

            run_frame = start_frame
            prev_frame = start_frame - 1
            prev_box = None

            for idx_box in range(len(each_track["bbox"])-1):
                frame_id = each_track["frames_appear"][idx_box]
                
                if frame_id - prev_frame > 1:
                    # should interpolate the middle, take the recent box to interpolate

                    times_repeat = frame_id - prev_frame - 1
                    all_middle_box = list(np.repeat([current_track["bbox"][-1]], times_repeat, 0))
                    current_track["bbox"].extend(all_middle_box)
                    current_track["frames_appear"].extend(range(prev_frame+1, frame_id))

                # add current
                current_track["bbox"].append(each_track["bbox"][idx_box])
                current_track["frames_appear"].append(frame_id)
                prev_frame = frame_id

            # remember to add the final frame
            current_track["bbox"].append(each_track["bbox"][-1])
            current_track["frames_appear"].append(each_track["frames_appear"][-1])

            interpolate_all_tracks.append(current_track)

        return interpolate_all_tracks
            

    # def extract_sequence_frames(self, video_path):
    #     print("==========Extracting ES features==============")
    #     # find all face tracks in a video
    #     all_face_tracks = self.find_face_tracks(video_path)

    #     softmax = torch.nn.Softmax(dim=1)

    #     #### Preprocessing facial tracks

    #     # filter noisy tracks       
    #     all_face_tracks = self.filter_noisy_track(all_face_tracks)

    #     # interpolate facial tracks
    #     ### !!!!! NO Interpolation here !!!!!!!
    #     all_face_tracks = self.interpolate_track(all_face_tracks)

    #     # iterate and extract
    #     all_es_feat_tracks = []
    #     all_emotion_category_tracks = []
    #     all_start_end_offset_track = []

    #     # This is for debugging visualization only
    #     cap = cv2.VideoCapture(video_path)

    #     for each_track in all_face_tracks:
    #         frames_appear = each_track['frames_appear']
    #         bbox = each_track['bbox']
            
    #         facial_img_track = []
    #         es_feature_track = []
    #         emotion_cat_track = []

    #         start_end_track = (frames_appear[0], frames_appear[-1])

    #         for each_frame, each_box in zip(frames_appear, bbox):
    #             [x1, y1, x2, y2] = each_box


    #             cap.set(cv2.CAP_PROP_POS_FRAMES, each_frame)

    #             ret, frame = cap.read()

    #             x1, x2  = min(max(0, x1), frame.shape[1]), min(max(0, x2), frame.shape[1])
    #             y1, y2 = min(max(0, y1), frame.shape[0]), min(max(0, y2), frame.shape[0])

    #             face_imgs = frame[y1:y2, x1:x2]
    #             emotion, scores = self.emotion_recognizer.predict_emotions(face_imgs, logits=True)
    #             scores = softmax(torch.Tensor(np.array([scores])))

    #             es_feature_track.append(scores[0].tolist())
    #             emotion_cat_track.append(emotion)

    #             facial_img_track.append(face_imgs) # this is just for visualization, comment this

    #         es_feature_track = np.array(es_feature_track)

    #         # debug writing here
    #         # for idx, each_face_track_img in enumerate(facial_img_track):
    #         #     name_path = os.path.join('./test_facial_track_debug', str(idx)+'.png')
    #         #     cv2.imwrite(name_path, each_face_track_img)


    #         all_emotion_category_tracks.append(emotion_cat_track)
    #         all_es_feat_tracks.append(es_feature_track)
    #         all_start_end_offset_track.append(start_end_track)
        
    #     cap.release()

    #     return all_es_feat_tracks, all_emotion_category_tracks, all_start_end_offset_track


    def extract_sequence_frames(self, video_path):
        # finding all face tracks in video. A face track is defined as t = (l, t) where:
        #   + l represents for list of face location for that track
        #   + t represents for frame-index to the video of that track
        print("===========Finding face tracks==============")
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(3)) # get width
        height = int(cap.get(4)) #get height
        softmax = torch.nn.Softmax(dim=1)
        
        all_tracks = []
        mark_old_track_idx = []
        idx_frame = -1

        all_emotion_category_tracks = []
        all_es_feat_tracks = []
        all_start_end_offset_track = []

        # FOR VISUALIZING ONLY
        if self.args.visualize_debug_face_track == True:
            fourcc = cv2.VideoWriter_fourcc(*'MPEG')
            output = cv2.VideoWriter(os.path.join('./test_debug_output', 'out_test_vid.mp4'), fourcc, 5.0, (width, height))

        while(cap.isOpened()):
            ret, frame = cap.read()
            
            if ret == False:
                break

            # skip frame
            idx_frame += 1
            if idx_frame % self.args.skip_frame != 0:
                continue

            # FOR VISUALIZING ONLY
            draw_face_track_bbox = []
            progress_frame = str(idx_frame)+"/"+str(self.args.total_vid_frame)
                
            print("Processing frame: ", progress_frame, video_path, self.args.batch_run)

            # detect faces
            bounding_boxes, probs = self.face_detector.detect(frame, landmarks=False)
            if bounding_boxes is not None:
                bounding_boxes = bounding_boxes[probs>self.args.threshold_face] # threshold_face = 0.6

            if bounding_boxes is None:
                continue
            

            # Stage 1: Process dying tracks
            for idx, each_active_tracks in enumerate(all_tracks):
                old_idx_frame = each_active_tracks['frames_appear'][-1]

                if idx_frame - old_idx_frame > self.args.threshold_dying_track_len:
                    # this is the dying track, mark it
                    mark_old_track_idx.append(idx)

            # Stage 2: Assign new boxes to remaining active tracks or create a new track if there are active tracks

            for idx, bbox in enumerate(bounding_boxes):
                box = bbox.astype(int)
                # x1, y1, x2, y2 = box[0:4]   
                
                # check each track
                best_match_track_id = None
                best_match_track_score = 0


                # ====================
                # Stage 2.0: Extracting ES features from facial image
                [x1, y1, x2, y2] = box
                x1, x2  = min(max(0, x1), frame.shape[1]), min(max(0, x2), frame.shape[1])
                y1, y2 = min(max(0, y1), frame.shape[0]), min(max(0, y2), frame.shape[0])

                face_imgs = frame[y1:y2, x1:x2]
                emotion, scores = self.emotion_recognizer.predict_emotions(face_imgs, logits=True)

                # softmax as feature
                scores = softmax(torch.Tensor(np.array([scores])))
                es_feature = scores[0].tolist() ### this is what we need
                emotion_cat = emotion ### this is what we need

                # raw feature
                # es_feature = scores.tolist()
                # emotion_cat = emotion

                # =====================
                # Stage 2.1: Finding to which track this es_feature belongs to based on iou
                for idx, each_active_tracks in enumerate(all_tracks):
                    if idx in mark_old_track_idx:
                        # ignore inactive track
                        continue
                    latest_track_box = each_active_tracks['bbox'][-1]

                    iou_score = cal_iou(latest_track_box, box)
                    
                    if iou_score > best_match_track_score and iou_score > self.args.threshold_iou_min_track:
                        best_match_track_id = idx
                        best_match_track_score = iou_score

                if best_match_track_id is None:
                    # there is no active track currently, then this will initialize a new track
                    new_track = {"bbox": [box], "id": len(all_tracks), "frames_appear": [idx_frame]}
                    all_tracks.append(new_track)


                    # also create new np array representing for new track here
                    new_es_array_track = np.array([es_feature])
                    new_start_end_offset_track = [idx_frame, idx_frame] #[start, end]

                    all_es_feat_tracks.append(new_es_array_track)
                    all_start_end_offset_track.append(new_start_end_offset_track)


                    # FOR VISUALIZING ONLY
                    if self.args.visualize_debug_face_track == True:
                        draw_face_track_bbox.append([box, new_track["id"]])
                else:
                    # update track
                    all_tracks[best_match_track_id]['bbox'].append(box)
                    all_tracks[best_match_track_id]['frames_appear'].append(idx_frame)

                    # update all_list

                    ### interpolate first

                    time_interpolate = idx_frame - all_tracks[best_match_track_id]['frames_appear'][-2] - 1

                    if time_interpolate > 0:
                        old_rep_track = all_es_feat_tracks[best_match_track_id][-1].tolist()
                        all_es_feat_tracks[best_match_track_id] = np.append(all_es_feat_tracks[best_match_track_id], [old_rep_track]*time_interpolate, axis=0)

                    ### then do update
                    all_es_feat_tracks[best_match_track_id] = np.append(all_es_feat_tracks[best_match_track_id], [es_feature], axis=0) # add more feature for this track
                    all_start_end_offset_track[best_match_track_id][-1] = idx_frame # change index frame

                    if self.args.visualize_debug_face_track == True:
                    # FOR VISUALIZING ONLY
                        draw_face_track_bbox.append([box, all_tracks[best_match_track_id]['id']])
                

            # FOR VISUALIZING ONLY, draw all face track box
            if self.args.visualize_debug_face_track == True:
                all_box = [l[0] for l in draw_face_track_bbox]
                all_id = ["ID:"+str(a[1]) for a in draw_face_track_bbox]

                # frame = bbv.draw_multiple_rectangles(frame, all_box)
                # frame = bbv.add_multiple_labels(frame, all_id, all_box)
                
                output.write(frame)

            if self.args.max_idx_frame_debug is not None:
                if idx_frame >= self.args.max_idx_frame_debug:
                    break

        cap.release()
        if self.args.visualize_debug_face_track == True:
            output.release()


        # filter again es signals, ignoring those tracks having length lesser than pre-defined numbers
        all_es_feat_tracks_filter = []
        all_start_end_offset_track_filter = []

        for es_feat_track, se_track in zip(all_es_feat_tracks, all_start_end_offset_track):
            length = es_feat_track.shape[0]
            if length >= self.args.len_face_tracks:
                all_es_feat_tracks_filter.append(es_feat_track)
                all_start_end_offset_track_filter.append(se_track)

        return all_es_feat_tracks_filter, None, all_start_end_offset_track_filter

            
if __name__ == "__main__":
    # test args 
    args = DotMap()
    args.device = 'cuda:1'
    args.threshold_face = 0.6
    args.model_name = 'enet_b0_8_best_afew'
    args.threshold_dying_track_len = 30
    args.threshold_iou_min_track = 0.3
    args.max_idx_frame_debug = 1000
    args.len_face_tracks = 20
    args.visualize_debug_face_track = False
    args.total_vid_frame = 1000


    # test
    testES = VisualES(args)

    path_test_video = "/home/nttung/research/Monash_CCU/mini_eval/visual_data/r2_v1_video/all_DARPA_video/format_mp4_video/M01000FN8.mp4"
    all_es_feat_tracks, all_emotion_category_tracks, all_start_end_offset_track = testES.extract_sequence_frames(path_test_video)
    pdb.set_trace()

    # save feature for later usage
    # with open('test_es_feature.npy', 'wb') as f:
    #     np.save(f, np.array(all_es_feat_tracks))


