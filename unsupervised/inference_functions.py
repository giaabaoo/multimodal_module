from moviepy.editor import *
import json
from datetime import datetime
import logging
import numpy as np 
import torch
from main_modules.CP_aggregator.segment_core import UniformSegmentator
from main_modules.CP_aggregator.aggregator_core import SimpleAggregator
from code.my_code.unsupervised_approach.multimodal_module.unsupervised.main_modules.ES_extractor.ES_audio.audio_feat import get_audio_features
from main_modules.UCP.inference_ucp import detect_CP_tracks
from utils import draw_result_graph
import pdb

def get_unsupervised_scores(config, ES_extractor):
    # load video file
    video = VideoFileClip(config.single_video_path)

    # calculate number of frames, fps and duration of the video
    video_num_frames = int(video.fps * video.duration)
    video_fps = int(video.fps)
    video_duration = int(video.duration)

    # print information about video
    print(f"Total video frames: {video_num_frames}")
    print(f'Start second of segment: {config.start_second}')
    print(f'End second of segment: {config.end_second}')
    print("===================")

    # update config with video information
    config.video_duration = video_duration
    config.fps = video_fps
    config.video_num_frames = video_num_frames

    # update configuration of ES extractor with new video information
    ES_extractor.update_args(config)

    # extract ES signals, all emotion category tracks, and all start-end offset tracks
    es_signals, all_emotion_category_tracks, all_start_end_offset_track = ES_extractor.extract_sequence_frames(video)
    
    if config.network.use_audio_features:
        pdb.set_trace()
        audio_signals, audio_start_end_offset_track = get_audio_features(video)
        es_signals += audio_signals
        all_start_end_offset_track += audio_start_end_offset_track
        
    # initialize flags for no CP detected
    no_cp_confirm1 = False
    no_cp_confirm2 = False

    # check if there are any ES signals
    if len(es_signals) == 0:
        no_cp_confirm1 = True

    # if there are ES signals, detect CPs using UCP Detector
    if no_cp_confirm1 is False:
        all_peaks_track, all_scores_track, all_start_end_offset_track = detect_CP_tracks(es_signals, all_start_end_offset_track)
        if len(all_peaks_track) == 0:
            no_cp_confirm2 = True

    # if there are no CPs detected, skip post-processing and aggregation
    if no_cp_confirm1 is False and no_cp_confirm2 is False:
        # create Softmax object for post-processing
        softmax = torch.nn.Softmax(dim=1)

        # refine peak indices using start-end offset tracks
        all_refined_peaks_track = []
        all_scores_pick_softmax_track = []
        for each_peak_track, each_start_end_offset_track, each_score_track in zip(all_peaks_track, all_start_end_offset_track, all_scores_track):
            start_idx_track = each_start_end_offset_track[0]
            all_refined_peaks_track.append(each_peak_track + start_idx_track)

            # also pick out score scalar value for specific change point location of that track
            score_pick_track = []
            for each_cp_pos in each_peak_track:
                score_pick_track.append(each_score_track[each_cp_pos])

            # apply Softmax to score_pick_track
            softmax_score = softmax(torch.Tensor(np.array([score_pick_track])))
            all_scores_pick_softmax_track.append(softmax_score[0].tolist())
        
        # convert refined peak indices to binary matrix
        binary_cp_matrix = np.zeros((len(all_refined_peaks_track), config.video_num_frames))
        score_cp_matrix = np.zeros((len(all_refined_peaks_track), config.video_num_frames))

        # num track x num frames
        for idx_track, each_track in enumerate(all_refined_peaks_track):
            for i, each_cp_index in enumerate(each_track):
                try:
                    binary_cp_matrix[idx_track][each_cp_index] = 1
                except:
                    continue
                score_cp_matrix[idx_track][each_cp_index] = all_scores_pick_softmax_track[idx_track][i]
        
        # Convert frame-level prediction into timestamp-level prediction 
        # For each track, a timestamp prediction is the frame with the highest score within that range
        config.timestamp_interval = 1 # interval between timestamps is 1 second
        num_timestamps = int(config.video_num_frames / (video_fps * 1.0 / config.timestamp_interval))
        binary_cp_matrix_ts = np.zeros((len(all_refined_peaks_track), num_timestamps))
        score_cp_matrix_ts = np.zeros((len(all_refined_peaks_track), num_timestamps))

        for ts in range(num_timestamps):
            start_frame = int(ts * video_fps * config.timestamp_interval)
            end_frame = int((ts + 1) * video_fps * config.timestamp_interval)
            for track_idx in range(len(score_cp_matrix_ts)):
                track_scores = score_cp_matrix[track_idx, start_frame:end_frame]
                if len(track_scores) == 0:
                    binary_cp_matrix_ts[track_idx, ts] = 0
                    score_cp_matrix_ts[track_idx, ts] = 0
                else:
                    max_score_idx = np.argmax(track_scores)
                    binary_cp_matrix_ts[track_idx, ts] = binary_cp_matrix[track_idx, start_frame + max_score_idx]
                    score_cp_matrix_ts[track_idx, ts] = track_scores[max_score_idx]


            
        ### Debug converting frame-level scores into timestamp-level
        draw_result_graph(config, score_cp_matrix, score_cp_matrix_ts)
    return score_cp_matrix_ts, binary_cp_matrix_ts

def run_pipeline_single_video(config, ES_extractor):
    # load video file
    video = VideoFileClip(config.single_video_path)

    # calculate number of frames, fps and duration of the video
    video_num_frames = int(video.fps * video.duration)
    video_fps = int(video.fps)
    video_duration = int(video.duration)

    # initialize empty lists to store results
    final_res_cp, final_res_score, final_la, final_res_stat, final_start_end_segment = [], [], [], [], []

    # start time
    start = datetime.now()

    # logging.info information about video
    logging.info(f"Total video frames: {video_num_frames}")
    logging.info(f'Start second of segment: {config.start_second}')
    logging.info(f'End second of segment: {config.end_second}')
    logging.info("===================")

    # update config with video information
    config.video_duration = video_duration
    config.fps = video_fps
    config.video_num_frames = video_num_frames

    # update configuration of ES extractor with new video information
    ES_extractor.update_args(config)

    # extract ES signals, all emotion category tracks, and all start-end offset tracks
    es_signals, all_emotion_category_tracks, all_start_end_offset_track = ES_extractor.extract_sequence_frames(video)
    
    if config.network.use_audio_features:
        audio_signals, audio_start_end_offset_track = get_audio_features(video)
        es_signals += audio_signals
        all_start_end_offset_track += audio_start_end_offset_track
        
    # pdb.set_trace()
    # pdb.set_trace()
    # initialize flags for no CP detected
    no_cp_confirm1 = False
    no_cp_confirm2 = False

    # initialize lists for storing result
    la = []
    res_stat = []
    res_cp = []
    res_score = []
    start_end_segment = (config.start_second, config.end_second)

    # check if there are any ES signals
    if len(es_signals) == 0:
        no_cp_confirm1 = True

    # if there are ES signals, detect CPs using UCP Detector
    if no_cp_confirm1 is False:
        all_peaks_track, all_scores_track, all_start_end_offset_track = detect_CP_tracks(es_signals, all_start_end_offset_track)
        if len(all_peaks_track) == 0:
            no_cp_confirm2 = True

    # if there are no CPs detected, skip post-processing and aggregation
    if no_cp_confirm1 is False and no_cp_confirm2 is False:
        # create Softmax object for post-processing
        softmax = torch.nn.Softmax(dim=1)

        # refine peak indices using start-end offset tracks
        all_refined_peaks_track = []
        all_scores_pick_softmax_track = []
        for each_peak_track, each_start_end_offset_track, each_score_track in zip(all_peaks_track, all_start_end_offset_track, all_scores_track):
            start_idx_track = each_start_end_offset_track[0]
            all_refined_peaks_track.append(each_peak_track + start_idx_track)

            # also pick out score scalar value for specific change point location of that track
            score_pick_track = []
            for each_cp_pos in each_peak_track:
                score_pick_track.append(each_score_track[each_cp_pos])

            # apply Softmax to score_pick_track
            softmax_score = softmax(torch.Tensor(np.array([score_pick_track])))
            all_scores_pick_softmax_track.append(softmax_score[0].tolist())
        
        # convert refined peak indices to binary matrix
        binary_cp_matrix = np.zeros((len(all_refined_peaks_track), config.video_num_frames))
        score_cp_matrix = np.zeros((len(all_refined_peaks_track), config.video_num_frames))

        # num track x num frames
        for idx_track, each_track in enumerate(all_refined_peaks_track):
            for i, each_cp_index in enumerate(each_track):
                try:
                    binary_cp_matrix[idx_track][each_cp_index] = 1
                except:
                    continue
                score_cp_matrix[idx_track][each_cp_index] = all_scores_pick_softmax_track[idx_track][i]
        
        
        # Video Segmentator
        # Create a uniform segmentator to divide the video into equal-sized segments.
        segmentator = UniformSegmentator()

        # Calculate the segment boundaries based on the total number of segments to extract (config.network.num_intervals) 
        # and the length of the video in frames (config.video_num_frames)
        res_segment_ids = UniformSegmentator.execute(config.network.num_intervals, config.video_num_frames)

        # Change Point Aggregator
        # Create a simple aggregator to combine the change points within each segment into a final set of change points.
        aggregator = SimpleAggregator(config)

        # Execute the aggregator on the segment boundaries and the binary and score change point matrices to produce a 
        # final set of change points (res_cp), their associated scores (res_score), and statistics about the number of 
        # change points found in each segment (stat_total_cp_interval).
        res_cp, res_score, stat_total_cp_interval = aggregator.execute(res_segment_ids, binary_cp_matrix, score_cp_matrix, config.network.max_cp_found)

        # Shift the final set of change points (res_cp) to align with the original video time indexes.
        for idx in range(len(res_cp)):
            res_cp[idx] += config.start_second

        # Add the results to the final output lists for this video.
        final_res_cp.append(res_cp)
        final_res_score.append(res_score)
        final_la.append(la)
        final_res_stat.append(res_stat)
        final_start_end_segment.append(start_end_segment)


    ################################
    time_processing = datetime.now() - start
    
    result = {"final_cp_result": final_res_cp, 
                "final_cp_llr": final_res_score,
                "type": "video", 
                "input_path": config.single_video_path,
                "total_video_frame": video_num_frames, 
                "num_frame_skip": config.skip_frame,
                'time_processing': int(time_processing.total_seconds()),
                "fps": int(video_fps), 
                "individual_cp_result": final_la,
                "stat_segment_seconds_total_cp_accum": final_res_stat
                }

    # save cp result
    with open(config.single_output_path, 'w') as fp:
        json.dump(result, fp, indent=4)