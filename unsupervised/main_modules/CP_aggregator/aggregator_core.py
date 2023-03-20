# Implementation of video segmentation method for finding change point

import numpy as np
import os
import pdb

from dotmap import DotMap

# temporal import
# import sys
# sys.path.append("/home/nttung/research/Monash_CCU/mini_eval/multimodal_module/UCP")
# from inference_ucp import *

# from segment_core import *

class SimpleAggregator():
    def __init__(self, args):
        self.args = args


    def execute(self, segment_ids, binary_matrix, score_matrix, max_cp_found):
        '''
            Input:
                + segment_ids: list of segment interval index
                + binary_matrix: np array representing for index-based change point result
        '''

        print("=============Aggregating to find final impactful change point================")
        # accumulated list
        h = [0]*(len(segment_ids))
        score = [0]*(len(segment_ids))

        for i in range(len(h)):
            h[i] = np.sum(binary_matrix[:, 0:(segment_ids[i])])
            score[i] = np.sum(score_matrix[:, 0:(segment_ids[i])])

        # calculate f(M) based on g
        index_list = []
        total_change_point_list = []
        total_score_list = []

        for i in range(len(h)):
            if i == 0:
                total_change_point_list.append(int(h[0]))
                index_list.append(segment_ids[0])
                total_score_list.append(score[0])
                continue

            g = h[i] - h[i-1]
            sum_score = score[i] - score[i-1]

            total_change_point_list.append(int(g))
            total_score_list.append(float(sum_score/g))#normalize based on total of change point
            index_list.append(segment_ids[i])

        # sort with respect to total_change_point_list
        idx_sort = np.argsort(np.array(total_change_point_list))
        prioritized_index_list = [index_list[a]  for a in idx_sort]
        prioritized_score_list = [total_score_list[a] for a in idx_sort]

        # this index refers to the frame index

        # view as list
        if len(prioritized_index_list) >= max_cp_found:
            max_index_list = prioritized_index_list[-max_cp_found:]
            max_score_list = prioritized_score_list[-max_cp_found:]
        else:
            max_index_list = prioritized_index_list
            max_score_list = prioritized_score_list

        final_res = []
        for res in max_index_list:
            convert_seconds = int(res/self.args.fps)
            final_res.append(convert_seconds)

        # total_change_point_list = sorted(total_change_point_list, reverse=True)
        return final_res, max_score_list, total_change_point_list


# if __name__ == "__main__":
#     args = DotMap()
#     args.num_intervals = 20
#     args.length_video_in_frame = 2001

#     # ES extractor and UCP
#     es_signals = np.load("../ES_extractor/test_es_feature.npy", allow_pickle=True)
#     all_peaks_track, all_scores_track = detect_CP_tracks(es_signals)

#     # transform all_peaks_track into index-based change point matrix
#     binary_cp_matrix = np.zeros((len(all_peaks_track), args.length_video_in_frame))

#     for idx_track, each_track in enumerate(all_peaks_track):
#         for each_cp_index in each_track:
#             binary_cp_matrix[idx_track][each_cp_index] = 1

#     # test segmentator
#     segmentator = UniformSegmentator()
#     res_segment_ids = UniformSegmentator.execute(args.num_intervals, args.length_video_in_frame)


#     # extract final change point
#     aggregator = SimpleAggregator()
#     res_cp = aggregator.execute(res_segment_ids, binary_cp_matrix)
#     pdb.set_trace()


# for i in range(len(h)):
        #     if i == 0:
        #         max_index = segment_ids[i]
        #         total_max_change_point = h[i]
        #         continue

        #     g = h[i] - h[i-1]

        #     if g > total_max_change_point:
        #         max_index = segment_ids[i]
        #         total_max_change_point = g

