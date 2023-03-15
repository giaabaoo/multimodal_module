# Implementation of video segmentation method for finding change point

import numpy as np
import os
import pdb

from dotmap import DotMap

class UniformSegmentator():
    def __init__(self):
        pass


    def execute(num_intervals, l):
        '''
            Return the list of index for each segment
        '''
        sequence_length = int(l / num_intervals)

        if sequence_length == 1:
            return [0, num_intervals, l]
        if sequence_length == 0:
            return [0, l]

        list_index = []
        start = 0
        while True:
            start += sequence_length - 1
            if start >= l:
                list_index.append(l)
                break
            else:
                list_index.append(start)

        return list_index

# if __name__ == "__main__":
#     args = DotMap()
#     args.num_intervals = 20

#     # test segmentator
#     segmentator = UniformSegmentator()
#     res = UniformSegmentator.execute(args.num_intervals, 1133)
#     pdb.set_trace()

