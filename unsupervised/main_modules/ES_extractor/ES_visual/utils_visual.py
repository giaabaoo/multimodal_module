# # preprocessing to optimize the retrieving process, mapping frame index to track (many-many)
#         frameidx_to_track = {}
    
#         for idx_track, each_track in enumerate(all_face_tracks):
#             frames_appear = each_track['frames_appear']
#             for each_frameidx in frames_appear:
#                 if each_frameidx not in frameidx_to_track:
#                     frameidx_to_track[each_frameidx] = []

#                 frameidx_to_track[each_frameidx].append(idx_track)
