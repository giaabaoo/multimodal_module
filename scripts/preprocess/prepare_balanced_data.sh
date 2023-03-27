cd ../../
python unsupervised/preprocessors/prepare_balanced_data.py \
--video_folder /home/dhgbao/Research_Monash/dataset/ccu-data/CCU_COMBINED_ANNOTATED_VIDEO/all_videos \
--audio_folder /home/dhgbao/Research_Monash/dataset/ccu-data/CCU_COMBINED_ANNOTATED_VIDEO/segmented_videos_audio \
--all_segments_file /home/dhgbao/Research_Monash/dataset/ccu-data/CCU_COMBINED_ANNOTATED_VIDEO/csv/all_segments.csv \
--changepoints_file /home/dhgbao/Research_Monash/dataset/ccu-data/CCU_COMBINED_ANNOTATED_VIDEO/csv/changepoints_preprocessed.csv