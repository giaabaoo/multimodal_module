cd ../../
python unsupervised/preprocessors/prepare_minieval_data.py \
--video_folder /home/dhgbao/Research_Monash/dataset/ccu-data/CCU_COMBINED_ANNOTATED_VIDEO/all_videos \
--audio_folder /home/dhgbao/Research_Monash/dataset/ccu-data/CCU_COMBINED_ANNOTATED_VIDEO/segmented_videos_audio \
--segments_tab_file /home/dhgbao/Research_Monash/code/my_code/unsupervised_approach/multimodal_module/data/mini_eval/segments.tab \
--all_segments_file /home/dhgbao/Research_Monash/dataset/ccu-data/CCU_COMBINED_ANNOTATED_VIDEO/csv/all_segments.csv \
--changepoints_file /home/dhgbao/Research_Monash/code/my_code/unsupervised_approach/multimodal_module/data/changepoints.csv