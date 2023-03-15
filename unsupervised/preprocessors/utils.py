import argparse

def get_args_parser():
    parser = argparse.ArgumentParser("Parsing arguments", add_help=False)
    parser.add_argument("--config", default=None, type=str)
    
    parser.add_argument("--video_folder", default=None, type=str)
    parser.add_argument("--all_segments_file", default=None, type=str)
    parser.add_argument("--changepoints_file", default=None, type=str)

    return parser