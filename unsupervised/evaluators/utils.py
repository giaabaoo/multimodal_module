import argparse

def get_args_parser():
    parser = argparse.ArgumentParser("Parsing arguments", add_help=False)
    parser.add_argument("--config", default="./configs/evaluator.yaml", type=str)
    parser.add_argument("--prediction_path", default=None, type=str)
    parser.add_argument("--gt_path", default=None, type=str)
    parser.add_argument("--mode", default='batch', type=str)
    parser.add_argument("--llr_threshold", default=1, type=int)
    
    return parser