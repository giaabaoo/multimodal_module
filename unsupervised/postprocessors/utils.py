import argparse

def get_args_parser():
    parser = argparse.ArgumentParser("Parsing arguments", add_help=False)
    parser.add_argument("--config", default=None, type=str)
    parser.add_argument("--output_path", default=None, type=str)

    return parser