from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description='Ensemble config')
    parser.add_argument('--config', type=str, required=True)

    return parser.parse_args()