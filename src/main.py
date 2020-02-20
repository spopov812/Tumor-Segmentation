import os

import torch

from Utils.argparsing import get_args
from Preprocessing.data_init import organized_data_download
from Preprocessing.data_loading import BratsDataset

if __name__ == "__main__":

    args = get_args()

    # If data needs to be downloaded
    if args['download']:
        organized_data_download(args['key_path'], args['bucket'])

    data = BratsDataset()
