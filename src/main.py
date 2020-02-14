import os

import torch

from Utils.argparsing import get_args
from Preprocessing.data_init import organized_data_download
from Preprocessing.data_loading import get_dataloader

if __name__ == "__main__":

    args = get_args()

    # If data needs to be downloaded
    if args['download']:
        organized_data_download(args['key_path'], args['bucket'])

    # Converting data to a dataloader
    data = get_dataloader(batch_size=args['batch_size'])

    for batch in data:
        print(batch['image'])
