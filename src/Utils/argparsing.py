import argparse

"""
Returns args parsed from the command line. Handles optional and required args
as well as if/else relationships between args.
"""
def get_args():

    # Setting up argparser
    parser = argparse.ArgumentParser(description = "Brain tumor segmentation",
    add_help = "How to use")

    # Optional args
    parser.add_argument("-d", "--download", action='store_true',
    help = "Whether to download data from cloud bucket. If True, then must provide -k arg. [DEFAULT: False]")

    parser.add_argument("-k", "--key_path", default = None, type = str,
    help = "Path to .json file containing credential to access cloud bucket. Only used if -d is specified. [DEFAULT: None]")

    parser.add_argument("--bucket", default = "brats", type = str,
    help = "Name of cloud bucket containing data. [DEFAULT: brats]")

    parser.add_argument("-b", "--batch_size", default = 8, type = int,
    help = "Batch sizes of data that will be fed to the model. [DEFAULT: 64]")

    parser.add_argument("--train", action='store_true',
    help = "Whether to train the model. [DEFAULT: False]")

    args = vars(parser.parse_args())

    # If user wants to download, but does not provide the path to cloud bucket key file
    if args['download'] and args['key_path'] is None:
        parser.error("--download flag requires --key_path.")

    return args
