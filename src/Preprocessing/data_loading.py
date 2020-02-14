import torch, os
from torch.utils.data import Dataset, DataLoader
from Utils.niiparsing import load_nii
import torch.nn.functional as F
from glob import glob

"""
Represents the entire brats dataset that is downloaded. Is manipulated by the 
torch DataLoader to extract batches of data.
"""
class BratsDataset(Dataset):

    """
    Constructor that loads paths to all data.
    """
    def __init__(self):

        self.data_paths = []

        for filename in glob(os.getcwd() + "/organized_data/*/*"):
            self.data_paths.append(filename)

    """
    Number of samples that have been downloaded.
    """
    def __len__(self):

        return len(self.data_paths)

    """
    Gets a single sample from the dataset. Returns dictionary of metadata about the
    sample for later processing. Image itself is returned as a tensor.
    """
    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Loading image
        image = torch.tensor(load_nii(self.data_paths[idx]))

        return {

            # TODO normalization
            'image' : image,
            'idx' : idx

        }

"""
Creates a torch DataLoader that loads data into memory more efficiently and handles batching.
"""
def get_dataloader(batch_size=64, shuffle=True, num_workers=4):

    dataset = BratsDataset()
    print("Dataset has been created")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
