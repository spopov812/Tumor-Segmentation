import torch, os
from torch.utils.data import Dataset, DataLoader
from Utils.niiparsing import load_nii
import torch.nn.functional as F
from glob import glob
import numpy as np

"""
Represents the entire brats dataset that is downloaded. Is manipulated by the 
torch DataLoader to extract batches of data.
"""
class BratsDataset(Dataset):

    """
    Constructor that loads paths to all data.
    """
    def __init__(self):

        self.x = []
        self.y = []

        for filename in glob("C:/Users/Jason/GitHub/Tumor-Segmentation/organized_data/*/*flair.nii.gz"):
            split = filename.split('flair')
            self.extract_features(filename, split[0] + 'seg' + split[1])

        print(len(self.x))
        print(len(self.y))

    """
    Uses 3x3 sliding window and 0 padding to go through brain images and extract features using all 9
    pixels. These features will be the features for only the central pixel.
    """
    def extract_features(self, x_path, y_path):

        # Loading and normalizing whole brain images
        x = load_nii(x_path)
        x = x / np.linalg.norm(x)
        y = load_nii(y_path)

        # Padding
        x_arr = np.pad(x, ((1, 1), (1, 1), (0, 0)), mode='constant')
        y_arr = np.pad(y, ((1, 1), (1, 1), (0, 0)), mode='constant')

        assert x_arr.shape[0] == x_arr.shape[1]

        # For each 2D slice
        for depth in range(x_arr.shape[2]):

            # Parameterizing sliding window
            for height in range(x_arr.shape[0] - 2):
                for width in range(x_arr.shape[1] - 2):

                    feature_vec = []

                    # Creating sliding window across slice
                    x_window = x_arr[height : height + 3, width : width + 3, depth]
                    y_window = y_arr[height + 1, width + 1, depth]
                
                    # Feature extraction
                    feature_vec.append(np.mean(x_window))
                    feature_vec.append(np.std(x_window))
                    feature_vec.append(np.var(x_window))

                    self.x.append(feature_vec)

                    if y_window == 0:
                        self.y.append([0])
                    else:
                        self.y.append([1])
            
    """
    Number of samples that have been downloaded.
    """
    def __len__(self):

        return len(self.x)

    """
    Gets a single sample from the dataset. Returns dictionary of metadata about the
    sample for later processing. Image itself is returned as a tensor.
    """
    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()


        return {

            'x' : self.x[idx],
            'y' : self.y[idx]

        }

"""
Creates a torch DataLoader that loads data into memory more efficiently and handles batching.
"""
def get_dataloader(batch_size=64, shuffle=True, num_workers=4):

    dataset = BratsDataset()
    print("Dataset has been created")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
