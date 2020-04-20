import torch, os, sys
from torch.utils.data import Dataset, DataLoader
from Utils.niiparsing import load_nii
import torch.nn.functional as F
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

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

        for i, filename in enumerate(glob("../organized_data/HGG/*flair.nii.gz")):
            
            if i == 5:
                break
            
            # sys.stdout.write("\rFile: %d" % (i + 1))
            # sys.stdout.flush()
            
            split = filename.split('flair')
            self.extract_features(filename, split[0] + 'seg' + split[1])

        self.x = np.vstack(self.x)
        print(self.x.shape)
        self.x = np.array(self.x).reshape(-1, 6)
        self.y = np.array(self.y).reshape(-1, 1)

    """
    Uses 3x3x3 sliding window and 0 padding to go through brain images and extract features using all 27
    pixels. These features will be the features for only the central pixel.
    """
    def extract_features(self, x_path, y_path):

        # Loading and normalizing whole brain images
        x_arr = load_nii(x_path)
        x_arr = x_arr / x_arr.max()
        y_arr = load_nii(y_path)
        
        windows = []
        
        num_c = 0
        num_nc = 0

        for depth in range(0, x_arr.shape[2] - 2, 3):

            # Parameterizing sliding window
            for row in range(0, x_arr.shape[0], 3):
                for col in range(0, x_arr.shape[1], 3):
                    
                    if x_arr[row + 1, col + 1, depth + 1] == 0:
                        num_c += 1
                        continue
                    else:
                        num_nc += 1

                    # Creating sliding window across slice
                    x_window = x_arr[row : row + 3, col : col + 3, depth : depth + 3]
                    y_window = y_arr[row + 1, 
                                     col + 1, 
                                     depth + 1]

                    # Flattening window
                    windows.append(np.array(x_window).reshape(-1))         

                    # Extracting label for central pixel
                    if y_window == 0:
                        self.y.append([0])
                    else:
                        self.y.append([1])

        windows = np.array(windows).reshape(-1, 27)
        
        inp = np.concatenate((
                             np.mean(windows, axis=1).reshape(-1, 1), 
                             np.std(windows, axis=1).reshape(-1, 1), 
                             np.var(windows, axis=1).reshape(-1, 1),
                             np.max(windows, axis=1).reshape(-1, 1),
                             np.min(windows, axis=1).reshape(-1, 1),
                             np.median(windows, axis=1).reshape(-1, 1)), axis=1)
        
        self.x.append(inp)

            
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
