import nibabel as nib
import numpy as np

def load_nii(path):

    img = nib.load(path)
    data = img.get_fdata()

    return data
