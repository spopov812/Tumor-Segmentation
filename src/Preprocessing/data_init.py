from glob import glob
import os, sys
import gzip
import shutil

from Utils.cloud import CController

# Will be initialized to a CController object
cloud = None

organized_dir_path = "organized_data/"
hgg_dir_path = organized_dir_path + "HGG"
lgg_dir_path = organized_dir_path + "LGG"

"""
Only method that should be imported from this python file. Handles creation of necessary
directories for the project, downloading information about the data, and the data itself.

Data will be stored as
/organized_data/[tumor_type]/[imagename].nii
"""
def organized_data_download(key_path, bucket_name):

    global cloud 

    cloud = CController(key_path, bucket_name)

    __make_initial_dirs()
    __download_images()

"""
Creates directory /organized_data which will contain data downloaded from cloud after
going through organization.
"""
def __make_initial_dirs():

    if not os.path.exists(organized_dir_path):

        print("[INFO] Creating folder %s" % organized_dir_path)
        os.makedirs(organized_dir_path)

    if not os.path.exists(hgg_dir_path):

        print("[INFO] Creating folder %s" % hgg_dir_path)
        os.makedirs(hgg_dir_path)

    if not os.path.exists(lgg_dir_path):

        print("[INFO] Creating folder %s" % lgg_dir_path)
        os.makedirs(lgg_dir_path)


"""
Downloads all images from cloud bucket and places them into appropriate patient folder.
If images have been already downloaded, will not redownload them. If necessary, creates 
folder for modalities under each patient. Adds path to each file to data_info.csv.
"""
def __download_images():

    # Iterating over all files in cloud bucket
    for image_number, file in enumerate(cloud.get_blob_iterator()):

            category = file.name.split("/")[0] + "/"
            filename = file.name.split("/")[-1]

            # If image has already been downloaded, don't redownload it
            if os.path.exists(organized_dir_path + category + filename):
                continue

            # Logging
            sys.stdout.write("\r[INFO] Downloading file %d" % (image_number + 1))
            sys.stdout.flush()

            # Downloading image from bucket to appropriate directory
            cloud.download_blob(file.name, organized_dir_path + category + filename)

            """
            with gzip.open(organized_dir_path + download_filename, 'rb') as f_in:
                with open(organized_dir_path + target_filename, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            """

    print()
