import os
import shutil
import json
import sys

from tqdm import tqdm

SOURCE_ROOT = '/media/vtouchinc02/database/RawData/deepfake-32frame-clean-non-fake-detections'
TARGET_ROOT = '/media/vtouchinc02/database/RawData/deepfake-32frame'

if __name__ == '__main__':
    if not os.path.isdir(TARGET_ROOT):
        os.mkdir(TARGET_ROOT)

    for i in range(5):
        folder_name = 'dfdc_train_part_%d' % i
        print(folder_name)
        metadata_path = os.path.join(SOURCE_ROOT, folder_name + '/metadata.json')
        if not os.path.isfile(metadata_path):
            print('Failed to find %s' % metadata_path)

        output_folder = os.path.join(TARGET_ROOT, folder_name)
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)
        
        output_metadata_path = os.path.join(output_folder, 'metadata.json')
        shutil.copy(metadata_path, output_metadata_path)