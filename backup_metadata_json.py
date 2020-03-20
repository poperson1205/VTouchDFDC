import os
import json
import sys

from tqdm import tqdm

DATA_ROOT = '/media/vtouchinc02/database/RawData/deepfake-32frame'
OUTPUT_ROOT = '/media/vtouchinc02/database/RawData/deepfake-32frame-json-backup'

if __name__ == '__main__':
    if not os.path.isdir(OUTPUT_ROOT):
        os.mkdir(OUTPUT_ROOT)

    for i in range(50):
        folder_name = 'dfdc_train_part_%d' % i
        print(folder_name)
        metadata_path = os.path.join(DATA_ROOT, folder_name + '/metadata.json')
        with open(metadata_path) as metadata_fp:
            metadata = json.load(metadata_fp)

        output_folder = os.path.join(OUTPUT_ROOT, folder_name)
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)

        output_metadata_path = os.path.join(output_folder, 'metadata.json')
        with open(output_metadata_path, "w") as fp:
            json.dump(metadata, fp)
