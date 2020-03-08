import os
import json
import sys

from tqdm import tqdm

INPUT_ROOT = '/media/vtouchinc02/database/RawData/deepfake-faces-retinaface-json-backup'
OUTPUT_ROOT = '/media/vtouchinc02/database/RawData/deepfake-faces-retinaface-json-backup-clean-fake-results'

if __name__ == '__main__':
    if not os.path.isdir(OUTPUT_ROOT):
        os.mkdir(OUTPUT_ROOT)

    for i in range(50):
        folder_name = 'dfdc_train_part_%d' % i
        print(folder_name)
        metadata_path = os.path.join(INPUT_ROOT, folder_name + '/metadata.json')
        with open(metadata_path) as metadata_fp:
            metadata = json.load(metadata_fp)

        # Clean 'face' in FAKE videos
        for video_name, attributes in tqdm(metadata.items()):
            if attributes['label'] == 'REAL':
                continue

            if 'face' in attributes:
                del attributes['face']

        output_folder = os.path.join(OUTPUT_ROOT, folder_name)
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)

        output_metadata_path = os.path.join(output_folder, 'metadata.json')
        with open(output_metadata_path, "w") as fp:
            json.dump(metadata, fp)
