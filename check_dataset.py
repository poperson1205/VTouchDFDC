import os
import json
import sys

from tqdm import tqdm

DATA_ROOT = '/media/vtouchinc02/database/RawData/deepfake'
OUTPUT_ROOT = '/media/vtouchinc02/database/RawData/deepfake-faces-retinaface'

if __name__ == '__main__':
    if not os.path.isdir(OUTPUT_ROOT):
        os.mkdir(OUTPUT_ROOT)

    for folder_name in tqdm(os.listdir(DATA_ROOT)):
        print(folder_name)
        metadata_path = os.path.join(DATA_ROOT, folder_name + '/metadata.json')
        with open(metadata_path) as metadata_fp:
            metadata = json.load(metadata_fp)

        output_folder = os.path.join(OUTPUT_ROOT, folder_name)
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)
        
        for video_name, attributes in tqdm(metadata.items()):
            print(video_name)
            video_path = os.path.join(DATA_ROOT, folder_name + '/' + video_name)
            if not os.path.isfile(video_path):
                print('Failed to find video')
                continue

            if attributes['label'] == 'FAKE':
                if attributes['original'] is None:
                    print('Original not specified')
                else:
                    original_video_name = attributes['original']
                    original_video_path = os.path.join(DATA_ROOT, folder_name + '/' + video_name)
                    if not os.path.isfile(original_video_path):
                        print('Original not found')