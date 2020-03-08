import os
import json
import sys

from tqdm import tqdm

DATA_ROOT = '/media/vtouchinc02/database/RawData/deepfake-faces-retinaface-json-backup'

if __name__ == '__main__':
    for i in range(50):
        folder_name = 'dfdc_train_part_%d' % i
        print(folder_name)
        metadata_path = os.path.join(DATA_ROOT, folder_name + '/metadata.json')
        with open(metadata_path) as metadata_fp:
            metadata = json.load(metadata_fp)

        output_folder = os.path.join(OUTPUT_ROOT, folder_name)
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)
        
        for video_name, attributes in tqdm(metadata.items()):
            # print(video_name)
            video_path = os.path.join(DATA_ROOT, folder_name + '/' + video_name)    

            if attributes['label'] == 'REAL':
                if not 'face' in attributes:
                    print("No face found: %s" % (video_name))
                    continue

                num_sampled = len(attributes['face'])
                # print(num_sampled)
                if num_sampled != 32:
                    print('# sampled: %d' % num_sampled)
                    # break
                    