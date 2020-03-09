import os
import json
import sys
import cv2
import numpy as np

from tqdm import tqdm

VIDEO_ROOT = '/media/vtouchinc02/database/RawData/deepfake'
TAG_ROOT = '/media/vtouchinc02/database/RawData/deepfake-faces-retinaface-json-backup-clean-fake-results'
OUTPUT_ROOT = '/media/vtouchinc02/database/RawData/deepfake-faces-retinaface-json-backup-clean-fake-results-images'

def get_frames(video_path, frame_idxs):
    capture = cv2.VideoCapture(video_path)

    frames = []
    idxs_read = []
    for frame_idx in range(frame_idxs[0], frame_idxs[-1] + 1):
        ret = capture.grab()
        if not ret:
            print("Error grabbing frame %d from movie %s" % (frame_idx, video_path))
            break
        current = len(idxs_read)
        if frame_idx == frame_idxs[current]:
            ret, frame = capture.retrieve()
            if not ret or frame is None:
                print("Error retrieving frame %d from movie %s" % (frame_idx, video_path))
                break
            frames.append(frame)
            idxs_read.append(frame_idx)

    return idxs_read, frames


if __name__ == '__main__':
    if not os.path.isdir(OUTPUT_ROOT):
        os.mkdir(OUTPUT_ROOT)

    for i in range(50):
        folder_name = 'dfdc_train_part_%d' % i
        print(folder_name)

        output_folder = os.path.join(OUTPUT_ROOT, folder_name)
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)

        metadata_path = os.path.join(TAG_ROOT, folder_name + '/metadata.json')
        with open(metadata_path) as metadata_fp:
            metadata = json.load(metadata_fp)

        for video_name, attributes in tqdm(metadata.items()):
            # print(video_name)
            video_path = os.path.join(VIDEO_ROOT, folder_name + '/' + video_name)

            # Select frames
            dict_frame_face = {}
            if attributes['label'] == 'REAL':
                if not ('face' in attributes):
                    print('No face in %s' % video_name)
                    continue
                dict_frame_face = attributes['face']
            else:
                original_video_name = attributes['original']
                original_attributes = metadata[original_video_name]
                if not ('face' in original_attributes):
                    print('No face in %s, %s' % (video_name, original_video_name))
                    continue
                dict_frame_face = original_attributes['face']

            if len(dict_frame_face) == 0:
                print('Zero face in %s' % video_name)

            tagged_frame_indices = []
            for frame_index in dict_frame_face.keys():
                tagged_frame_indices.append(int(frame_index))

            frame_indices, frames = get_frames(video_path, tagged_frame_indices)

            for frame_index, frame in zip(frame_indices, frames):
                for face_index, detection_result in enumerate(dict_frame_face[str(frame_index)]):
                    box = np.array(detection_result['box'])
                    box = box.astype(int)
                    face = frame[box[1]:box[3], box[0]:box[2]]
                    if face.size == 0:
                        continue
                    path = os.path.join(output_folder, '%s-%d-%d.png' % (video_name[:-4], frame_index, face_index))
                    cv2.imwrite(path, face)

            output_metadata_path = os.path.join(output_folder, 'metadata.json')
            with open(output_metadata_path, "w") as fp:
                json.dump(metadata, fp)
