import pandas as pd
import json
import os
import math

test_dataset_dir = '/media/vtouchinc02/database/RawData/deepfake/dfdc_train_part_48'

if __name__ == '__main__':
    # Ground truth
    tag_json = json.load(open(os.path.join(test_dataset_dir, 'metadata.json')))
    
    df = pd.read_csv('submission.csv')
    loss = 0
    count = 0
    count_fake = 0
    count_fake_gt = 0
    for index, row in df.iterrows():
        filename, label = row['filename'], row['label']
        gt_label = 1.0 if tag_json[filename]['label'] == 'FAKE' else 0.0
        
        # y_hat = 0.5
        y_hat = float(label)
        y = float(gt_label)
        if y > 0.5:
            count_fake_gt += 1
        if y_hat >= 0.5:
            count_fake += 1
        loss += y*math.log(y_hat) + (1.0-y)*math.log(1.0-y_hat)
        count += 1
    print('Total: {0}, Fake(Predict): {1}, Fake(GT): {2}'.format(count, count_fake, count_fake_gt))
    loss /= float(count)
    loss *= -1.0
    print(loss)

        