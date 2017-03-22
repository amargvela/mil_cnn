import os
import numpy as np
import json

import scipy.ndimage
from tensorbase.base import Data


class CancerData(Data):
    def __init__(self, flags):
        self.flags = flags
        self.size = flags['image_dim']
        self.path = flags['path_to_image_directory']
        with open(flags['path_to_metadata'], 'r') as f:
            self.metadata = json.load(f)
        super().__init__(flags)

    def _get_image(self, image_path):
        full_path = image_path
        im = scipy.ndimage.imread(full_path, mode='L')
        if self.size != im.shape[0] or  self.size != im.shape[1]:
            im = scipy.misc.imresize(im, size=(self.size, self.size))
        #print(max(im.max(axis=1)))
        #print(min(im.min(axis=1)))
        im = im.reshape((self.size, self.size, 1))
        return im

    def _get_label(self, row):
        birad_score = row[u"ASSESS"]
        birad_pertains_to_image = row[u"is_img_report_relevant"]
        if birad_pertains_to_image == False:
            birad_score = 1
        if birad_score == None:
            return None
        return int(birad_score > self.flags['class_threshold'] or birad_score == 0)

    def get_data(self, meta):
        X = []
        Y = []
        for i in range(len(meta)):
            #if i % 50 == 0:
            #    print("50 images processed")
            row = meta[i]
            image_path = self.path + row['image_path']
            if not os.path.exists(image_path):
                print("File %s does not exist, skipping this row.", row['image_path'])
                continue

            X.append(self._get_image(image_path))

            label = self._get_label(row)
            one_hot = [0 for i in range(self.flags['num_classes'])]
            one_hot[label] = 1
            Y.append(one_hot)

        #print(len(X), len(X[0][0]), len(X[0]))
        return np.array(X), np.array(Y)


    def load_data(self, test_percent):
        train_meta = [row for row in self.metadata
                      if row.get("split_group", None) == 'train']
        train_meta = [row for row in train_meta
                      if os.path.exists(self.path + row['image_path'])]
        test_meta = [row for row in self.metadata
                     if row.get("split_group", None) == 'test']
        test_meta = [row for row in test_meta
                     if os.path.exists(self.path + row['image_path'])]
        print("Data sizes :", len(train_meta), len(test_meta))

        train_X = train_meta
        train_Y = train_meta
        test_X = test_meta
        test_Y = test_meta

        #data = [train_X, train_Y, test_X, test_Y]
        #print(data)
        #with open("images.json", 'w') as f:
            #json.dump(data, f)

        return np.array(train_X), np.array(train_Y), np.array(test_X), np.array(test_Y)

