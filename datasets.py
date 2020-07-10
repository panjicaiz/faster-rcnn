'''
dataset loader
'''
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


classes = ['background', 'head']


class ObjectDetectionDataset(Dataset):
    def __init__(self, path='train', label=None):
        self.images_path = os.path.join('data', path, 'images')
        self.images = sorted(os.listdir(self.images_path))
        self.labels_path = os.path.join('data', path, 'labels')
        if not os.path.exists(self.labels_path):
            os.makedirs(self.labels_path)
            if label.split('.')[1] == 'csv':
                self.make_labels_from_csv('data/'+label)
        self.labels = sorted(os.listdir(self.labels_path))
        delete_list = []
        for i, each in enumerate(self.images):
            if each.split('.')[0]+'.txt' not in self.labels:
                delete_list.append(i)
        for i in delete_list[::-1]:
            del self.images[i]
        self.images = sorted(self.images)
            
    def __getitem__(self, idx):
        # assert self.labels[idx].split('.')[0] == self.images[idx].split('.')[0], 'image id is not the same'
        img_path = os.path.join(self.images_path, self.images[idx])
        label_path = os.path.join(self.labels_path, self.images[idx].split('.')[0]+'.txt')
        img = Image.open(img_path)
        img = np.array(img)
        boxes = []
        labels = []
        with open(label_path) as f:
            for each in  f.readlines():
                label, xmin, ymin, xmax, ymax = map(float, each.split())
                labels.append(label)
                boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        img = torch.as_tensor(img, dtype=torch.float32)
        img = img.permute(2, 0, 1)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels

        return img, target

    def __len__(self):
        return len(self.images)

    def make_labels_from_csv(self, label_path):
        labels = pd.read_csv(label_path)
        for i, image_id_suffix in enumerate(self.images):
            image_id = image_id_suffix.split('.')[0]
            image_labels = labels.loc[labels['image_id'] == image_id]
            if len(image_labels) == 0:
                continue
            for j in range(len(image_labels)):
                row = image_labels.iloc[j]
                width = float(row['width'])
                height = float(row['height'])
                bbox = row['bbox']
                bbox = bbox.replace('[', '').replace(']', '')
                xmin, ymin, owidth, oheight = bbox.split(', ')
                xmin, ymin, owidth, oheight = map(float, [xmin, ymin, owidth, oheight])
                x1, y1, x2, y2 = xmin, ymin, xmin+owidth, ymin+oheight
                bbox = map(str, [x1, y1, x2, y2])
                class_ = 'head'
                file_name = os.path.join(self.labels_path, '%s.txt' % image_id)
                with open(file_name, 'a') as f:
                    line = str(classes.index(class_)) + ' ' + ' '.join([s for s in bbox]) + '\n'
                    f.write(line)

