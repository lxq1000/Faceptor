import os
import time
import pickle
import random
from easydict import EasyDict as edict
import numpy as np
import torch.utils.data as data
from PIL import Image
from ..transform import transform_entry

import os
import torch


__all__ = ['BiAttrDataset']

class BiAttrDataset(data.Dataset):

    def __init__(self, data_path, augmentation, dataset_name, split, **kwargs):

        assert dataset_name in ['CelebA', 'LFWA', 'FaceMask'], \
            f'dataset name {dataset_name} is not exist'

        self.data_path = data_path
        self.split = split

        self.label_path = os.path.join(self.data_path, "label.pkl")
        
        if "train" not in split:
            augmentation.type="attribute_test_transform"
        self.transform = transform_entry(augmentation)
        
        
        with open(self.label_path, "rb+") as f:
            dataset_info = pickle.load(f)
        dataset_info = edict(dataset_info)

        image_names = dataset_info.image_name
        labels = dataset_info.attr

        
        assert split in dataset_info.partition.keys(), f'split {split} is not exist'

        self.attr_names = dataset_info.attr_name
        self.attr_num = len(self.attr_names)

        self.final_idx = dataset_info.partition[split]
        self.final_idx = np.array(self.final_idx)
        
        self.final_image_names = [image_names[i] for i in self.final_idx]
        self.final_labels = [labels[i] for i in self.final_idx]


    def __getitem__(self, index):
        img_name, label = self.final_image_names[index], self.final_labels[index]
        img_path = os.path.join(self.data_path, "data", img_name)

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        img = torch.tensor(np.asarray(img))
        label = np.array(label).astype(np.int)

        output = {'image': img, 'label': label, 'filename': img_name}
        return output

    def __len__(self):
        return len(self.final_image_names)

    def __repr__(self):
        return self.__class__.__name__ + \
               f'\nsplit: {self.split}\ndataset_len: {len(self.final_image_names)}\naugmentation: {self.transform}' \
               f'\nattr_num: {self.attr_num}\nattr_names: {self.attr_names}'

