
from torch.utils.data import Dataset
import torch
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from ..transform import transform_entry
import torch.nn.functional as F
import math


class MORPH2Dataset_V2(Dataset):
    def __init__(self, data_path, augmentation, train, **kwargs):

        self.data_path = data_path
        self.protocol = "RANDOM_80_20"
        self.train = train

        self.image_paths = []
        self.labels = []

        self.class_labels = []

        if not self.train:
            augmentation.type="attribute_test_transform"
        self.transform = transform_entry(augmentation)

        self.split = "train" if self.train else "test"

        self.MORPH_data = os.path.join(self.data_path, "data")
        self.MORPH_label = os.path.join(self.data_path, "label", self.protocol, (self.split + ".txt"))

        with open(self.MORPH_label, "r") as f:
            data = f.readlines()

            for i in range(0, len(data)):
                line = data[i].split()
                if len(line) > 0:
                    image_path = os.path.join(self.MORPH_data, line[0])

                    self.image_paths.append(image_path)
                    self.labels.append(int(line[1]))
  

    def __getitem__(self, idx):

        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        img = torch.tensor(np.asarray(img))
        age = self.labels[idx]

        dis = [normal_sampling(age, i) for i in range(101)]
        dis = torch.Tensor(dis)
        dis = F.normalize(dis, p=1, dim=0)


        return {'image':img, 'label':{"avg_label":age, "distribution":dis}, 'filename':img_path}

    def __len__(self):
        return len(self.image_paths)
    
    def __repr__(self):
        return self.__class__.__name__ + \
               f'\nsplit: {self.split}\ndataset_len: {len(self.image_paths)}\naugmentation: {self.transform}'


class UTKFaceDataset_V2(Dataset):
    def __init__(self, data_path, augmentation, train, **kwargs):

        self.data_path = data_path
        self.train = train

        self.image_paths = []
        self.labels = []

        self.class_labels = []

        if not self.train:
            augmentation.type="attribute_test_transform"
        self.transform = transform_entry(augmentation)

        self.split = "train" if self.train else "test"

        self.UTKFace_data = os.path.join(self.data_path, "data")
        self.UTKFace_label = os.path.join(self.data_path, "label", (self.split + ".csv"))

        with open(self.UTKFace_label, "r") as f:
            data = f.readlines()

            for i in range(1, len(data)):
                line = data[i].split(",")
                if len(line) > 0:
                    image_path = os.path.join(self.UTKFace_data, line[1])

                    self.image_paths.append(image_path)
                    self.labels.append(int(line[2]))
  

    def __getitem__(self, idx):

        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        img = torch.tensor(np.asarray(img))

        age = self.labels[idx]
        
        dis = [normal_sampling(age, i) for i in range(101)]
        dis = torch.Tensor(dis)
        dis = F.normalize(dis, p=1, dim=0)     

        return {'image':img, 'label':{"avg_label":age, "distribution":dis}, 'filename':img_path}

    def __len__(self):
        return len(self.image_paths)
    
    def __repr__(self):
        return self.__class__.__name__ + \
               f'\nsplit: {self.split}\ndataset_len: {len(self.image_paths)}\naugmentation: {self.transform}'
    

def normal_sampling(mean, label_k, std=2):
    return math.exp(-(label_k-mean)**2/(2*std**2))/(math.sqrt(2*math.pi)*std)
 