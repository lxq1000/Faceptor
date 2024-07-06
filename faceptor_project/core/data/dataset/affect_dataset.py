
from torch.utils.data import Dataset
import torch
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from ..transform import transform_entry



# Ensure that the facial expression labels of all datasets are consistent with AffectNet.

# 0 Neutral
# 1 Happy
# 2 Sad
# 3 Surprise
# 4 Fear
# 5 Disgust
# 6 Anger
# 7 Contempt



# 7 classes
class AffectNetDataset_V2(Dataset):
    def __init__(self, data_path, augmentation, train, **kwargs):

        self.image_names = []
        self.labels = []
        self.valence_labels = []
        self.arousal_labels = []

        self.train = train
        self.split = "train" if self.train else "test"

        self.final_data_path = os.path.join(data_path, "data")
        self.label_path = os.path.join(data_path, "label", self.split+".txt")

        if not self.train:
            augmentation.type="attribute_test_transform"

        self.transform = transform_entry(augmentation)

        with open(self.label_path, "r") as f:
                data = f.readlines()

        for i in range(0, len(data)):
            line = data[i].strip('\n').split()

            if int(line[1]) < 7:
                self.image_names.append(line[0])
                self.labels.append(int(line[1]))



    def __getitem__(self, idx):

        img_path = os.path.join(self.final_data_path, self.image_names[idx])
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        img = torch.tensor(np.asarray(img))
        expression_label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return {'image':img, 
                'label': expression_label,
                'filename':self.image_names[idx]}
        
    def __len__(self):
        return len(self.image_names)
    
    def __repr__(self):
        return self.__class__.__name__ + \
               f'\nsplit: {self.split}\ndataset_len: {len(self.image_names)}\naugmentation: {self.transform}'
    

class RAFDBDataset(Dataset):
    def __init__(self, data_path, augmentation, train, **kwargs):

        self.standard_label = [-1, 3, 4, 5, 1, 2, 6, 0]

        self.image_names = []
        self.labels = []

        self.final_data_path = os.path.join(data_path, "data")
        self.label_path = os.path.join(data_path, "label.txt")

        self.train = train

        self.split = "train" if self.train else "test"

        if not self.train:
            augmentation.type="attribute_test_transform"

        self.transform = transform_entry(augmentation)

        with open(self.label_path, "r") as f:
                data = f.readlines()

        for i in range(0, len(data)):
            line = data[i].strip('\n').split(" ")

            image_name = line[0]
            sample_temp = image_name.split("_")[0]

            if not self.train and sample_temp == "test":
                self.image_names.append(image_name)
                self.labels.append(self.standard_label[int(line[1])])

            if self.train and sample_temp == "train":
                self.image_names.append(image_name)
                self.labels.append(self.standard_label[int(line[1])])


    def __getitem__(self, idx):

        img_path = os.path.join(self.final_data_path, self.image_names[idx])
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        img = torch.tensor(np.asarray(img))
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return {'image':img, 'label':label, 'filename':self.image_names[idx]}

    def __len__(self):
        return len(self.image_names)
    
    def __repr__(self):
        return self.__class__.__name__ + \
               f'\nsplit: {self.split}\ndataset_len: {len(self.image_names)}\naugmentation: {self.transform}'
    

class FERPlusDataset_V2(Dataset):
    def __init__(self, data_path, augmentation, train, **kwargs):

        self.image_paths = []
        self.labels = []

        self.standard_label = [0, 1, 3, 2, 6, 5, 4, 7]

        self.train = train
        self.split = "train_val" if self.train else "test"

        self.final_data_path = os.path.join(data_path, "data")
        

        if not self.train:
            augmentation.type="attribute_test_transform"

        self.transform = transform_entry(augmentation)



        if self.split == "train_val":

            self.label_path = os.path.join(data_path, "label", "train.txt")
            with open(self.label_path, "r") as f:
                data = f.readlines()

            for i in range(0, len(data)):
                line = data[i].strip('\n').split()

                temp_path = os.path.join(self.final_data_path, line[0])
                self.image_paths.append(temp_path)
                self.labels.append(self.standard_label[int(line[1])])
            
            self.label_path = os.path.join(data_path, "label", "val.txt")
            with open(self.label_path, "r") as f:
                data = f.readlines()

            for i in range(0, len(data)):
                line = data[i].strip('\n').split()

                temp_path = os.path.join(self.final_data_path, line[0])
                self.image_paths.append(temp_path)
                self.labels.append(self.standard_label[int(line[1])])


        elif self.split == "test":
            self.label_path = os.path.join(data_path, "label", "test.txt")
            with open(self.label_path, "r") as f:
                data = f.readlines()

            for i in range(0, len(data)):
                line = data[i].strip('\n').split()

                temp_path = os.path.join(self.final_data_path, line[0])
                self.image_paths.append(temp_path)
                self.labels.append(self.standard_label[int(line[1])])


    def __getitem__(self, idx):

        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        img = torch.tensor(np.asarray(img))
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return {'image': img, 
                'label': label, 
                'filename':self.image_paths[idx]}

    def __len__(self):
        return len(self.image_paths)
    
    def __repr__(self):
        return self.__class__.__name__ + \
               f'\nsplit: {self.split}\ndataset_len: {len(self.image_paths)}\naugmentation: {self.transform}'
    
