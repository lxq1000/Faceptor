import os
import numpy as np
import cv2
import functools
from typing import Dict, List
from ..transform import transform_entry
from ..transform.dense_prediction_transform import to_tensor_and_normalize
from torch.utils.data import Dataset
from torchvision import transforms

@functools.lru_cache()
def _cached_imread(fname, flags=None):
    return cv2.imread(fname, flags=flags)


class CelebAMaskHQDataset(Dataset):
    def __init__(self, data_path, split, augmentation, label_type="all", **kwargs):

        self.data_path = data_path
        self.split = split
        self.hq_names = []

        hq_to_orig_mapping = dict()
        orig_to_hq_mapping = dict()
        mapping_file = os.path.join(self.data_path, 'CelebA-HQ-to-CelebA-mapping.txt')

        for s in open(mapping_file, 'r'):
            if '.jpg' not in s:
                continue
            idx, _, orig_file = s.split()
            hq_to_orig_mapping[int(idx)] = orig_file
            orig_to_hq_mapping[orig_file] = int(idx)

        # load partition
        partition_file = os.path.join(self.data_path, 'list_eval_partition.txt')
            
        for s in open(partition_file, 'r'):
            if '.jpg' not in s:
                continue
            orig_file, group = s.split()
            group = int(group)
            if orig_file not in orig_to_hq_mapping:
                continue
            hq_id = orig_to_hq_mapping[orig_file]
                
            if split == "train" and group == 0:
                self.hq_names.append(str(hq_id))
            elif split == "val" and group == 1:
                self.hq_names.append(str(hq_id))
            elif split == "test" and group == 2:
                self.hq_names.append(str(hq_id))
            elif split == "train_val" and group in [0, 1]:
                self.hq_names.append(str(hq_id))

        self.label_setting = {
            'human': {
                'suffix': [
                    'neck', 'skin', 'cloth', 'l_ear', 'r_ear', 'l_brow', 'r_brow',
                    'l_eye', 'r_eye', 'nose', 'mouth', 'l_lip', 'u_lip', 'hair'
                ],
                'names': [
                    'bg', 'neck', 'face', 'cloth', 'rr', 'lr', 'rb', 'lb', 're',
                    'le', 'nose', 'imouth', 'llip', 'ulip', 'hair'
                ]
            },
            'aux': {
                'suffix': [
                    'eye_g', 'hat', 'ear_r', 'neck_l',
                ],
                'names': [
                    'normal', 'glass', 'hat', 'earr', 'neckl'
                ]
            },
            'all': {
                'suffix': [
                    'neck', 'skin', 'cloth', 'l_ear', 'r_ear', 'l_brow', 'r_brow',
                    'l_eye', 'r_eye', 'nose', 'mouth', 'l_lip', 'u_lip', 'hair',
                    'eye_g', 'hat', 'ear_r', 'neck_l',
                ],
                'names': [
                    'bg', 'neck', 'face', 'cloth', 'rr', 'lr', 'rb', 'lb', 're',
                    'le', 'nose', 'imouth', 'llip', 'ulip', 'hair',
                    'glass', 'hat', 'earr', 'neckl'
                ]
            }
        }[label_type]

        self.transform = transform_entry(augmentation)
        self.to_tensor_and_normalize = to_tensor_and_normalize(**augmentation.kwargs)

    def make_label(self, index, ordered_label_suffix):
        label = np.zeros((512, 512), np.uint8)
        name = self.hq_names[index]
        name_id = int(name)
        name5 = '%05d' % name_id
        p = os.path.join(self.data_path, 'CelebAMask-HQ-mask-anno',
                         str(name_id // 2000), name5)
        for i, label_suffix in enumerate(ordered_label_suffix):
            label_value = i + 1
            label_fname = os.path.join(p + '_' + label_suffix + '.png')
            if os.path.exists(label_fname):
                mask = _cached_imread(label_fname, cv2.IMREAD_GRAYSCALE)
                label = np.where(mask > 0,
                                 np.ones_like(label) * label_value, label)
        return label

    def __getitem__(self, idx):
        name = self.hq_names[idx]

        img = cv2.imread(os.path.join(self.data_path, 'CelebA-HQ-img', name + '.jpg'))[:, :, ::-1]
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)

        label = self.make_label(idx, self.label_setting['suffix'])

        data = {'image': img, 'label': label, 'filename':name}
        data = self.transform.process(data)

        
        data["image"] = self.to_tensor_and_normalize(data["image"])
    
        return data

    def __len__(self):
        return len(self.hq_names)
    

    def __repr__(self):
        return self.__class__.__name__ + \
               f'\nsplit: {self.split}\ndataset_len: {len(self.hq_names)}' \
               f'\nclass_num: {len(self.label_setting["names"])}\nclass_names: {self.label_setting["names"]}'
    

class LaPaDataset(Dataset):
    """LaPa face parsing dataset

    Args:
        root (str): The directory that contains subdirs 'image', 'labels'
    """

    def __init__(self, data_path, split, augmentation, **kwargs):
        assert os.path.isdir(data_path)
        self.data_path = data_path
        self.split = split

        subfolders = []
        if split == "train":
            subfolders = ['train']
        elif split == "val":
            subfolders = ['val']
        elif split == "test":
            subfolders = ['test']
        elif split == "train_val":
            subfolders = ['train', 'val']
        elif split == "toy":
            subfolders = ['toy']

        self.info = []
        for subf in subfolders:
            for name in os.listdir(os.path.join(self.data_path, subf, 'images')):
                if not name.endswith('.jpg'):
                    continue
                name = name.split('.')[0]
                image_path = os.path.join(
                    self.data_path, subf, 'images', f'{name}.jpg')
                label_path = os.path.join(
                    self.data_path, subf, 'labels', f'{name}.png')
                landmark_path = os.path.join(
                    self.data_path, subf, 'landmarks', f'{name}.txt')
                assert os.path.exists(image_path)
                assert os.path.exists(label_path)
                assert os.path.exists(landmark_path)
                landmarks = [float(v) for v in open(
                    landmark_path, 'r').read().split()]
                assert landmarks[0] == 106 and len(landmarks) == 106*2+1
                landmarks = np.reshape(
                    np.array(landmarks[1:], np.float32), [106, 2])
                sample_name = f'{subf}.{name}'
                self.info.append(
                    {'image_path': image_path, 'label_path': label_path,
                     'landmarks': landmarks, 'sample_name': sample_name})
                
        self.label_names = ['background', 'face_lr_rr', 'lb', 'rb', 'le', 're', 'nose', 'ul', 'im', 'll', 'hair']
                
        self.transform = transform_entry(augmentation)
        self.to_tensor_and_normalize = to_tensor_and_normalize(**augmentation.kwargs)

    def __getitem__(self, index):
        info = self.info[index]
        image = cv2.imread(info['image_path'])[:, :, ::-1].copy()
        label = cv2.imread(info['label_path'], cv2.IMREAD_GRAYSCALE)
        landmarks = info['landmarks']

        data = {'image': image, 'label': label, 'landmarks': landmarks, "filename": info['sample_name']}

        data = self.transform.process(data)


        data["image"] = self.to_tensor_and_normalize(data["image"])

        return data

    def __len__(self):
        return len(self.info)
    

    def __repr__(self):
        return self.__class__.__name__ + \
               f'\nsplit: {self.split}\ndataset_len: {len(self.info)}' \
               f'\nclass_num: {len(self.label_names)}\nclass_names: {self.label_names}'
    

    @staticmethod
    def draw_landmarks(im, landmarks, color, thickness=5, eye_radius=3):
        landmarks = landmarks.astype(np.int32)
        cv2.polylines(im, [landmarks[0:33]], False,
                      color, thickness, cv2.LINE_AA)
        cv2.polylines(im, [landmarks[33:42]], True,
                      color, thickness, cv2.LINE_AA)
        cv2.polylines(im, [landmarks[42:51]], True,
                      color, thickness, cv2.LINE_AA)
        cv2.polylines(im, [landmarks[51:55]], False,
                      color, thickness, cv2.LINE_AA)
        cv2.polylines(im, [landmarks[55:66]], False,
                      color, thickness, cv2.LINE_AA)
        cv2.polylines(im, [landmarks[66:74]], True,
                      color, thickness, cv2.LINE_AA)
        cv2.circle(im, (landmarks[74, 0], landmarks[74, 1]),
                   eye_radius, color, thickness, cv2.LINE_AA)
        cv2.polylines(im, [landmarks[75:83]], True,
                      color, thickness, cv2.LINE_AA)
        cv2.circle(im, (landmarks[83, 0], landmarks[83, 1]),
                   eye_radius, color, thickness, cv2.LINE_AA)
        cv2.polylines(im, [landmarks[84:96]], True,
                      color, thickness, cv2.LINE_AA)
        cv2.polylines(im, [landmarks[96:-2]], True,
                      color, thickness, cv2.LINE_AA)
        return im