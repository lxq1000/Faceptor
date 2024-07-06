import os
import numpy as np
import cv2
import scipy.io
from hdf5storage import loadmat
import math
from torch.utils.data import Dataset
from ..transform import transform_entry
from ..transform.dense_prediction_transform import to_tensor_and_normalize


class AFLWDataset(Dataset):
    def __init__(self, data_path, split, augmentation, **kwargs):
        self.images_root = os.path.join(data_path, 'data', 'flickr')
        info = scipy.io.loadmat(os.path.join(
            data_path, 'AFLWinfo_release.mat'))
        self.bbox = info['bbox']  # 24386x4 left, right, top bottom
        self.data = info['data']  # 24386x38 x1,x2...,xn,y1,y2...,yn
        self.mask = info['mask_new']  # 24386x19
        self.name_list = [s[0][0] for s in info['nameList']]

        ra = np.reshape(info['ra'].astype(np.int32), [-1])-1
        assert ra.min() == 0
        assert ra.max() == self.bbox.shape[0] - 1

        if split == "train":
            self.indices = ra[:20000]
        elif split == 'test_full':
            self.indices = ra[20000:]
        elif split == 'test_frontal':
            all_visible = np.all(self.mask == 1, axis=1)  # 24386
            self.indices = np.array(
                    [ind for ind in ra[20000:] if all_visible[ind]])
        
        self.split = split

        self.transform = transform_entry(augmentation)
        self.to_tensor_and_normalize = to_tensor_and_normalize(**augmentation.kwargs)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        ind = self.indices[index]
        image_path = os.path.join(
            self.images_root, self.name_list[ind])
        assert os.path.exists(image_path)
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        landmarks = np.reshape(self.data[ind], [2, 19]).transpose()

        left, right, top, bottom = self.bbox[ind]
        box_y1x1y2x2 = np.array([top, left, bottom, right], dtype=np.float32)

        visibility = self.mask[ind]
        data = {
            'image': image,
            'box': box_y1x1y2x2,
            'landmarks': landmarks,
            'visibility': visibility,
            'filename': self.name_list[ind]
        }

        data = self.transform.process(data)
        
        data["image"] = self.to_tensor_and_normalize(data["image"])

        return data
    
        

    def __repr__(self):
        return self.__class__.__name__ + \
               f'\nsplit: {self.split}\ndataset_len: {len(self.indices)}' \
               f'\nlandmark_num: {19}\n'
    

class IBUG300WDataset(Dataset):
    def __init__(self, data_path, split, augmentation, **kwargs):
        self.data_path = data_path
        self.anno = []

        if split == "train":
            anno_file = 'face_landmarks_300w_train.csv'
        elif split == "valid_common":
            anno_file = 'face_landmarks_300w_valid_common.csv'
        elif split == "valid_challenge":
            anno_file = 'face_landmarks_300w_valid_challenge.csv'
        elif split == "valid":
            anno_file = 'face_landmarks_300w_valid.csv'
        elif split == "test":
            anno_file = 'face_landmarks_300w_test.csv'
            
        else:
            raise RuntimeError(f'Unsupported split {split} for IBUG300W')

        error_im_paths = {
            'ibug/image_092_01.jpg': 'ibug/image_092 _01.jpg'
        }

        self.info_list = []
        with open(os.path.join(self.data_path, anno_file), 'r') as fd:
            fd.readline()  # skip the first line
            for line in fd:
                line = line.strip()
                if len(line) == 0:
                    continue
                if line.startswith('#'):
                    continue
                im_path, scale, center_w, center_h, * \
                    landmarks = line.split(',')

                if im_path in error_im_paths:
                    im_path = error_im_paths[im_path]

                sample_name = os.path.splitext(im_path)[0].replace('/', '_')

                im_path = os.path.join(self.data_path, "data", im_path)
                assert os.path.exists(im_path)

                self.info_list.append({
                    'sample_name': sample_name,
                    'im_path': im_path,
                    'landmarks': np.reshape(np.array([float(v)-2.0 for v in landmarks], dtype=np.float32), [68, 2]),
                    'box_info': (float(scale), float(center_w)-2.0, float(center_h)-2.0)
                })
        
        self.split = split

        self.transform = transform_entry(augmentation)
        self.to_tensor_and_normalize = to_tensor_and_normalize(**augmentation.kwargs)

    def __len__(self):
        return len(self.info_list)

    def __getitem__(self, index):
        info = self.info_list[index]
        image = cv2.cvtColor(cv2.imread(info['im_path']), cv2.COLOR_BGR2RGB)
        scale, center_w, center_h = info['box_info']
        box_half_size = 100.0 * scale

        data = {
            'image': image,
            'box': np.array([center_h-box_half_size, center_w-box_half_size,
                             center_h+box_half_size, center_w+box_half_size],
                            dtype=np.float32),
            'landmarks': info['landmarks'],
            'filename': info['sample_name']
        }
    
        data = self.transform.process(data)
        
        data["image"] = self.to_tensor_and_normalize(data["image"])
        return data
        

    def __repr__(self):
        return self.__class__.__name__ + \
               f'\nsplit: {self.split}\ndataset_len: {len(self.info_list)}' \
               f'\nlandmark_num: {68}\n'
    

class WFLWDataset(Dataset):
    def __init__(self, data_path, split:str, augmentation, **kwargs):
        self.data_path = data_path

        anno_file = None
        if split == "train":
            anno_file = 'face_landmarks_wflw_train.csv'
        elif split.startswith("test"):
            anno_file = f'face_landmarks_wflw_{split}.csv'

        self.info_list = []
        with open(os.path.join(self.data_path, anno_file), 'r') as fd:
            fd.readline()  # skip the first line
            for line in fd:
                line = line.strip()
                if len(line) == 0:
                    continue
                if line.startswith('#'):
                    continue
                im_path, scale, center_w, center_h, * \
                    landmarks = line.split(',')

                landmarks = np.reshape(
                    np.array([float(v) for v in landmarks], dtype=np.float32), [98, 2])
                cx, cy = np.mean(landmarks, axis=0)

                sample_name = os.path.splitext(im_path)[0].replace(
                    '/', '.') + ('_%.3f_%.3f' % (cx, cy))
                im_path = os.path.join(self.data_path, 'data', im_path)

                assert os.path.exists(im_path)                

                self.info_list.append({
                    'sample_name': sample_name,
                    'im_path': im_path,
                    'landmarks': landmarks,
                    'box_info': (float(scale), float(center_w), float(center_h))
                })
        self.split = split

        self.transform = transform_entry(augmentation)
        self.to_tensor_and_normalize = to_tensor_and_normalize(**augmentation.kwargs)


    def __len__(self):
        return len(self.info_list)

    def __getitem__(self, index):
        info = self.info_list[index]
        image = cv2.cvtColor(cv2.imread(info['im_path']), cv2.COLOR_BGR2RGB)
        scale, center_w, center_h = info['box_info']
        box_half_size = 100.0 * scale

        data = {
            'image': image,
            'box': np.array([center_h-box_half_size, center_w-box_half_size,
                             center_h+box_half_size, center_w+box_half_size],
                            dtype=np.float32),
            'landmarks': info['landmarks']
        }

        data = self.transform.process(data)
        
        data["image"] = self.to_tensor_and_normalize(data["image"])
        return data
    
    def __repr__(self):
        return self.__class__.__name__ + \
               f'\nsplit: {self.split}\ndataset_len: {len(self.info_list)}' \
               f'\nlandmark_num: {98}\n'


class COFWDataset(Dataset):

    def __init__(self, data_path, split, augmentation, **kwargs):
        # specify annotation file for dataset
        if split == "train":
            self.mat_file = os.path.join(data_path, "COFW_train_color.mat")
        else:
            self.mat_file = os.path.join(data_path, "COFW_test_color.mat")

        self.data_path = data_path


        # load annotations
        self.mat = loadmat(self.mat_file)
        if split == "train":
            self.images = self.mat['IsTr']
            self.pts = self.mat['phisTr']
        else:
            self.images = self.mat['IsT']
            self.pts = self.mat['phisT']

        
        self.split = split

        self.transform = transform_entry(augmentation)
        self.to_tensor_and_normalize = to_tensor_and_normalize(**augmentation.kwargs)


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img = self.images[idx][0]

        if len(img.shape) == 2:
            img = img.reshape(img.shape[0], img.shape[1], 1)
            img = np.repeat(img, 3, axis=2)

        pts = self.pts[idx][0:58].reshape(2, -1).transpose()

        xmin = np.min(pts[:, 0])
        xmax = np.max(pts[:, 0])
        ymin = np.min(pts[:, 1])
        ymax = np.max(pts[:, 1])

        center_w = (math.floor(xmin) + math.ceil(xmax)) / 2.0
        center_h = (math.floor(ymin) + math.ceil(ymax)) / 2.0

        box_half_size = max(math.ceil(xmax) - math.floor(xmin), math.ceil(ymax) - math.floor(ymin))/2

        data = {
            'image': img,
            'box': np.array([center_h-box_half_size, center_w-box_half_size,
                             center_h+box_half_size, center_w+box_half_size],
                            dtype=np.float32),
            'landmarks': pts
        }

        data = self.transform.process(data)

        data["image"] = self.to_tensor_and_normalize(data["image"])
        

        return data
    

    def __repr__(self):
        return self.__class__.__name__ + \
               f'\nsplit: {self.split}\ndataset_len: {len(self.images)}' \
               f'\nlandmark_num: {29}\n'
