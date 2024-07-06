from torch.utils.data import Dataset
import numpy as np
import os
from torchvision import transforms
import numbers
import mxnet as mx
import torch
from torchvision.datasets import ImageFolder

from ..transform import transform_entry


class MXFaceDataset(Dataset):
    def __init__(self, data_path, augmentation, **kwargs):
        super(MXFaceDataset, self).__init__()
        
        self.transform = transform_entry(augmentation)

        self.root_dir = data_path

        path_imgrec = os.path.join(self.root_dir, 'train.rec')
        path_imgidx = os.path.join(self.root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))


    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)

        output = {'image': sample, 'label': label, 'filename': str(idx)}

        return output

    def __len__(self):
        return len(self.imgidx)
    

    def __repr__(self):
        return self.__class__.__name__ + \
               f'\ndataset_len: {len(self.imgidx)}\ntransform: {self.transform}'


