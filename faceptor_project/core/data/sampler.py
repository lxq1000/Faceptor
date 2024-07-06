import os
import numpy as np
import torch

from core.utils import printlog, get_dist_info
from torch.utils.data.sampler import Sampler


def sampler_entry(config):
    #printlog('Dataset config[kwargs]',config['kwargs'])
    return globals()[config['type']]

class DistributedGivenIterationSampler(Sampler):
    def __init__(self, dataset, task_name, total_iter, batch_size, world_size=None, rank=None, last_iter=-1,
                 shuffle_strategy=1, random_seed=0, ret_save_path=None):

        printlog('Sampler: rank={}, world_size={}, random_seed={}'.format(rank, world_size, random_seed))

        self.dataset = dataset
        self.task_name = task_name
        self.total_iter = total_iter
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.last_iter = last_iter
        self.shuffle_strategy = shuffle_strategy
        self.random_seed = random_seed
        self.ret_save_path = ret_save_path
       

        self.total_size = self.total_iter * self.batch_size

        self.call = 0

        # generate indices
        if self.ret_save_path is not None:
            self.this_ret_path = os.path.join(self.ret_save_path, '_'.join(
                [self.task_name, str(self.world_size), str(self.rank)]) + ".pth.tar")
            if os.path.exists(self.this_ret_path):
                ret_file = torch.load(self.this_ret_path)
                # ensure this task and task size is unchanged
                if ret_file['task_name'] == self.task_name and ret_file['task_size'] == self.world_size and ret_file[
                    'task_rank'] == self.rank:
                    printlog(" load task sampler from ------> {}".format(self.this_ret_path))
                    self.indices = ret_file['ret_file']
                    self.dataset.received_indices = True
                    return
            else:
                printlog("sampler file ({}) is not existed, and will be generated now--->".format(self.this_ret_path))

        if self.shuffle_strategy in [0, 1]:
            self.indices = self.gen_new_list()
            self.dataset.indices = self.indices
            self.dataset.received_indices = True

        else:
            raise RuntimeError("Invalid shuffle_strategy!")

        if self.ret_save_path is not None and not os.path.exists(self.ret_save_path):
            self.save()
  

    def __iter__(self):

        return iter(self.indices)

    def gen_new_list(self):

        # each process shuffle independently
        if self.shuffle_strategy == 0:

            np.random.seed(self.rank)

            indices = np.arange(len(self.dataset))
            indices = indices[:self.total_size]
            num_repeat = (self.total_size - 1) // indices.shape[0] + 1
            indices = np.tile(indices, num_repeat)
            indices = indices[:self.total_size]

            for beg in range(0, self.total_size, len(self.dataset)):
                end = min(beg + len(self.dataset), self.total_size)
                np.random.shuffle(indices[beg:end])

        # each process shuffle all list with same seed, and pick one piece according to rank
        elif self.shuffle_strategy == 1:

            np.random.seed(self.random_seed)

            all_size = self.total_size * self.world_size
            indices = np.arange(len(self.dataset))
            indices = indices[:all_size]
            num_repeat = (all_size - 1) // indices.shape[0] + 1
            indices = np.tile(indices, num_repeat)
            indices = indices[:all_size]

            np.random.shuffle(indices)
            beg = self.total_size * self.rank
            indices = indices[beg:beg + self.total_size]

        else:
            raise RuntimeError('unknow shuffle strategy')

        assert len(indices) == self.total_size

        return indices[(self.last_iter + 1) * self.batch_size:]

    def __len__(self):
        return self.total_size - (self.last_iter + 1) * self.batch_size

    def save(self):
        torch.save({'task_name': self.task_name,
                    'task_size': self.world_size,
                    'task_rank': self.rank,
                    'ret_file': self.indices}, self.this_ret_path)
        printlog("save sampler file  ------> {}".format(self.this_ret_path))

