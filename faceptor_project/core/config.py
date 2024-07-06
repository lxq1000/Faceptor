import yaml
import numpy as np
from easydict import EasyDict as edict
import re
from core.utils import printlog

loader = yaml.SafeLoader
loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))


class Config(object):

    def __init__(self, args, rank, local_rank, world_size):

        config_path = args.config

        with open(config_path) as f:
            self.config = yaml.load(f, Loader=loader)

        self.common = edict(self.config["common"])
        
        if args.config is not None:
            self.common.config = args.config
        if args.expname is not None:
            self.common.expname = args.expname
        if args.start_time is not None:
            self.common.start_time = args.start_time
        if args.now is not None:
            self.common.now = args.now
        if args.out_dir is not None:
            self.common.out_dir = args.out_dir
        if args.load_path is not None:
            self.common.load_path = args.load_path
        if args.load_iter is not None:
            self.common.load_iter = args.load_iter
        if args.load_ignore is not None:
            self.common.load_ignore = args.load_ignore
        
        
        self.tasks = dict()
        self.num_tasks = len(self.config["tasks"])

        self.re_weighted = False
        print(self.common)
        if self.common.get("task_weight", None):
            self.re_weighted = True
            self.task_type_info = dict()
            for each in self.common.task_weight.keys():
                self.task_type_info[each] = {"weight_sum": 0., "task_ids":[]}
                

        temp_batch_size = 0
        for i in range(self.num_tasks):
            self.tasks[i] = edict(self.config["tasks"][i])

            if self.re_weighted:
                task_type = self.tasks[i].name.split("_")[0]
                self.task_type_info[task_type]["weight_sum"] += float(self.tasks[i].loss_weight)
                self.task_type_info[task_type]["task_ids"].append(i)

                if task_type not in self.task_type_info.keys():
                    self.task_type_info[task_type]

            temp_batch_size += int(self.tasks[i].sampler.batch_size)

        if self.re_weighted:
            for each in self.common.task_weight.keys():
                for i in self.task_type_info[each]["task_ids"]:
                    self.tasks[i].loss_weight = self.tasks[i].loss_weight / self.task_type_info[each]["weight_sum"] * float(self.common.task_weight[each])


        self.common.total_batch_size = temp_batch_size * world_size

        if self.common.lr_scale:
            self.common.lr_scheduler.kwargs.base_lr = self.common.lr_scheduler.kwargs.base_lr / self.common.lr_base * self.common.total_batch_size
            self.common.lr_scheduler.kwargs.warmup_lr = self.common.lr_scheduler.kwargs.warmup_lr / self.common.lr_base * self.common.total_batch_size

        printlog("config.common:", self.common)

        for i in range(self.num_tasks):
            printlog("config.tasks.{}:".format(i), self.tasks[i])
        
        self.config_path = config_path
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size



        




        