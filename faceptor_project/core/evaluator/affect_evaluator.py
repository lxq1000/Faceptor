
import torch
import numpy as np
from timm.utils import accuracy, AverageMeter
from core.data.dataset import dataset_entry

from core.utils import get_dist_info, LimitedAvgMeter



class SingferEvaluator(object):

    def __init__(self, test_dataset_cfg, test_batch_size, mark):

        rank, world_size = get_dist_info()
        self.rank = rank
        self.mark = mark

        self.acc_highest = 0.0

        if self.rank is 0:
            self.dataset = dataset_entry(test_dataset_cfg)
            self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=test_batch_size, 
                                                               shuffle=False, pin_memory=True, drop_last=False)

        self.acc_lmeter = LimitedAvgMeter(max_num=10, best_mode="max")

    def set_tb_logger(self, tb_logger, wandb_logger=None):
        self.tb_logger = tb_logger

    def set_logger(self, logger):
        self.logger = logger
    
    def ver_test(self, model, global_step):

        self.logger.info(self.mark)

        acc_meter = AverageMeter()

        for idx, input_var in enumerate(self.dataloader):

            input_var["image"]=input_var["image"].cuda()
            target = input_var["label"].cuda()
            out_var = model(input_var)
            logits = out_var["head_output"]

            acc1 = accuracy(logits, target, topk=(1,))[0]
            acc_meter.update(acc1.item(), target.size(0))

        if acc_meter.avg > self.acc_highest:
            self.acc_highest = acc_meter.avg

        self.acc_lmeter.append(acc_meter.avg)

        self.tb_logger.add_scalar(tag=f"{self.mark}_acc", scalar_value=acc_meter.avg, global_step=global_step)

        self.logger.info('[%s][%d]Acc: %1.5f' % (self.mark, global_step, acc_meter.avg))
        self.logger.info('[%s][%d]Acc-Highest: %1.5f' % (self.mark, global_step, self.acc_highest))
        self.logger.info('[%s][%d]Acc-Mean@10: %1.5f' % (self.mark, global_step, self.acc_lmeter.avg))

    def __call__(self, num_update, model):
        if self.rank is 0 and num_update > 0:
            model.eval()
            self.ver_test(model, num_update)
            model.train()
            torch.cuda.empty_cache()


