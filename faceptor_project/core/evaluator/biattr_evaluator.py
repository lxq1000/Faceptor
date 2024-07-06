
import torch
import numpy as np
from timm.utils import accuracy, AverageMeter
from core.data.dataset import dataset_entry

from core.utils import get_dist_info, LimitedAvgMeter, binary_accuracy



class BiAttrEvaluator(object):

    def __init__(self, test_dataset_cfg, test_batch_size, mark):

        rank, world_size = get_dist_info()
        self.rank = rank
        self.mark = mark

        self.mean_acc_highest = 0.0

        if self.rank is 0:
            self.dataset = dataset_entry(test_dataset_cfg)
            self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=test_batch_size, 
                                                               shuffle=False, pin_memory=True, drop_last=False)
            
            self.attr_num = self.dataset.attr_num
            self.attr_names = self.dataset.attr_names        

        self.mean_acc_lmeter = LimitedAvgMeter(max_num=10, best_mode="max")

    def set_tb_logger(self, tb_logger, wandb_logger=None):
        self.tb_logger = tb_logger

    def set_logger(self, logger):
        self.logger = logger

    def ver_test(self, model, global_step):

        self.logger.info(self.mark)

        mean_acc_meter = AverageMeter()
        attr_acc_meters = [AverageMeter() for i in range(self.attr_num)]

        for idx, input_var in enumerate(self.data_loader):
            
            input_var["image"]=input_var["image"].cuda()
            out_var = model(input_var)
            logits = out_var["head_output"]

            labels = input_var["label"].cuda()

            acc_list = binary_accuracy(logits, labels)

            mean_acc = sum(acc_list) / len(acc_list)

            mean_acc_meter.update(mean_acc, labels.shape[0])
            for i in range(self.attr_num):
                attr_acc_meters[i].update(acc_list[i], labels.shape[0])

        if mean_acc_meter.avg > self.mean_acc_highest:
            self.mean_acc_highest = mean_acc_meter.avg

        self.mean_acc_lmeter.append(mean_acc_meter.avg)

        self.tb_logger.add_scalar(tag=f"{self.mark}_mean_acc", scalar_value=mean_acc_meter.avg, global_step=global_step)
        for i in range(self.attr_num):
            self.tb_logger.add_scalar(tag=f"{self.mark}_{self.attr_names[i]}_acc", scalar_value=attr_acc_meters[i].avg, global_step=global_step)

        self.logger.info('[%s][%d]Mean-Acc: %f' % (self.mark, global_step, mean_acc_meter.avg))
        self.logger.info('[%s][%d]Mean-Acc-Highest: %f' % (self.mark, global_step, self.mean_acc_highest))
        self.logger.info('[%s][%d]Mean-Acc-Mean@10: %f' % (self.mark, global_step, self.mean_acc_lmeter.avg))

        for i in range(self.attr_num):
            self.logger.info('[%s][%d]%s-Acc: %f' % (self.mark, global_step,self.attr_names[i], attr_acc_meters[i].avg))
        
  

    def __call__(self, num_update, model):
        #if self.rank is 0 and num_update > 0:
        model.eval()
        self.ver_test(model, num_update)
        model.train()



'''
 ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 
 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 
 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 
 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 
 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 
 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 
 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 
 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
'''