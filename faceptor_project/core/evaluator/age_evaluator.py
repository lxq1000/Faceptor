
import torch
import numpy as np
from timm.utils import accuracy, AverageMeter
from core.data.dataset import dataset_entry

from core.utils import get_dist_info, LimitedAvgMeter
import torch.nn.functional as F



class AgeEvaluator_V2(object):

    def __init__(self, test_dataset_cfg, test_batch_size, mark):


        rank, world_size = get_dist_info()
        self.rank = rank
        self.mark = mark

        self.mae_lowest = 100.0
        self.cs_highest = 0.0
        self.eps_error_lowest = 100.0

        if self.rank is 0:
            self.dataset = dataset_entry(test_dataset_cfg)
            self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=test_batch_size, 
                                                               shuffle=False, pin_memory=True, drop_last=False)
        
        self.mae_lmeter = LimitedAvgMeter(max_num=10, best_mode="min")
        self.cs_lmeter=LimitedAvgMeter(max_num=10, best_mode="max")
        self.eps_error_lmeter=LimitedAvgMeter(max_num=10, best_mode="min")

    def set_tb_logger(self, tb_logger, wandb_logger=None):
        self.tb_logger = tb_logger

    def set_logger(self, logger):
        self.logger = logger

    def ver_test(self, model, global_step):

        self.logger.info(self.mark)

        mae_meter = AverageMeter()
        cs_meter = AverageMeter()
        eps_error_meter = AverageMeter()

        for idx, input_var in enumerate(self.data_loader):

            input_var["image"]=input_var["image"].cuda()
            out_var = model(input_var)
            age_output = out_var["head_output"]


            age_output = F.sigmoid(age_output)
            age_output = F.normalize(age_output, p=1, dim=1)


            label = input_var["label"]
            std_label = 3.0
            avg_label = label["avg_label"].numpy()


            rank = torch.Tensor([i for i in range(101)]).cuda()
            age_output = torch.sum(age_output*rank, dim=1)

            age_output = age_output.cpu().detach().numpy()

            mae = np.array(abs(age_output - avg_label), dtype=np.float32).mean()
            cs = np.array(abs(age_output - avg_label) <= 5, dtype=np.float32).mean()*100
            eps_error = 1 - np.mean((1 / np.exp(np.square(np.subtract(age_output, avg_label)) / (2 * np.square(std_label)))))

            mae_meter.update(mae, age_output.shape[0])
            cs_meter.update(cs, age_output.shape[0])
            eps_error_meter.update(eps_error, age_output.shape[0])
            

        if mae_meter.avg < self.mae_lowest:
            self.mae_lowest = mae_meter.avg
        if cs_meter.avg > self.cs_highest:
            self.cs_highest = cs_meter.avg
        if eps_error_meter.avg < self.eps_error_lowest:
            self.eps_error_lowest = eps_error_meter.avg

        self.mae_lmeter.append(mae_meter.avg)
        self.cs_lmeter.append(cs_meter.avg)
        self.eps_error_lmeter.append(eps_error_meter.avg)

        self.tb_logger.add_scalar(tag=f"{self.mark}_mae", scalar_value=mae_meter.avg, global_step=global_step)
        self.tb_logger.add_scalar(tag=f"{self.mark}_cs", scalar_value=cs_meter.avg, global_step=global_step)
        self.tb_logger.add_scalar(tag=f"{self.mark}_eps_error", scalar_value=eps_error_meter.avg, global_step=global_step)


        self.logger.info('[%s][%d]MAE: %f' % (self.mark, global_step, mae_meter.avg))
        self.logger.info('[%s][%d]MAE-Lowest: %f' % (self.mark, global_step, self.mae_lowest))
        self.logger.info('[%s][%d]MAE-Mean@10: %f' % (self.mark, global_step, self.mae_lmeter.avg))
        self.logger.info('[%s][%d]CS: %f' % (self.mark, global_step, cs_meter.avg))
        self.logger.info('[%s][%d]CS-Highest: %f' % (self.mark, global_step, self.cs_highest))
        self.logger.info('[%s][%d]CS-Mean@10: %f' % (self.mark, global_step, self.cs_lmeter.avg))
        self.logger.info('[%s][%d]Eps-Error: %f' % (self.mark, global_step, eps_error_meter.avg))
        self.logger.info('[%s][%d]Eps-Error-Lowest: %f' % (self.mark, global_step, self.eps_error_lowest))
        self.logger.info('[%s][%d]Eps-Error-Mean@10: %f' % (self.mark, global_step, self.eps_error_lmeter.avg))


    def __call__(self, num_update, model):
        if self.rank is 0 and num_update > 0:
            model.eval()
            self.ver_test(model, num_update)
            model.train()
            torch.cuda.empty_cache()

