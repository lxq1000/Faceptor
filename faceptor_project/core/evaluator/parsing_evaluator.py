import torch
import numpy as np
from timm.utils import accuracy, AverageMeter
from core.data.dataset import dataset_entry
from core.data.transform import transform_entry

from core.utils import get_dist_info, LimitedAvgMeter


class F1Score(object):
    """Compute F1 score among label and pred_label.

    Args:
        label_tag (str): The tag for groundtruth label, which is a np.ndarray with dtype=int.
        pred_label_tag (str): The tag for predicted label, which is a np.ndarray with dtype=int 
            and same shape with groundtruth label.
        label_names (List[str]): Names of the label values.
        num_labels (int): The number of valid label values.
    """

    def __init__(self, label_names, bg_label_name='background', compute_fg_mean=True) -> None:

        self.label_names = label_names
        self.num_labels = len(label_names)
        self.compute_fg_mean = compute_fg_mean
        self.bg_label_name = bg_label_name

    def init_evaluation(self):
        self.hists_sum = np.zeros(
            [self.num_labels, self.num_labels], dtype=np.int64)
        self.count = 0
        self.num_pixels = 0

    def evaluate(self, pred, target): 

        if target.shape != pred.shape:
            raise RuntimeError(
                f'The label shape {target.shape} mismatches the pred_label shape {pred.shape}')

        hist = __class__._collect_hist(
            target, pred, self.num_labels, self.num_labels)
        self.hists_sum += hist
        self.count += 1
        self.num_pixels += target.shape[0] * target.shape[1]

    def finalize_evaluation(self):
        # gather all hists_sum
        hists_sum = torch.from_numpy(self.hists_sum).contiguous().cuda()

        # compute F1 score
        A = hists_sum.sum(0).to(dtype=torch.float64)
        B = hists_sum.sum(1).to(dtype=torch.float64)
        intersected = hists_sum.diagonal().to(dtype=torch.float64)
        f1 = 2 * intersected / (A + B)

        f1s = {self.label_names[i]: f1[i].item()
               for i in range(self.num_labels)}
        if self.compute_fg_mean:
            f1s_fg = [f1[i].item() for i in range(self.num_labels)
                      if self.label_names[i] != self.bg_label_name]
            f1s['fg_mean'] = sum(f1s_fg) / len(f1s_fg)
        return f1s

    @staticmethod
    def _collect_hist(a: np.ndarray, b: np.ndarray, na: int, nb: int) -> np.ndarray:
        """
        fast histogram calculation

        Args:
            a, b: Non negative label ids, a.shape == b.shape, a in [0, ... na-1], b in [0, ..., nb-1]

        Returns:
            hist (np.ndarray): The histogram matrix with shape [na, nb].
        """
        assert np.all((a >= 0) & (a < na) & (b >= 0) & (b < nb))
        hist = np.bincount(
            nb * a.reshape([-1]).astype(np.int64) +
            b.reshape([-1]).astype(np.int64),
            minlength=na * nb).reshape(na, nb)
        assert np.sum(hist) == a.size
        return hist
    


class ParsingEvaluator(object):

    def __init__(self, test_dataset_cfg, mark, test_batch_size, test_post_trans, label_names, bg_label_name, evaluate_interval):

        rank, world_size = get_dist_info()
        self.rank = rank
        self.mark = mark

        self.evaluate_interval = evaluate_interval

        if self.rank is 0:
            self.dataset = dataset_entry(test_dataset_cfg)
            if test_dataset_cfg.type == "LaPaDataset":
                test_batch_size = 1
            self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=test_batch_size, 
                                                               shuffle=False, pin_memory=True, drop_last=False)
        
        self.f1_scorer = F1Score(label_names, bg_label_name)

        self.transform = transform_entry(test_post_trans)

        self.fg_mean_lmeter = LimitedAvgMeter(max_num=10, best_mode="max")


    def set_tb_logger(self, tb_logger, wandb_logger=None):
        self.tb_logger = tb_logger

    def set_logger(self, logger):
        self.logger = logger

    def ver_test(self, model, global_step):

        self.logger.info(self.mark)
        self.f1_scorer.init_evaluation()

        for idx, input_var in enumerate(self.data_loader):

            input_var["image"]=input_var["image"].cuda()
            out_var = model(input_var)
            output = out_var["head_output"].detach().cpu().numpy()

            label = input_var["label"].cpu().numpy()
            ori_image = input_var["ori_image"].cpu().numpy()
            align_matrix = input_var["align_matrix"].cpu().numpy()

            for j in range(label.shape[0]):
                
                target = label[j]
                temp=dict()
                temp["pred_warped_logits"] = output[j]
                temp["ori_image"] = ori_image[j]
                temp["align_matrix"] = align_matrix[j]
                pred = self.transform.process(temp)["pred"]

                self.f1_scorer.evaluate(pred, target)

        
        result = self.f1_scorer.finalize_evaluation()

        fg_mean = result["fg_mean"]
            
        self.fg_mean_lmeter.append(fg_mean)

        self.tb_logger.add_scalar(tag=f"{self.mark}_fg_mean", scalar_value=fg_mean, global_step=global_step)
        for each in result.keys():
            if each != "fg_mean":
                continue
            self.tb_logger.add_scalar(tag=f"{self.mark}_{each}", scalar_value=result[each], global_step=global_step)


        self.logger.info('[%s][%d]FG-Mean: %f' % (self.mark, global_step, fg_mean))
        self.logger.info('[%s][%d]FG-Mean-Highest: %f' % (self.mark, global_step, self.fg_mean_lmeter.best))
        self.logger.info('[%s][%d]FG-Mean-Mean@10: %f' % (self.mark, global_step, self.fg_mean_lmeter.avg))

        for each in result.keys():
            if each != "fg_mean":
                self.logger.info('[%s][%d]FG-%s: %f' % (self.mark, global_step, each, result[each]))



    def __call__(self, num_update, model):
        if self.rank is 0 and num_update > 0 and ((num_update+1) % self.evaluate_interval == 0):
            model.eval()
            self.ver_test(model, num_update)
            model.train()
            torch.cuda.empty_cache()