import torch
import numpy as np
from timm.utils import accuracy, AverageMeter
from core.data.dataset import dataset_entry
from core.data.transform import transform_entry

from core.utils import get_dist_info, LimitedAvgMeter

from typing import Mapping, List, Union, Optional
from collections import defaultdict

import math

import numpy as np
from scipy.integrate import simps
import torch
import torch.distributed as dist

from core.model.geometry import denormalize_points


class NormalizeInfo:
    def get_unit_dist(self, data) -> float:
        raise NotImplementedError()


class NormalizeByLandmarks(NormalizeInfo):
    def __init__(self, landmark_tag: str, left_id: Union[int, List[int]], right_id: Union[int, List[int]]):
        self.landmark_tag = landmark_tag
        if isinstance(left_id, int):
            left_id = [left_id]
        if isinstance(right_id, int):
            right_id = [right_id]
        self.left_id, self.right_id = left_id, right_id

    def get_unit_dist(self, data) -> float:
        landmark = data[self.landmark_tag]
        unit_dist = np.linalg.norm(landmark[self.left_id, :].mean(0) -
                                   landmark[self.right_id, :].mean(0), axis=-1)
        return unit_dist


class NormalizeByBox(NormalizeInfo):
    def __init__(self, box_tag: str):
        self.box_tag = box_tag

    def get_unit_dist(self, data) -> float:
        y1, x1, y2, x2 = data[self.box_tag]
        h = y2 - y1
        w = x2 - x1
        return math.sqrt(h * w)


class NormalizeByBoxDiag(NormalizeInfo):
    def __init__(self, box_tag: str):
        self.box_tag = box_tag

    def get_unit_dist(self, data) -> float:
        y1, x1, y2, x2 = data[self.box_tag]
        h = y2 - y1
        w = x2 - x1
        diag = math.sqrt(w * w + h * h)
        return diag


class NME(object):
    """Compute Normalized Mean Error among 2D landmarks and predicted 2D landmarks.

    Attributes:
        normalize_infos: Mapping[str, NormalizeInfo]: 
            Information to normalize for NME calculation.
    """

    def __init__(self, landmark_tag: str, pred_landmark_tag: str,
                 normalize_infos: Mapping[str, NormalizeInfo]) -> None:
        self.landmark_tag = landmark_tag
        self.pred_landmark_tag = pred_landmark_tag
        self.normalize_infos = normalize_infos

    def init_evaluation(self):
        self.nmes_sum = defaultdict(float)  # norm_name: str -> float
        self.count = defaultdict(int)  # norm_name: str -> int

    def evaluate(self, data: Mapping[str, np.ndarray]):
        landmark = data[self.landmark_tag]
        pred_landmark = data[self.pred_landmark_tag]

        if landmark.shape != pred_landmark.shape:
            raise RuntimeError(
                f'The landmark shape {landmark.shape} mismatches '
                f'the pred_landmark shape {pred_landmark.shape}')

        for norm_name, norm_info in self.normalize_infos.items():
            # compute unit distance for nme normalization
            unit_dist = norm_info.get_unit_dist(data)

            # compute normalized nme for this sample
            # [npoints] -> scalar
            nme = (np.linalg.norm(
                landmark - pred_landmark, axis=-1) / unit_dist).mean()
            self.nmes_sum[norm_name] += nme

            self.count[norm_name] += 1

    def finalize_evaluation(self) -> Mapping[str, float]:
        # gather all nmes_sum
        names_array: List[str] = list(self.nmes_sum.keys())

        nmes_sum = torch.tensor(
            [self.nmes_sum[name] for name in names_array],
            dtype=torch.float32, device='cuda')

        count_sum = torch.tensor(
            [self.count[name] for name in names_array],
            dtype=torch.int64, device='cuda')

        scores = dict()

        # compute nme scores
        for name, nmes_sum_val, count_val in zip(names_array, nmes_sum, count_sum):
            scores[name] = nmes_sum_val.item() / count_val.item()

        # compute final nme
        return scores


class AUC_FR(object):
    """Compute AUC and FR (Failure Rate).

    Output scores with name `'auc_{suffix_name}'` and `'fr_{suffix_name}'`.
    """

    def __init__(self, landmark_tag: str, pred_landmark_tag: str,
                 normalize_info: NormalizeInfo,
                 threshold: float, suffix_name: str, step: float = 0.0001,
                 gather_part_size: Optional[int] = 5) -> None:
        self.landmark_tag = landmark_tag
        self.pred_landmark_tag = pred_landmark_tag
        self.normalize_info = normalize_info
        self.threshold = threshold
        self.suffix_name = suffix_name
        self.step = step
        self.gather_part_size = gather_part_size

    def init_evaluation(self):
        self.nmes = []

    def evaluate(self, data: Mapping[str, np.ndarray]):
        landmark = data[self.landmark_tag]
        pred_landmark = data[self.pred_landmark_tag]

        if landmark.shape != pred_landmark.shape:
            raise RuntimeError(
                f'The landmark shape {landmark.shape} mismatches '
                f'the pred_landmark shape {pred_landmark.shape}')

        # compute unit distance for nme normalization
        unit_dist = self.normalize_info.get_unit_dist(data)

        # compute normalized nme for this sample
        nme = (np.linalg.norm(
            landmark - pred_landmark, axis=-1) / unit_dist).mean()
        self.nmes.append(nme)

    def finalize_evaluation(self) -> Mapping[str, float]:
        # gather all nmes
        nmes = self.nmes
        nmes = torch.tensor(nmes)
        nmes = nmes.sort(dim=0).values.cpu().numpy()

        # from https://github.com/HRNet/HRNet-Facial-Landmark-Detection/issues/6#issuecomment-503898737
        count = len(nmes)
        xaxis = list(np.arange(0., self.threshold + self.step, self.step))
        ced = [float(np.count_nonzero([nmes <= x])) / count for x in xaxis]
        auc = simps(ced, x=xaxis) / self.threshold
        fr = 1. - ced[-1]

        return {f'auc_{self.suffix_name}': auc, f'fr_{self.suffix_name}': fr}

    
def log(logger, tb_logger, mark, global_step, value, lmeter, logger_str, tb_logger_str):
        lmeter.append(value)
        logger.info('[%s][%d]%s: %f' % (mark, global_step, logger_str, value))
        logger.info('[%s][%d]%s-Best: %f' % (mark, global_step, logger_str, lmeter.best))
        logger.info('[%s][%d]%s-Mean@10: %f' % (mark, global_step, logger_str, lmeter.avg))

        tb_logger.add_scalar(tag=f"{mark}_{tb_logger_str}", scalar_value=value, global_step=global_step)
        
class IBUG300WEvaluator(object):

    def __init__(self, test_dataset_cfg, mark, test_batch_size, test_post_trans):

        rank, world_size = get_dist_info()
        self.rank = rank
        self.mark = mark

        if self.rank is 0:
            self.dataset = dataset_entry(test_dataset_cfg)
            test_batch_size = 1 ##
            self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=test_batch_size, 
                                                               shuffle=False, pin_memory=True, drop_last=False)
        
        self.NME_scorer = NME(landmark_tag="landmarks", 
                              pred_landmark_tag="pred_landmarks",
                              normalize_infos={
                                  "inter_ocular": NormalizeByLandmarks(landmark_tag="landmarks",
                                                                       left_id=36,
                                                                       right_id=45),
                                  "inter_pupil": NormalizeByLandmarks(landmark_tag="landmarks",
                                                                      left_id=[36, 37, 38, 39, 40, 41],
                                                                      right_id=[42, 43, 44, 45, 46, 47])})
        

        self.transform = transform_entry(test_post_trans)

        self.inter_ocular_lmeter = LimitedAvgMeter(max_num=10, best_mode="min")
        self.inter_pupil_lmeter = LimitedAvgMeter(max_num=10, best_mode="min")

    def set_tb_logger(self, tb_logger, wandb_logger=None):
        self.tb_logger = tb_logger

    def set_logger(self, logger):
        self.logger = logger

    def ver_test(self, model, global_step):

        self.logger.info(self.mark)

        self.NME_scorer.init_evaluation()

        for idx, input_var in enumerate(self.data_loader):

            input_var["image"]=input_var["image"].cuda()
            b, _, h, w = input_var["image"].shape

            out_var = model(input_var)
            output = denormalize_points(out_var["head_output"]["landmark"], h, w).detach().cpu().numpy()

            landmarks = input_var["landmarks"].cpu().numpy()
            crop_matrix = input_var["crop_matrix"].cpu().numpy()
            filename = input_var["filename"]

            for j in range(landmarks.shape[0]):

                temp=dict()
                temp["pred_warped_landmarks"] = output[j]
                temp["crop_matrix"] = crop_matrix[j]
                temp["landmarks"] = landmarks[j]
                temp["filename"] = filename[j]
                temp = self.transform.process(temp)

                self.NME_scorer.evaluate(temp)

        
        result = self.NME_scorer.finalize_evaluation()


        log(self.logger, self.tb_logger, self.mark, global_step, result["inter_ocular"], self.inter_ocular_lmeter, "Inter-Ocular", "inter_ocular")
        log(self.logger, self.tb_logger, self.mark, global_step, result["inter_pupil"], self.inter_pupil_lmeter, "Inter-Pupil", "inter_pupil")
    

    def __call__(self, num_update, model):
        if self.rank is 0 and num_update > 0:
            model.eval()
            self.ver_test(model, num_update)
            model.train()
            torch.cuda.empty_cache()

class COFWEvaluator(object):

    def __init__(self, test_dataset_cfg, mark, test_batch_size, test_post_trans):
        rank, world_size = get_dist_info()
        self.rank = rank
        self.mark = mark

        self.dataloaders = dict()

        if self.rank is 0:
            self.dataset = dataset_entry(test_dataset_cfg)
            test_batch_size = 1 ##
            self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=test_batch_size, 
                                                               shuffle=False, pin_memory=True, drop_last=False)
        
        self.NME_scorer = NME(landmark_tag="landmarks", 
                              pred_landmark_tag="pred_landmarks",
                              normalize_infos={
                                  "inter_ocular": NormalizeByLandmarks(landmark_tag="landmarks",
                                                                       left_id=8,
                                                                       right_id=9),
                                  })
        

        self.AUC_FR_scorer = AUC_FR(landmark_tag="landmarks",
                                    pred_landmark_tag="pred_landmarks",
                                    normalize_info=NormalizeByLandmarks(landmark_tag="landmarks", left_id=8, right_id=9),
                                    threshold=0.1,
                                    suffix_name="inter_ocular_10")
        

        self.transform = transform_entry(test_post_trans)

        self.inter_ocular_lmeter = LimitedAvgMeter(max_num=10, best_mode="min")
        self.auc_inter_ocular_10_lmeter = LimitedAvgMeter(max_num=10, best_mode="max")
        self.fr_inter_ocular_10_lmeter = LimitedAvgMeter(max_num=10, best_mode="max")

    def set_tb_logger(self, tb_logger, wandb_logger=None):
        self.tb_logger = tb_logger

    def set_logger(self, logger):
        self.logger = logger

    def ver_test(self, model, global_step):

        self.logger.info(self.mark)

        self.NME_scorer.init_evaluation()
        self.AUC_FR_scorer.init_evaluation()

        for idx, input_var in enumerate(self.data_loader):

            input_var["image"]=input_var["image"].cuda()
            b, _, h, w = input_var["image"].shape
            
            out_var = model(input_var)
            output = denormalize_points(out_var["head_output"]["landmark"], h, w).detach().cpu().numpy()

            landmarks = input_var["landmarks"].cpu().numpy()
            crop_matrix = input_var["crop_matrix"].cpu().numpy()

            for j in range(landmarks.shape[0]):

                temp=dict()
                temp["pred_warped_landmarks"] = output[j]
                temp["crop_matrix"] = crop_matrix[j]
                temp["landmarks"] = landmarks[j]
                temp = self.transform.process(temp)

                self.NME_scorer.evaluate(temp)
                self.AUC_FR_scorer.evaluate(temp)

        
        result1 = self.NME_scorer.finalize_evaluation()
        result2 = self.AUC_FR_scorer.finalize_evaluation()

        log(self.logger, self.tb_logger, self.mark, global_step, result1["inter_ocular"], self.inter_ocular_lmeter, "Inter-Ocular", "inter_ocular")
        log(self.logger, self.tb_logger, self.mark, global_step, result2["auc_inter_ocular_10"], self.auc_inter_ocular_10_lmeter, "AUC-Inter-Ocular-10", "auc_inter_ocular_10")
        log(self.logger, self.tb_logger, self.mark, global_step, result2["fr_inter_ocular_10"], self.fr_inter_ocular_10_lmeter, "FR-Inter-Ocular-10", "fr_inter_ocular_10")
    
    

    def __call__(self, num_update, model):
        if self.rank is 0 and num_update > 0:
            model.eval()
            self.ver_test(model, num_update)
            model.train()
            torch.cuda.empty_cache()


class WFLWEvaluator(object):

    def __init__(self, test_dataset_cfg, mark, test_batch_size, test_post_trans):
        
        rank, world_size = get_dist_info()
        self.rank = rank
        self.mark = mark

        self.dataloaders = dict()

        if self.rank is 0:
            self.dataset = dataset_entry(test_dataset_cfg)
            test_batch_size = 1 ##
            self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=test_batch_size, 
                                                               shuffle=False, pin_memory=True, drop_last=False)
        
        self.NME_scorer = NME(landmark_tag="landmarks", 
                              pred_landmark_tag="pred_landmarks",
                              normalize_infos={
                                  "inter_ocular": NormalizeByLandmarks(landmark_tag="landmarks",
                                                                       left_id=60,
                                                                       right_id=72),
                                  "inter_pupil": NormalizeByLandmarks(landmark_tag="landmarks",
                                                                      left_id=96,
                                                                      right_id=97)})
        

        self.AUC_FR_scorer = AUC_FR(landmark_tag="landmarks",
                                    pred_landmark_tag="pred_landmarks",
                                    normalize_info=NormalizeByLandmarks(landmark_tag="landmarks", left_id=60, right_id=72),
                                    threshold=0.1,
                                    suffix_name="inter_ocular_10")

        self.transform = transform_entry(test_post_trans)

        self.inter_ocular_lmeter = LimitedAvgMeter(max_num=10, best_mode="min")
        self.inter_pupil_lmeter = LimitedAvgMeter(max_num=10, best_mode="min")
        self.auc_inter_ocular_10_lmeter = LimitedAvgMeter(max_num=10, best_mode="max")
        self.fr_inter_ocular_10_lmeter = LimitedAvgMeter(max_num=10, best_mode="max")

    def set_tb_logger(self, tb_logger, wandb_logger=None):
        self.tb_logger = tb_logger

    def set_logger(self, logger):
        self.logger = logger

    def ver_test(self, model, global_step):

        self.logger.info(self.mark)

        self.NME_scorer.init_evaluation()
        self.AUC_FR_scorer.init_evaluation()


        for idx, input_var in enumerate(self.data_loader):

            input_var["image"]=input_var["image"].cuda()
            b, _, h, w = input_var["image"].shape
            out_var = model(input_var)
            output = denormalize_points(out_var["head_output"]["landmark"], h, w).detach().cpu().numpy()

            landmarks = input_var["landmarks"].cpu().numpy()
            crop_matrix = input_var["crop_matrix"].cpu().numpy()

            for j in range(landmarks.shape[0]):

                temp=dict()
                temp["pred_warped_landmarks"] = output[j]
                temp["crop_matrix"] = crop_matrix[j]
                temp["landmarks"] = landmarks[j]
                temp = self.transform.process(temp)

                self.NME_scorer.evaluate(temp)
                self.AUC_FR_scorer.evaluate(temp)

        
        result1 = self.NME_scorer.finalize_evaluation()
        result2 = self.AUC_FR_scorer.finalize_evaluation()

        log(self.logger, self.tb_logger, self.mark, global_step, result1["inter_ocular"], self.inter_ocular_lmeter, "Inter-Ocular", "inter_ocular")
        log(self.logger, self.tb_logger, self.mark, global_step, result1["inter_pupil"], self.inter_pupil_lmeter, "Inter-Pupil", "inter_pupil")
        log(self.logger, self.tb_logger, self.mark, global_step, result2["auc_inter_ocular_10"], self.auc_inter_ocular_10_lmeter, "AUC-Inter-Ocular-10", "auc_inter_ocular_10")
        log(self.logger, self.tb_logger, self.mark, global_step, result2["fr_inter_ocular_10"], self.fr_inter_ocular_10_lmeter, "FR-Inter-Ocular-10", "fr_inter_ocular_10")
    
    

    def __call__(self, num_update, model):
        if self.rank is 0 and num_update > 0:
            model.eval()
            self.ver_test(model, num_update)
            model.train()
            torch.cuda.empty_cache()


class AFLWEvaluator(object):

    def __init__(self, test_dataset_cfg, mark, test_batch_size, test_post_trans):


        rank, world_size = get_dist_info()
        self.rank = rank
        self.mark = mark

        self.dataloaders = dict()

        if self.rank is 0:
            self.dataset = dataset_entry(test_dataset_cfg)
            test_batch_size = 1 ##
            self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=test_batch_size, 
                                                               shuffle=False, pin_memory=True, drop_last=False)
        
        self.NME_scorer = NME(landmark_tag="landmarks", 
                              pred_landmark_tag="pred_landmarks",
                              normalize_infos={
                                  "inter_ocular": NormalizeByLandmarks(landmark_tag="landmarks",
                                                                       left_id=6,
                                                                       right_id=11),
                                  "inter_pupil": NormalizeByLandmarks(landmark_tag="landmarks",
                                                                      left_id=7,
                                                                      right_id=10),
                                  "box": NormalizeByBox(box_tag="box"),
                                  "diag": NormalizeByBoxDiag(box_tag="box")})
        

        self.AUC_FR_scorer = AUC_FR(landmark_tag="landmarks",
                                    pred_landmark_tag="pred_landmarks",
                                    normalize_info=NormalizeByBox(box_tag="box"),
                                    threshold=0.07,
                                    suffix_name="box_7")
        

        self.transform = transform_entry(test_post_trans)

        self.inter_ocular_lmeter = LimitedAvgMeter(max_num=10, best_mode="min")
        self.inter_pupil_lmeter = LimitedAvgMeter(max_num=10, best_mode="min")
        self.box_lmeter = LimitedAvgMeter(max_num=10, best_mode="min")
        self.diag_lmeter = LimitedAvgMeter(max_num=10, best_mode="min")
        self.auc_box_7_lmeter = LimitedAvgMeter(max_num=10, best_mode="max")
        self.fr_box_7_lmeter = LimitedAvgMeter(max_num=10, best_mode="max")

    def set_tb_logger(self, tb_logger, wandb_logger=None):
        self.tb_logger = tb_logger

    def set_logger(self, logger):
        self.logger = logger

    def ver_test(self, model, global_step):

        self.logger.info(self.mark)

        self.NME_scorer.init_evaluation()
        self.AUC_FR_scorer.init_evaluation()

        for idx, input_var in enumerate(self.data_loader):

            input_var["image"]=input_var["image"].cuda()
            b, _, h, w = input_var["image"].shape
            
            out_var = model(input_var)
            output = denormalize_points(out_var["head_output"]["landmark"], h, w).detach().cpu().numpy()


            landmarks = input_var["landmarks"].cpu().numpy()
            crop_matrix = input_var["crop_matrix"].cpu().numpy()
            box = input_var["box"].cpu().numpy()
            filename = input_var["filename"]

            for j in range(landmarks.shape[0]):
                temp=dict()
                temp["pred_warped_landmarks"] = output[j]
                temp["crop_matrix"] = crop_matrix[j]
                temp["landmarks"] = landmarks[j]
                temp["box"] = box[j]
                temp["filename"] = filename[j]
                temp = self.transform.process(temp)

                self.NME_scorer.evaluate(temp)
                self.AUC_FR_scorer.evaluate(temp)

        
        result1 = self.NME_scorer.finalize_evaluation()
        result2 = self.AUC_FR_scorer.finalize_evaluation()

        log(self.logger, self.tb_logger, self.mark, global_step, result1["inter_ocular"], self.inter_ocular_lmeter, "Inter-Ocular", "inter_ocular")
        log(self.logger, self.tb_logger, self.mark, global_step, result1["inter_pupil"], self.inter_pupil_lmeter, "Inter-Pupil", "inter_pupil")
        log(self.logger, self.tb_logger, self.mark, global_step, result1["box"], self.box_lmeter, "Box", "box")
        log(self.logger, self.tb_logger, self.mark, global_step, result1["diag"], self.diag_lmeter, "Diag", "diag")

        log(self.logger, self.tb_logger, self.mark, global_step, result2["auc_box_7"], self.box_lmeter, "AUC-Box-7", "auc_box_7")
        log(self.logger, self.tb_logger, self.mark, global_step, result2["fr_box_7"], self.diag_lmeter, "FR-Box-7", "fr_box_7")



    def __call__(self, num_update, model):
        if self.rank is 0 and num_update > 0:
            model.eval()
            self.ver_test(model, num_update)
            model.train()
            torch.cuda.empty_cache()


    



