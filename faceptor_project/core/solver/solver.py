import numpy as np
import torch
import os
import shutil
from functools import partial
import time
import copy
import cv2

from torch import distributed
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from easydict import EasyDict as edict

from PIL import Image
from torchvision import transforms


from core.utils import setup_seed, printlog, worker_init_fn, AverageMeter, create_logger, load_state_model


from core.model.backbone import backbone_entry
from core.model.heads import heads_holder_entry
from core.model.loss import LossesHolder
from core.model.model_entry import AIOEntry

from core.optimizer import optimizer_entry
from core.lr_scheduler import lr_scheduler_entry

from core.data.transform import transform_entry
from core.data.dataset import dataset_entry
from core.data.sampler import sampler_entry
from core.data.dataloader import DataLoaderX

from core.evaluator import evaluator_entry





from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook



class Solver(object):

    def __init__(self, C):

        self.C = C
        cfg = C.common
        self.cfg = cfg
        self.task_cfgs = C.tasks
        self.tmp = edict()

        # global control random seed
        setup_seed(seed=cfg.seed, cuda_deterministic=cfg.cuda_deterministic)
        torch.cuda.set_device(C.local_rank)

        # output
        if cfg.get('out_dir', None) and cfg.get('expname', None) and cfg.get('start_time', None):
            self.final_out_dir = os.path.join(cfg.out_dir, cfg.expname, cfg.start_time)
        else:
            self.final_out_dir = "./"

        self.tb_path = '{}/tensorboard'.format(self.final_out_dir)
        self.ckpt_path = '{}/checkpoints'.format(self.final_out_dir)
        self.logs_path = '{}/logs'.format(self.final_out_dir)
        if C.rank == 0:
            os.makedirs(self.tb_path, exist_ok=True)
            os.makedirs(self.ckpt_path, exist_ok=True)
            os.makedirs(self.logs_path, exist_ok=True)

            self.tb_logger = SummaryWriter(log_dir=self.tb_path)

            config_save_to = os.path.join(self.ckpt_path, 'config.yaml')
            if not os.path.exists(config_save_to):
                shutil.copy(C.config_path, config_save_to)

            self.logger = create_logger('global_logger', '{}/training_log.txt'.format(self.logs_path))

        else:
            while not os.path.exists(self.logs_path):
                time.sleep(1)

        self.wandb_logger = None
        if cfg.wandb.use:
            import wandb
            # Sign in to wandb
            try:
                wandb.login(key=cfg.wandb.key)
            except Exception as e:
                printlog("WandB Key must be provided in config file.")
                printlog(f"Config Error: {e}")
            # Initialize wandb
            run_name = datetime.now().strftime("%y%m%d_%H%M") + f"_GPU{C.rank}"
            run_name = run_name if cfg.wandb.suffix_run_name is None else run_name + f"_{cfg.wandb.suffix_run_name}"
            try:
                self.wandb_logger = wandb.init(
                    entity = cfg.wandb.entity, 
                    project = cfg.wandb.project, 
                    sync_tensorboard = True,
                    resume=cfg.wandb.resume,
                    name = run_name, 
                    notes = cfg.wandb.notes) if C.rank == 0 or cfg.wandb_log_all else None
                if self.wandb_logger:
                    self.wandb_logger.config.update(cfg)
            except Exception as e:
                print("WandB Data (Entity and Project name) must be provided in config file.")
                print(f"Config Error: {e}")


        self.last_iter = -1


        printlog("Start!")

    
    def initialize(self):
        printlog(f"Initialize!")

        printlog(f"Create Model!")
        self.create_model()
        printlog(f"Done!")

        printlog(f"Create Optimizer!")
        self.create_optimizer()
        printlog(f"Done!")

        printlog(f"Create LR_Scheduler!")
        self.create_lr_scheduler()
        printlog(f"Done!")

        printlog(f"Load!")
        self.load()
        printlog(f"Done!")

        printlog(f"Create Dataloaders!")
        self.create_dataloaders()
        printlog(f"Done!")

        printlog(f"Create Evaluator!")
        self.create_evaluators()
        printlog(f"Done!")

        printlog("AMP!")
        self.amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)
        printlog(f"Done!")

        printlog("Initialization Done!")


    def create_model(self):

        backbone = backbone_entry(self.cfg.backbone)
        heads_holder = heads_holder_entry(self.cfg.heads)
        losses_holder = LossesHolder(self.task_cfgs)

        model = AIOEntry(self.cfg.model_entry, self.task_cfgs, backbone, heads_holder, losses_holder)
        model.cuda()

        if self.C.rank == 0:
            printlog("model: ",model)

        model = torch.nn.parallel.DistributedDataParallel(module=model, broadcast_buffers=False, 
                                                          device_ids=[self.C.local_rank], bucket_cap_mb=16,
                                                          find_unused_parameters=True)
        model.register_comm_hook(None, fp16_compress_hook)

        self.model = model


    def create_optimizer(self):
        ## param_group
        defaults = {}
        defaults["lr"] = self.cfg.lr_scheduler.kwargs.base_lr
        defaults["weight_decay"] = self.cfg.optimizer.kwargs.weight_decay

        memo = set()
        param_groups = []

        count = edict({'backbone': 0, 'heads': 0, 'losses':0, 'pos_embed': 0})

        for module_name, module in self.model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)
                
                # >>> param statistics:
                if "backbone_module" in module_name:
                    count.backbone += value.data.nelement()
                    if 'pos_embed' in module_param_name:
                        count.pos_embed += value.data.nelement()
                elif "heads_module" in module_name:
                    count.heads += value.data.nelement()
                elif "losses_module" in  module_name:
                    count.losses += value.data.nelement()
                
                # Set learning rate.
                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * self.cfg.get('backbone_multiplier', 1.0)
                if "heads" in module_name:
                    if "interpreters" in module_name:
                        hyperparams["lr"] = hyperparams["lr"] * self.cfg.get('interpreters_multiplier', 1.0)
                    elif "decoder" in module_name:
                        hyperparams["lr"] = hyperparams["lr"] * self.cfg.get('decoder_multiplier', 1.0)
                    else:
                        hyperparams["lr"] = hyperparams["lr"] * self.cfg.get('heads_multiplier', 1.0)
                if "losses_module" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * self.cfg.get('losses_multiplier', 1.0)

                param_groups.append({"params": [value], **hyperparams})
                
                if self.C.rank == 0:
                    self.logger.info(f"module_name: {module_name} \t\t "
                                     f"module_param_name: {module_param_name} \t\t "
                                     f"specification: {hyperparams} \t\t "
                                     f"params: {value.data.nelement()}"
                                     )
                    

        if self.C.rank == 0:

            printlog(f"Backbone Params: {count.backbone / 1e6:.2f}M")
            printlog(f"Heads Params: {count.heads / 1e6:.2f}M")
            printlog(f"Backbone PE Params: {count.pos_embed / 1e6:.2f}M")
            printlog(f"Losses Params: {count.losses / 1e6:.2f}M")
    
            total = count.backbone + count.heads + count.losses
            total_inference = count.backbone + count.heads

            printlog(f"All Params: {total / 1e6:.2f}M")
            printlog(f"All Inference Params: {total_inference / 1e6:.2f}M")


        self.cfg.optimizer.kwargs.params = param_groups
        self.cfg.optimizer.kwargs.lr = self.cfg.lr_scheduler.kwargs.base_lr
        self.optimizer = optimizer_entry(self.cfg.optimizer)


    def create_lr_scheduler(self):

        self.cfg.lr_scheduler.kwargs.optimizer = self.optimizer
        self.cfg.lr_scheduler.kwargs.last_iter = self.last_iter
        self.cfg.lr_scheduler.kwargs.max_iter = self.cfg.max_iter
        self.lr_scheduler = lr_scheduler_entry(self.cfg.lr_scheduler)

    
    def _get_model_name(self, load_path, rank, iter=""):

        print(load_path, rank, iter)

        if iter == "":
            max_iter = 0
            files = os.listdir(load_path)
            for file in files:
                if file.startswith('ckpt_rank0') and file.endswith('.pth.tar'):
                    try:
                        cur_iter = int(file.split('_')[-1].split('.')[0])
                    except:
                        cur_iter = 0
                    max_iter = max(max_iter, cur_iter)
            if max_iter > 0:
                model_name = 'checkpoint_rank{}_iter_{}.pth.tar'.format(rank, max_iter)
            else:
                printlog(f"No model in given path!")
                model_name = None
        else:
            model_name = 'checkpoint_rank{}_iter_{}.pth.tar'.format(rank, iter)

        final_load_path = os.path.join(load_path, model_name)
        if not os.path.exists(final_load_path):

            model_name = None

        return model_name
    

    def load(self, load_items=["state_dict", "optimizer", "step"]):

        if self.cfg['start_time'] == self.cfg['now']:
            printlog(f"No need to load!")
            return
        
        if self.cfg["load_path"] == "":
            load_path = self.ckpt_path
        else:
            load_path = self.cfg["load_path"]
        
        model_name = self._get_model_name(load_path, self.C.rank, self.cfg['load_iter'])



        if model_name == None:
            printlog(f"No need to load!")
            return
        final_load_path = os.path.join(load_path, model_name)

        try:
            checkpoint = torch.load(final_load_path, 'cpu')
        except:
            raise FileNotFoundError(f'=> no checkpoint found at {final_load_path}')
        printlog(f"Recovering from {final_load_path}, keys={list(checkpoint.keys())}")


        if 'state_dict' in checkpoint and 'state_dict' in load_items:
            pretrained_state_dict = checkpoint['state_dict']

            ignores = self.cfg['load_ignore']
            if len(ignores) > 0:
                for k in list(pretrained_state_dict.keys()):
                    flag = False
                    for prefix in ignores:
                        if k.startswith(prefix):
                            flag = True
                            the_prefix = prefix
                            break
                    if flag:
                        print('ignoring {} (prefix: {})'.format(k, the_prefix))
                        del pretrained_state_dict[k]
                        
            load_state_model(self.model, pretrained_state_dict)
        
        if 'optimizer' in checkpoint and 'optimizer' in load_items:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        if 'step' in checkpoint and 'step' in load_items:
            self.last_iter = checkpoint['step'] - 1


    def create_dataloader(self, task_cfg):
    
        dataset = dataset_entry(task_cfg.dataset)

        if self.cfg.backbone.type in ["FaRLVisualFeatures"]:

            from core.data.transform import FARL_INIT_MEAN, FARL_INIT_STD
            task_cfg.dataset.kwargs.augmentation.kwargs.mean = FARL_INIT_MEAN
            task_cfg.dataset.kwargs.augmentation.kwargs.std = FARL_INIT_STD

        printlog("Dataset for task {}:".format(task_cfg.name), dataset.__repr__())


        train_sampler = sampler_entry(task_cfg.sampler)(
            dataset=dataset, 
            task_name=task_cfg.name, 
            total_iter=self.cfg.max_iter,
            batch_size=task_cfg.sampler.batch_size, 
            world_size=self.C.world_size, 
            rank=self.C.rank,
            last_iter=self.last_iter,
            shuffle_strategy=task_cfg.sampler.shuffle_strategy, 
            random_seed=self.cfg.seed,
            ret_save_path=task_cfg.sampler.get('ret_save_path', None))


        init_fn = partial(worker_init_fn, num_workers=self.cfg.num_workers, rank=self.C.rank, seed=self.cfg.seed)

        train_loader = DataLoaderX(
            local_rank=self.C.local_rank,
            dataset=dataset,
            batch_size=task_cfg.sampler.batch_size,
            sampler=train_sampler,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=init_fn,
        )

        return train_loader


    def create_dataloaders(self):

        self.dataloaders = list()
        for i in range(self.C.num_tasks):
            dataloader = self.create_dataloader(self.task_cfgs[i])
            self.dataloaders.append(dataloader)


    def create_evaluators(self):
        if self.C.rank == 0:
            self.evaluators = dict()
            for i in range(self.C.num_tasks):
                if "evaluator" in self.task_cfgs[i].keys():
                    if self.task_cfgs[i].evaluator.get("use", True):
                        evaluator = evaluator_entry(self.task_cfgs[i].evaluator)
                        evaluator.set_tb_logger(self.tb_logger)
                        evaluator.set_logger(self.logger)
                        self.evaluators[self.task_cfgs[i].name] = evaluator

                        printlog("Dataset for eval {}:".format(self.task_cfgs[i].name), evaluator.dataset.__repr__())
            

    def run(self):

        cfg = self.cfg
        tmp = self.tmp
        tasks_cfg = self.task_cfgs

        self.pre_run()

        end = time.time()

        for i, data in enumerate(zip(*self.dataloaders)):

            tmp.current_step = self.last_iter + i + 1
            tmp.data_time.update(time.time() - end)
            
            loss, output = self.model(data, current_step=tmp.current_step)


            if cfg.fp16:
                self.amp.scale(loss).backward()
                if tmp.current_step % cfg.gradient_acc == 0:
                    self.amp.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                    self.amp.step(self.optimizer)
                    self.amp.update()
                    self.optimizer.zero_grad()
            else:
                loss.backward()
                if tmp.current_step % cfg.gradient_acc == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            self.lr_scheduler.step(tmp.current_step)
            tmp.current_lr = self.lr_scheduler.get_lr()[0]


            tmp.batch_time.update(time.time() - end)
            end = time.time()

            with torch.no_grad():
                self.tb_logging(output, loss)
                self.logging()
                self.evaluate()

            self.save()


    def pre_run(self):
        tmp = self.tmp
        tmp.data_time = AverageMeter(10)
        tmp.batch_time = AverageMeter(10)
        tmp.loss_total = AverageMeter(10)

        tmp.loss_list = [AverageMeter(10) for _ in range(self.C.num_tasks)]
        tmp.top1_list = [AverageMeter(10) for _ in range(self.C.num_tasks)]

        self.model.train()
        # FIXME using gradient checkpoint if there are some unused parameters will cause error
        self.model._set_static_graph()

    

    def tb_logging(self, output, loss):
        if self.C.rank == 0:
            tmp = self.tmp

            tmp.loss_total.update(loss.item())

            self.tb_logger.add_scalar('total_loss', loss.item(), tmp.current_step)

            for i in range(self.C.num_tasks):
                task_name = self.task_cfgs[i].name
                tmp.loss_list[i].update(output[task_name]["tloss"].item())
                tmp.top1_list[i].update(output[task_name]["top1"].item())
                
                self.tb_logger.add_scalar('task{}_{}_tloss'.format(i, self.task_cfgs[i].name), output[task_name]["tloss"].item(), tmp.current_step)
                self.tb_logger.add_scalar('task{}_{}_top1'.format(i, self.task_cfgs[i].name), output[task_name]["top1"].item(), tmp.current_step)
                
                if "losses" in output[task_name].keys():
                    for j in range(len(output[task_name]["losses"])):
                        self.tb_logger.add_scalar('task{}_{}_{}_loss'.format(i, self.task_cfgs[i].name, output[task_name]["loss_names"][j]), output[task_name]["losses"][j].item(), tmp.current_step)
                        self.tb_logger.add_scalar('task{}_{}_{}_weight'.format(i, self.task_cfgs[i].name, output[task_name]["loss_names"][j]), output[task_name]["weights"][j], tmp.current_step)


            self.tb_logger.add_scalar('lr', tmp.current_lr, tmp.current_step)

    def logging(self):
        if (self.tmp.current_step + 1) % self.cfg.get('print_interval', 10) == 0 and self.C.rank == 0:

            tmp = self.tmp
            cfg = self.cfg

            log_msg = '\t'.join([
                'Iter: [{0}/{1}] ',
                'Time: {batch_time.avg:.3f} (ETA:{eta:.2f}h) ({data_time.avg:.3f}) ',
                'Total Loss: {loss.avg:.4f} ',
                'LR: {current_lr} ',
                '{meters} ',
                'max mem: {memory:.0f}'
            ])

            MB = 1024.0 * 1024.0

            loss_str = []
            for i in range(self.C.num_tasks):
                loss_str.append(
                    "{}_loss(top1): {:4f}({:4f}) ".format(self.task_cfgs[i].name, tmp.loss_list[i].avg, tmp.top1_list[i].avg)
                )
            loss_str = '\t'.join(loss_str)
            log_msg = log_msg.format(tmp.current_step, cfg.max_iter, \
                            batch_time=tmp.batch_time, \
                            eta=(cfg.max_iter-tmp.current_step)*tmp.batch_time.avg/3600, \
                            data_time=tmp.data_time, \
                            loss=tmp.loss_total, \
                            current_lr=tmp.current_lr, \
                            meters=loss_str, \
                            memory=torch.cuda.max_memory_allocated() / MB)
            
            log_msg+="\n"

            self.logger.info(log_msg)

    def save(self):

        checkpoint = {
                'step': self.tmp.current_step+1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }

        if (self.tmp.current_step + 1) % self.cfg.get('ckpt_interval', 1000) == 0:
            torch.save(checkpoint, os.path.join(self.ckpt_path, f"checkpoint_rank{self.C.rank}_iter_newest.pth.tar"))
        
        if self.cfg.get('save_interval', -1) > 0 and ((self.tmp.current_step + 1) % self.cfg.save_interval == 0 or self.tmp.current_step + 1 == self.cfg.max_iter):
            torch.save(checkpoint, os.path.join(self.ckpt_path, f"checkpoint_rank{self.C.rank}_iter_{self.tmp.current_step + 1}.pth.tar"))


    def evaluate(self):
        

        if self.cfg.get('evaluate_interval', -1) > 0 and ((self.tmp.current_step + 1) % self.cfg.evaluate_interval == 0 or self.tmp.current_step + 1 == self.cfg.max_iter) and self.C.rank == 0:
            self.model.module.set_mode_to_evaluate()
            for task_name in self.evaluators.keys():

                evaluator = self.evaluators[task_name]
                self.model.module.set_evaluation_task(task_name)
                evaluator(self.tmp.current_step, self.model)

            self.model.module.set_mode_to_train()


    def test(self):
        printlog(f"Initialize!")

        printlog(f"Create Model!")
        self.create_model()
        printlog(f"Done!")

        printlog(f"Load!")
        self.load(load_items=["state_dict", "step"])
        printlog(f"Done!")

        printlog(f"Create Evaluator!")
        self.create_evaluators()
        printlog(f"Done!")


        self.model.module.set_mode_to_evaluate()
        for task_name in self.evaluators.keys():
            evaluator = self.evaluators[task_name]
            self.model.module.set_evaluation_task(task_name)
            evaluator(self.last_iter, self.model)





