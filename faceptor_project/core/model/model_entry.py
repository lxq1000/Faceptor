import torch
import torch.nn as nn
import torch.nn.functional as F
from core.model.geometry import denormalize_points

class AIOEntry(nn.Module):
    def __init__(self, model_entry_cfg, task_cfgs, backbone_module, heads_module, losses_module):
        super(AIOEntry, self).__init__()

        self.model_entry_cfg = model_entry_cfg
        self.task_cfgs = task_cfgs
        self.backbone_module = backbone_module
        self.heads_module = heads_module
        self.losses_module = losses_module

        self.mode = "train"

        # set group_infos
        self.num_groups = len(self.model_entry_cfg.kwargs.size_group)
        self.group_infos = [self.model_entry_cfg.kwargs.size_group["group_"+str(i)] for i in range(self.num_groups)]
        
        for each in self.group_infos:
            each["task_names"] = []
            each["task_ids"] = []
            each["batch_sizes"] = []

            batch_sum = 0
            each["batch_range"] = [0]

            for task_type in each["task_types"]:
                for i in range(len(self.task_cfgs)):
                    if task_type in self.task_cfgs[i].name:
                        each["task_names"].append(self.task_cfgs[i].name)
                        each["task_ids"].append(i)
                        each["batch_sizes"].append(self.task_cfgs[i].sampler.batch_size)
                        batch_sum += self.task_cfgs[i].sampler.batch_size
                        each["batch_range"].append(batch_sum)


    def forward_backbone(self, data, current_step=0):

        output = dict()

        for group_idx in range(self.num_groups):
            images = [data[i]["image"] for i in self.group_infos[group_idx]["task_ids"]]

            images = torch.cat(images, dim=0) 
  
            x = self.backbone_module(images)

            for i in range(len(self.group_infos[group_idx]["task_ids"])):
                temp=dict()
                name = self.group_infos[group_idx]["task_names"][i]
                batch_range = self.group_infos[group_idx]["batch_range"]
                temp['backbone_output'] = [each[batch_range[i]: batch_range[i+1]] for each in x]
                output[name] = temp

        return output
    
    def set_evaluation_task(self, task_name):
        self.evaluation_task = task_name

    def set_mode_to_train(self):
        self.mode = "train"
    
    def set_mode_to_evaluate(self):
        self.mode = "evaluate"


    def forward(self, data, current_step=0):

        if self.mode == "train":
            outputs = self.forward_backbone(data, current_step) # backbone_output
            
            for i in range(len(data)):
                task_name = self.task_cfgs[i].name
                outputs[task_name]["label"] = data[i]["label"] 
                outputs[task_name]["task_name"] = task_name
                outputs[task_name]["task_id"] = i # backbone_output label task_name task_id
                outputs[task_name]["current_iter"] = current_step

            outputs = self.heads_module(outputs) # backbone_output label task_name task_id head_output

            total_loss, outputs = self.losses_module(outputs) # tloss acc1 losses(list) weights(list)

            return total_loss, outputs
        
        if self.mode == "evaluate":
            outputs = dict()
            outputs[self.evaluation_task] = dict()
            outputs[self.evaluation_task]["task_name"] = self.evaluation_task

            x = self.backbone_module(data["image"])

            outputs[self.evaluation_task]["backbone_output"] = x
            outputs = self.heads_module(outputs)

            return outputs[self.evaluation_task]
        