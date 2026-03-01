import time
import math

import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from scipy.linalg import qr


def get_target_modules_list(model, target_modules):
    target_names = []
    for n, _ in model.named_modules():
        if any(t in n for t in target_modules) :
            target_names.append(n)
    return target_names


def replace_pica_with_fused_linear(model, target_modules_list):
    print("Replacing PiCa layers with new Linear layers")

    model.eval()
    target_modules_list = [l for l in target_modules_list if "pica_layer" not in l]

    shared_param_names = [name for name, _ in model.named_parameters() if name.startswith("shared_m_")]
    
    for target_path in tqdm(reversed(target_modules_list), total=len(target_modules_list)):
        parent_path = target_path[: target_path.rfind(".")] if "." in target_path else ""
        target_name = target_path.split(".")[-1]
        parent = model.get_submodule(parent_path) if parent_path else model
        target = model.get_submodule(target_path)

        if not hasattr(target, 'pica_layer') or not isinstance(target.pica_layer, PiCaLayer):
            continue
        
        in_dim = target.pica_layer.in_dim
        out_dim = target.pica_layer.out_dim
        if target.bias is None:
            lin = torch.nn.Linear(in_dim, out_dim, bias=False)
        else:
            lin = torch.nn.Linear(in_dim, out_dim, bias=True)
            lin.bias.data = target.bias.data
        lin.weight.data = target.merge_and_unload()
        parent.__setattr__(target_name, lin)

    state_dict = model.state_dict()
    for param_name in shared_param_names:
        if param_name in state_dict:
            del state_dict[param_name]
    
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")
    
    for param_name in shared_param_names:
        if param_name.count('.') == 0: 
            if hasattr(model, param_name):
                delattr(model, param_name)
                
def create_and_replace_modules(model, target_modules_list, create_fn):
    print("Replacing Linear layers with PiCa layers")

    for target_path in tqdm(reversed(target_modules_list), total=len(target_modules_list)):
        parent_path = target_path[: target_path.rfind(".")] if "." in target_path else ""
        target_name = target_path.split(".")[-1]
        parent = model.get_submodule(parent_path) if parent_path else model
        target = model.get_submodule(target_path)
        if not isinstance(target, torch.nn.Linear):
            continue
        parent.__setattr__(target_name, create_fn(target, name=target_path))



class PiCaLayer(nn.Module):
    def __init__(self, u, s, v, lora_rank=1, weight=None, shared_m=None):

        super().__init__()
        self.in_dim = weight.shape[1]
        self.out_dim = weight.shape[0]

        self.weight = nn.Parameter(weight.detach().clone(), requires_grad=False)
        
       
        self.p = nn.Parameter(v.detach().clone(), requires_grad=False)

        if shared_m is not None:
            self.m = shared_m
        else:
            self.m = nn.Parameter(torch.zeros(u.shape[0], lora_rank))  # (out_dim, lora_rank)

    def forward(self, x):
        x = x @ self.get_weights().T
        return x

    def get_weights(self):
        weight = self.weight + self.m @ self.p 
        return weight

    def merge_and_unload(self):
        return self.get_weights().contiguous()


class LinearWithPiCa(nn.Module):

    def __init__(self, linear, lora_rank=1, shared_m=None):

        super().__init__()
        self.bias = linear.bias

        u, s, v = torch.linalg.svd(linear.weight, full_matrices=False)

        r = len(s)

        self.pica_layer = PiCaLayer(
            u[:, :lora_rank], 
            None, 
            v[:lora_rank, :],
            r=r, 
            lora_rank=lora_rank, 
            weight=linear.weight,
            shared_m=shared_m,  # pass the shared m
        )

    @property
    def weight(self):
        return self.pica_layer.get_weights()
    
    def forward(self, x):
        if self.bias is not None:
            return self.pica_layer(x) + self.bias
        else:
            return self.pica_layer(x)

    def merge_and_unload(self):
        return self.pica_layer.merge_and_unload()



def freeze_model(model, exclude_list = None):
    ''' Freeze all parameters of the model '''
    if exclude_list is None:
        exclude_list = []

    for n, p in model.named_parameters():
        if not any(e in n for e in exclude_list):
            p.requires_grad = False