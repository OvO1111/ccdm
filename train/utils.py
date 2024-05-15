from typing import Callable

import os
import matplotlib.font_manager
import torch
import wandb
import shutil
import importlib
import numpy as np

import matplotlib
import SimpleITK as sitk
import matplotlib.pyplot as plt

from queue import Queue
from tabulate import tabulate
from einops import rearrange
from matplotlib.cm import get_cmap
from torchvision.utils import make_grid


def minmax(val, minimum=None, maximum=None):
    if maximum is not None:
        val = min(val, maximum)
    if minimum is not None:
        val = max(val, minimum)
    return val
    

def default(expr, defeval=None):
    return defeval if expr is None else expr


def identity(inputs, *args, **kwargs):
    return inputs


def maybe_mkdir(d):
    os.makedirs(d, exist_ok=True)
    return d


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def check_loss(loss: torch.Tensor, is_kl=False):
    if torch.isnan(loss).any():
        print("nan found in loss!!")

    if torch.isinf(loss).any():
        print("inf found in loss!!")

    if is_kl:
        if (loss.sum(1) < -1e-3).any():
            print(f"negative KL divergence {loss.sum()} in loss!!")
            

def clip_loss(loss: torch.Tensor, mi=0, mx=100000):
    return torch.clip(loss, mi, mx)
            
        
def recursive_assign(dic:dict, key:str, value, sep='.'):
    split_key = key.split(sep, 1)
    if len(split_key) > 1:
        return recursive_assign(dic[split_key[0]], split_key[1], value, sep)
    original_value = dic[split_key[0]]
    dic[split_key[0]] = value
    return original_value
        

def visualize(image: torch.Tensor, n: int=11, num_images=8):
    if len(image.shape) == 4:
        b, h = image.shape[:2]
        if h > num_images: image = image[:, ::h // num_images]
        image = rearrange(image, "b h w d -> (b h) 1 w d")
    image = make_grid(image, nrow=min(num_images, h), normalize=image.dtype == torch.float32)

    if image.dtype == torch.long:
        cmap = get_cmap("viridis")
        rgb = torch.tensor([(0, 0, 0)] + [cmap(i)[:-1] for i in np.arange(0.3, n) / n], device=image.device)
        colored_mask = rearrange(rgb[image][0], "i j n -> 1 n i j")
        return colored_mask
    else:
        return image
    
    
def print_parameters(**kwargs):
    size = {torch.float32: 4, torch.float16: 2, torch.long: 4, torch.int16: 2}
    for model, param in kwargs.items():
        if param is None:
            continue
        param_count, frozen_param_count = 0., 0.
        param_size = 0.
        for p in param.parameters():
            if p.requires_grad: 
                param_count += torch.numel(p)
                param_size += torch.numel(p) * size.get(p.dtype, torch.nan)
            else: frozen_param_count += torch.numel(p)
        print(f"{model}: {param_count/1e6:.2f}M trainable, {frozen_param_count/1e6:.2f}M frozen, estimated {param_count * 4/1e6:.2f}MB model size")


def make_tabulate_data_from_nested_dict(nested_dict):
    _flatten = dict()
    
    def _trunc(s, l=25):
        s = str(s)
        return s if len(s) < l else s[:l-3] + "..."
    
    def _nested_assign(d, _depth=0, pk=''):
        max_depth = _depth
        for k, v in d.items():
            if isinstance(v, dict):
                max_depth = max(max_depth, _nested_assign(v, _depth + 1, pk + "." + str(k)))
            else: _flatten[pk + "." + str(k)] = _trunc(v)
        return max_depth
        
    prev = ''
    msg = []
    mdp = _nested_assign(nested_dict)
    headers = ["key"]  + [" "] * mdp + ["value"]
    for k, v in _flatten.items():
        
        same_hier = 0 
        prev_split = prev.split('.')
        k_split = k.split(".")
        trunc_k_split = list(map(lambda _: _trunc(_), k_split))
        
        while same_hier < min(len(k_split), len(prev_split)) and k_split[same_hier] == prev_split[same_hier]: same_hier += 1
        msg.append(["--"] * (same_hier - 1) + trunc_k_split[same_hier:] + [" "] * (mdp - len(k_split) + 2) + [v])    
        prev = k
            
    print(tabulate(tabular_data=msg, headers=headers, tablefmt="rounded_outline"))
    

class BasicLogger:
    def __init__(self, logger_name, logger_queue_len=10, logger_base_path=None, **kwargs):
        self.logger = None
        self.kwargs = kwargs
        self.name = logger_name
        self.logger_q = Queue(logger_queue_len)
        self.logger_base = maybe_mkdir(os.path.join(logger_base_path, logger_name))
        
        existent_files = sorted([os.path.join(self.logger_base, n) for n in os.listdir(self.logger_base)],
                                key=lambda x: os.path.getmtime(x))

    @classmethod
    def _image_logger(cls, dict_of_images, path):
        ind_vis = {}
        for k, v in dict_of_images.items():
            if isinstance(v, torch.Tensor): ind_vis[str(k)] = visualize(v).squeeze().data.cpu().numpy()
            elif isinstance(v, str): ind_vis[str(k)] = v
        h = max([getattr(x, "shape", [0, 0, 0])[1] for x in ind_vis.values()])
        w = sum([getattr(x, "shape", [0, 0, 0])[2] for x in ind_vis.values()])
        fig = plt.figure(figsize=(minmax(w // 1024, 15, 30), minmax(h // 1024, 5, 10)), dpi=300)
        for i, (k, v) in enumerate(ind_vis.items()):
            ax = fig.add_subplot(1, len(dict_of_images), i + 1)
            ax.set_title(k)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.yaxis.set_tick_params(labelleft=False)
            if isinstance(v, np.ndarray):
                ax.imshow(rearrange(v, "c h w -> h w c"))
            if isinstance(v, str):
                ax.imshow(np.zeros((5, 10)))
                ax.text(0, 0, "\n".join([v[i * 20: (i + 1) * 20] for i in range(np.ceil(len(v) / 20).astype(int))]),
                        color="white",
                        fontproperties=matplotlib.font_manager.FontProperties(size=5,
                                                                              fname='/ailab/user/dailinrui/data/dependency/Arial-Unicode-Bold.ttf'))
        
        plt.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0)
        image_from_plt = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plt = image_from_plt.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return image_from_plt
        
    def _nifti_logger(self, data, path):
        if not isinstance(data, np.ndarray):
            if isinstance(data, torch.Tensor): 
                data = data.cpu().numpy()
        if data.dtype == np.int64: data = data.astype(np.uint8)
        image = sitk.GetImageFromArray(data)
        image.SetSpacing(self.kwargs.get("spacing", (1, 1, 1)))
        sitk.WriteImage(image, path)
    
    def _model_logger(self, data, path):
        torch.save(data, path)
    
    def log(self, data=None, suffix=None, ext=None):
        path = os.path.join(self.logger_base, suffix)
        if ext is None: ext = '.'.join(suffix.split('.')[1:])
        assert len(ext) > 0 and suffix.endswith(ext)
    
        if self.logger_q.full():
            outdate = self.logger_q.get()
            if os.path.exists(outdate): os.remove(outdate)
        self.logger_q.put(path)
                
        if ext in ["png", "jpg"]:
            self.logger = self._image_logger
        elif ext in ["nii.gz", "nii"]:
            self.logger = self._nifti_logger
        elif ext in ["pt", "ckpt", "pth"]:
            self.logger = self._model_logger
        return self.logger(data, path)

    def __call__(self, data=None, suffix=None, ext=None):
        return self.log(data, suffix, ext) 
        
        
class dummy_context:
    def __enter__(self):
        ...
    
    def __exit__(self, *args):
        pass
    