import sys, os
import torch
import wandb
import torch.distributed
import torch.nn as nn
import torch.nn.functional as f

from omegaconf import OmegaConf
from torch.optim import AdamW, SGD
from train.ccdm import DenoisingModel, DiffusionModel
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

sys.path.append("/mnt/workspace/dailinrui/code/multimodal/trajectory_generation/ccdm")
from train.utils import default, identity, get_obj_from_str, instantiate_from_config, print_parameters
from train.ccdm import CategoricalDiffusionModel, OneHotCategoricalBCHW


def identity(x, *args, **kwargs):
    return x


def default(x, dval=None):
    if not exist(x): return dval
    else: return x

def exist(x):
    return x is None


class LossWrapper(nn.Module):
    def __init__(self, coeff, module):
        self.coeff = coeff
        self.module = module
        
    def __call__(self, *args, **kwargs):
        return self.coeff * self.module(*args, **kwargs)


class CCDM(pl.LightningModule):
    legends = ["background", "spleen", "kidney_left", "kidney_right", "liver", "stomach", "pancreas", "small_bowel",
               "duodenum", "colon", "uniary_bladder", "colorectal_cancer"]
    def __init__(self, diffusion_model_config, denoising_model_config, loss_config,
                 conditional_encoder_config=None,
                 train_ddim_sigmas=False,
                 is_conditional=True,
                 data_key="mask",
                 cond_key="text",
                 timesteps=1000,
                 conditioning_key="crossattn") -> None:
        super().__init__()
        self.data_key = data_key
        self.cond_key = cond_key
        self.timesteps = timesteps
        self.conditioning_key = conditioning_key
        self.is_conditional = is_conditional
        self.train_ddim_sigmas = train_ddim_sigmas
        
        self.loss_fn = dict()
        self.diffusion_model: DiffusionModel = instantiate_from_config(**diffusion_model_config)
        self.denoising_model: DenoisingModel = instantiate_from_config(**denoising_model_config)
        if self.is_conditional:
            if conditional_encoder_config is None: self.condition_encoder = nn.Identity()
            else: self.condition_encoder = instantiate_from_config(**conditional_encoder_config)
            
        if "kl_div" in loss_config:
            config = loss_config["kl_div"]
            self.loss_fn["kl_div"] = LossWrapper(config.get("coeff", 1), 
                                                  get_obj_from_str(config["target"])(**config.get("params", {}), 
                                                                                    attn_weight=torch.tensor(self.train_ds.cls_weight, device=self.device),
                                                                                    diffusion_model=self.diffusion_model))
        if "lpips" in loss_config:
            config = loss_config["lpips"]
            self.loss_fn["lpips"] = LossWrapper(config.get("coeff", 1), 
                                                get_obj_from_str(config["target"])(**config.get("params", {}),
                                                                                   device=self.device))
            
        if "recover_loss" in loss_config:
            config = loss_config["recover_loss"]
            self.loss_fn["recover_loss"] = LossWrapper(config.get("coeff", 1), 
                                                       get_obj_from_str(config["target"])(**config.get("params", {}),
                                                                                        image_in_size=self.denoising_model.unet.fc_in))
        
        print_parameters(model=self)
        self.val_image_buffer = []
     
    def read_spec(self, specs):
        dataset_spec, model_spec, encoder_spec = specs["dataset"], specs['model'], specs['encoder']
        
        self.train_ds = get_cls_from_pkg(dataset_spec["train"])
        self.val_ds = get_cls_from_pkg(dataset_spec["validation"])
        self.test_ds = get_cls_from_pkg(dataset_spec.get("test", dataset_spec["validation"]))
        
        self.x_encoder = default(get_cls_from_pkg(encoder_spec["data_encoder"]), identity)
        self.condition_encoder = default(get_cls_from_pkg(encoder_spec["condition_encoder"]), identity)
        self.context_encoder = default(get_cls_from_pkg(encoder_spec["context_encoder"]), identity)
        
        self.model = get_cls_from_pkg(model_spec,
                                      num_classes=len(self.legends),
                                      num_timesteps=self.timesteps,
                                      spatial_size=self.train_ds.spatial_size[::-1],
                                      condition_channels=getattr(self.condition_encoder, "in_channels", 0))
    
    def get_input(self, batch, *keys):
        ret = []
        for key in keys:
            ret.append(batch.get(key))
        return ret
        
    def training_step(self, batch, batch_idx):
        x0, c = self.get_input(batch, self.data_key, self.cond_key)
        c = {f"c_{self.conditioning_key}": c}
        b, *shp = x0.shape
        t = torch.multinomial(torch.arange(self.timesteps + 1, device=self.device) ** 1.5, b)
        noise = OneHotCategoricalBCHW(logits=torch.zeros_like(x0)).sample()
        xt = self.diffusion_model.q_xt_given_x0(x0, t, noise=noise).sample()
        f = self.model(xt.contiguous(),
                       self.condition_encoder(c.get("c_concat", None)),
                       None, t,
                       context=self.condition_encoder(c.get("c_crossattn", None)))
        x0_ = f["diffusion_out"]
        c_ = f["cond_pred_logits"]
        
        batch_loss = 0.
        loss_log = {}
        if "kl_div" in self.loss_fn:
            loss = self.loss_fn["kl_div"](xt, x0, x0_, t, noise=noise)
            loss_log["train/kl_div_loss"] = loss.item()
            batch_loss += loss
        if "ce" in self.loss_fn:
            loss = self.loss_fn["ce"](c_, batch["class"])
            loss_log["train/ce_loss"] = loss.item()
            batch_loss += loss
            
        self.log_dict(batch_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return batch_loss
    
    @torch.no_grad()
    def log_images_train(self, batch, only_inputs=False, **kwargs):
        x0, c = self.get_input(batch, self.data_key, self.cond_key)
        b = x0.shape[0]
        t = torch.multinomial(torch.arange(self.timesteps + 1, device=self.device) ** 1.5, b)
        noise = OneHotCategoricalBCHW(logits=torch.zeros_like(x0)).sample()
        xt = self.diffusion_model.q_xt_given_x0(x0, t, noise=noise).sample()
        f = self.model(xt.contiguous(),
                       self.condition_encoder(c.get("c_concat", None)),
                       None, t,
                       context=self.condition_encoder(c.get("c_crossattn", None)))
        x0_ = f["diffusion_out"]
        
        image = dict(inputs=x0.argmax(1), 
                     xt=xt.argmax(1), 
                     t=str(t.cpu().numpy().tolist()), 
                     samples=x0_.argmax(1))
        return image
    
    @torch.no_grad()
    def log_images_val(self, batch, **kwargs):
        image = self.val_image_buffer.pop()
        return image
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x0, c = self.get_input(batch, self.data_key, self.cond_key)
        x0_shape = (x0.shape[0], self.diffusion_model.num_classes, *x0.shape[2:])
        xt = OneHotCategoricalBCHW(logits=torch.zeros(x0_shape, device=self.device)).sample()
        # ret = model.ddim_sampling(xt.contiguous(),
        #                         self.condition_encoder(None),
        #                         feature_condition=None,
        #                         context=self.context_encoder(context),
        #                         ref_class=class_id, 
        #                         loss_fn=nn.CrossEntropyLoss(ignore_index=5))
        # x0_pred, x0_denoise, ce_loss = ret
        ret = self.denoising_model(xt.contiguous(), 
                                   self.condition_encoder(c.get("c_concat", None)),
                                   None, None,
                                   context=self.condition_encoder(c.get("c_crossattn", None)))
        x0_ = ret["diffusion_out"]
        
        self.val_image_buffer.append(dict(noise=xt.argmax(1), samples=x0_.argmax(1), text=batch["text"]))
        if "lpips" in self.loss_fn:
            lpips = self.loss_fn(x0_, x0)
            self.log("val/lpips_metric", lpips, prog_bar=True, on_step=True)


if __name__ == "__main__":
    spec = OmegaConf.to_container(OmegaConf.load("./run/train_ruijin_ccdm.yaml"))
            