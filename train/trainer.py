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
from train.ccdm import OneHotCategoricalBCHW


def identity(x, *args, **kwargs):
    return x


def default(x, dval=None):
    if not exist(x): return dval
    else: return x

def exist(x):
    return x is None


class LossWrapper(nn.Module):
    def __init__(self, coeff, module):
        super().__init__()
        self.coeff = coeff
        self.module = module
        
    def __call__(self, *args, **kwargs):
        return self.coeff * self.module(*args, **kwargs)


class CCDM(pl.LightningModule):
    def __init__(self, diffusion_model_config, denoising_model_config, loss_config,
                 conditional_encoder_config=None,
                 train_ddim_sigmas=False,
                 is_conditional=True,
                 data_key="mask",
                 cond_key="context",
                 timesteps=1000,
                 use_scheduler=True,
                 scheduler_config=None,
                 monitor=None,
                 conditioning_key="crossattn") -> None:
        super().__init__()
        self.data_key = data_key
        self.cond_key = cond_key
        self.timesteps = timesteps
        self.conditioning_key = conditioning_key
        self.is_conditional = is_conditional
        self.use_scheduler = use_scheduler
        self.train_ddim_sigmas = train_ddim_sigmas
        if self.use_scheduler:
            self.scheduler_config = scheduler_config
        if monitor is not None:
            self.monitor = monitor
        
        self.loss_fn = dict()
        self.diffusion_model: DiffusionModel = instantiate_from_config(diffusion_model_config, time_steps=timesteps)
        self.denoising_model: DenoisingModel = instantiate_from_config(denoising_model_config, diffusion=self.diffusion_model)
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
                                                get_obj_from_str(config["target"])(**config.get("params", {}),))
        if "recover_loss" in loss_config:
            config = loss_config["recover_loss"]
            self.loss_fn["recover_loss"] = LossWrapper(config.get("coeff", 1), 
                                                       get_obj_from_str(config["target"])(**config.get("params", {}),
                                                                                        image_in_size=self.denoising_model.unet.fc_in))
        if "ce_loss" in loss_config:
            config = loss_config["ce_loss"]
            self.loss_fn["ce_loss"] = LossWrapper(config.get("coeff", 1), nn.CrossEntropyLoss(reduction='mean'))
        if "l1_loss" in loss_config:
            config = loss_config["l1_loss"]
            self.loss_fn["l1_loss"] = LossWrapper(config.get("coeff", 1), nn.L1Loss(reduction='mean'))
        
        print_parameters(model=self)
        self.val_image_buffer = []
     
    def get_input(self, batch, data_key, cond_key):
        x = batch.get(data_key)
        c = self.condition_encoder(batch.get(cond_key))
        c = {f"c_{self.conditioning_key}": c}
        ret = [x, c]
        return ret
        
    def training_step(self, batch, batch_idx):
        x0, c = self.get_input(batch, self.data_key, self.cond_key)
        b, *shp = x0.shape
        t = torch.multinomial(torch.arange(self.timesteps + 1, device=self.device) ** 1.5, b)
        noise = OneHotCategoricalBCHW(logits=torch.zeros_like(x0)).sample()
        xt = self.diffusion_model.q_xt_given_x0(x0, t, noise=noise).sample()
        f = self.denoising_model(xt.contiguous(),
                                 c.get("c_concat"),
                                 None, t,
                                 context=c.get("c_crossattn", None))
        x0_ = f["diffusion_out"]
        c_ = f["cond_pred_logits"]
        
        batch_loss = 0.
        loss_log = {}
        if "kl_div" in self.loss_fn:
            loss = self.loss_fn["kl_div"](xt, x0, x0_, t, noise=noise)
            loss_log["train/kl_div_loss"] = loss.item()
            batch_loss += loss
        if "ce_loss" in self.loss_fn:
            loss = self.loss_fn["ce_loss"](c_, batch["class"])
            loss_log["train/ce_loss"] = loss.item()
            batch_loss += loss
        if "l1_loss" in self.loss_fn:
            loss = self.loss_fn["l1_loss"](x0_, x0)
            loss_log["train/l1_loss"] = loss.item()
            batch_loss += loss
            
        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)
            
        self.log_dict(loss_log, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return batch_loss
    
    @torch.no_grad()
    def log_images_train(self, batch, only_inputs=False, **kwargs):
        x0, c = self.get_input(batch, self.data_key, self.cond_key)
        b = x0.shape[0]
        t = torch.multinomial(torch.arange(self.timesteps + 1, device=self.device) ** 1.5, b)
        noise = OneHotCategoricalBCHW(logits=torch.zeros_like(x0)).sample()
        xt = self.diffusion_model.q_xt_given_x0(x0, t, noise=noise).sample()
        f = self.denoising_model(xt.contiguous(),
                                c.get("c_concat", None),
                                None, t,
                                context=c.get("c_crossattn", None),
                                is_logging_image=True)
        x0_ = f["diffusion_out"]
        
        image = dict(inputs=x0.argmax(1), 
                     xt=xt.argmax(1), 
                     t=str(t.cpu().numpy().tolist()), 
                     samples=x0_.argmax(1))
        return image
    
    @torch.no_grad()
    def log_images_val(self, batch, ddim_steps=None, **kwargs):
        if not ddim_steps:
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
            
    def configure_optimizers(self):
        lr = self.learning_rate
        opt = torch.optim.AdamW(list(self.denoising_model.parameters()), lr=lr)
        if self.use_scheduler:
            scheduler = instantiate_from_config(self.scheduler_config)
            sch = [
                {
                    'scheduler': torch.optim.lr_scheduler.LambdaLR(opt, scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], sch
        return opt


if __name__ == "__main__":
    spec = OmegaConf.to_container(OmegaConf.load("./run/train_ruijin_ccdm.yaml"))
            