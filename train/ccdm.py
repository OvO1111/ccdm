
from typing import Optional, Tuple, cast, Union
import logging
import math

import torch
from torch import Tensor
from torch import nn
import numpy as np
from tqdm import tqdm

from train.utils import instantiate_from_config
from torch.distributions import OneHotCategorical

LOGGER = logging.getLogger(__name__)

__all__ = ["DiffusionModel", "DenoisingModel"]


def linear_schedule(time_steps: int, start=1e-2, end=0.2) -> Tuple[Tensor, Tensor, Tensor]:
    betas = torch.linspace(start, end, time_steps)
    alphas = 1 - betas
    cumalphas = torch.cumprod(alphas, dim=0)
    cumalphas_prev = torch.cat([torch.tensor((1,)), cumalphas[:-1]])
    return betas, alphas, cumalphas, cumalphas_prev


def cosine_schedule(time_steps: int, s: float = 8e-3) -> Tuple[Tensor, Tensor, Tensor]:
    t = torch.arange(0, time_steps)
    s = 0.008
    cumalphas = torch.cos(((t / time_steps + s) / (1 + s)) * (math.pi / 2)) ** 2
    cumalphas_prev = torch.cat([torch.tensor((1,)), cumalphas[:-1]])

    def func(t): return math.cos((t + s) / (1.0 + s) * math.pi / 2) ** 2

    betas_ = []
    for i in range(time_steps):
        t1 = i / time_steps
        t2 = (i + 1) / time_steps
        betas_.append(min(1 - func(t2) / func(t1), 0.999))
    betas = torch.tensor(betas_)
    alphas = 1 - betas
    return betas, alphas, cumalphas, cumalphas_prev


def make_ddim_timesteps(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps, verbose=True):
    if ddim_discr_method == 'uniform':
        c = num_ddpm_timesteps // num_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == 'quad':
        ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * .8), num_ddim_timesteps)) ** 2).astype(int)
    else:
        raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')

    # assert ddim_timesteps.shape[0] == num_ddim_timesteps
    # add one to get the final alpha values right (the ones from first scale to data during sampling)
    steps_out = ddim_timesteps + 1
    if verbose:
        print(f'Selected timesteps for ddim sampler: {steps_out}')
    return steps_out


def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta, verbose=True):
    # select alphas for computing the variance schedule
    alphas = alphacums[ddim_timesteps]
    alphas_prev = torch.tensor(np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist()))

    # according the the formula provided in https://arxiv.org/abs/2010.02502
    sigmas = eta * np.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
    if verbose:
        print(f'Selected alphas for ddim sampler: a_t: {alphas}; a_(t-1): {alphas_prev}')
        print(f'For the chosen value of eta, which is {eta}, '
              f'this results in the following sigma_t schedule for ddim sampler {sigmas}')
    return sigmas, alphas, alphas_prev


class OneHotCategoricalBCHW(OneHotCategorical):
    """Like OneHotCategorical, but the probabilities are along dim=1."""

    def __init__(
            self,
            probs: Optional[torch.Tensor] = None,
            logits: Optional[torch.Tensor] = None,
            validate_args=None):

        if probs is not None and probs.ndim < 2:
            raise ValueError("`probs.ndim` should be at least 2")

        if logits is not None and logits.ndim < 2:
            raise ValueError("`logits.ndim` should be at least 2")

        probs = self.channels_last(probs) if probs is not None else None
        logits = self.channels_last(logits) if logits is not None else None

        super().__init__(probs, logits, validate_args)

    def sample(self, sample_shape=torch.Size()):
        res = super().sample(sample_shape)
        return self.channels_second(res)

    @staticmethod
    def channels_last(arr: torch.Tensor) -> torch.Tensor:
        """Move the channel dimension from dim=1 to dim=-1"""
        dim_order = (0,) + tuple(range(2, arr.ndim)) + (1,)
        return arr.permute(dim_order)

    @staticmethod
    def channels_second(arr: torch.Tensor) -> torch.Tensor:
        """Move the channel dimension from dim=-1 to dim=1"""
        dim_order = (0, arr.ndim - 1) + tuple(range(1, arr.ndim - 1))
        return arr.permute(dim_order)

    def max_prob_sample(self):
        """Sample with maximum probability"""
        num_classes = self.probs.shape[-1]
        res = torch.nn.functional.one_hot(self.probs.argmax(dim=-1), num_classes)
        return self.channels_second(res)

    def prob_sample(self):
        """Sample with probabilities"""
        return self.channels_second(self.probs)


class DiffusionModel(nn.Module):
    betas: Tensor
    alphas: Tensor
    cumalphas: Tensor
    cumalphas_prev: Tensor

    def __init__(self, schedule: str, time_steps: int, num_classes: int, schedule_params=None, dims=3):
        super().__init__()

        schedule_func = {
            "linear": linear_schedule,
            "cosine": cosine_schedule
        }[schedule]
        if schedule_params is not None:
            LOGGER.info(f"noise schedule '{schedule}' with params {schedule_params} with time steps={time_steps}")
            betas, alphas, cumalphas, cumalphas_prev = schedule_func(time_steps, **schedule_params)
        else:
            LOGGER.info(f"noise schedule '{schedule}' with default params (schedule_params = {schedule_params})"
                        f" with time steps={time_steps}")
            betas, alphas, cumalphas, cumalphas_prev = schedule_func(time_steps)

        self.dims = dims
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("cumalphas", cumalphas)
        self.register_buffer("cumalphas_prev", cumalphas_prev)

        self.num_classes = num_classes

    @property
    def time_steps(self):
        return len(self.betas)

    def q_xt_given_xtm1(self, xtm1: Tensor, t: Tensor, noise: Tensor=None) -> OneHotCategoricalBCHW:
        t = t - 1
        betas = self.betas[t]
        betas = betas[..., None, None, None]
        if self.dims == 3: betas = betas[..., None]
        if noise is None: probs = (1 - betas) * xtm1 + betas / self.num_classes
        else: probs = (1 - betas) * xtm1 + betas * noise
        return OneHotCategoricalBCHW(probs)

    def q_xt_given_x0(self, x0: Tensor, t: Tensor, noise: Tensor=None) -> OneHotCategoricalBCHW:
        t = t - 1
        cumalphas = self.cumalphas[t]
        cumalphas = cumalphas[..., None, None, None]
        if self.dims == 3: cumalphas = cumalphas[..., None]
        if noise is None: probs = cumalphas * x0 + (1 - cumalphas) / self.num_classes
        else: probs = cumalphas * x0 + (1 - cumalphas) * noise
        return OneHotCategoricalBCHW(probs)
    
    def theta_post_v2(self, xt: Tensor, x0: Tensor, t: Tensor, noise: Tensor=None) -> Tensor:
        # computes q_xtm1 given q_xt, q_x0, by setting gamma=0: q=\alpha e_xt+\beta e_x0
        t = t - 1
        cumalphas_t = self.cumalphas[t]
        cumalphas_tm1 = self.cumalphas[t - 1]
        
        cumalphas_t[t == 0] = 1.
        cumalphas_tm1[t == 0] = 1.
        alpha = (1 - cumalphas_tm1) / (1 - cumalphas_t)[..., None, None, None]
        beta = (cumalphas_tm1 - cumalphas_t) / ((1 - cumalphas_t) * (1 + (self.num_classes - 1) * cumalphas_t))[..., None, None, None]
        if self.dims == 3:
            alpha, beta = alpha[..., None], beta[..., None]
            
        theta = alpha * xt + beta * x0
        return theta / theta.sum(dim=1, keepdim=True)

    def theta_post(self, xt: Tensor, x0: Tensor, t: Tensor, noise: Tensor=None) -> Tensor:
        # computes q_xtm1 given q_xt, q_x0, noise
        t = t - 1
        smooth = 1e-8
        alphas_t = self.alphas[t][..., None, None, None]
        cumalphas_tm1 = self.cumalphas[t - 1][..., None, None, None]
        if self.dims == 3:
            alphas_t = alphas_t[..., None]
            cumalphas_tm1 = cumalphas_tm1[..., None]
        alphas_t[t == 0] = 0.0
        cumalphas_tm1[t == 0] = 1.0
        if noise is None: theta = ((alphas_t * xt + (1 - alphas_t) / self.num_classes) * (cumalphas_tm1 * x0 + (1 - cumalphas_tm1) / self.num_classes))
        else: theta = ((alphas_t * xt + (1 - alphas_t) * noise) * (cumalphas_tm1 * x0 + (1 - cumalphas_tm1) * noise))
        return (theta + smooth) / (theta.sum(dim=1, keepdim=True) + smooth)

    def theta_post_prob(self, xt: Tensor, theta_x0: Tensor, t: Tensor, noise: Tensor=None) -> Tensor:
        """
        This is equivalent to calling theta_post with all possible values of x0
        from 0 to C-1 and multiplying each answer times theta_x0[:, c].

        This should be used when x0 is unknown and what you have is a probability
        distribution over x0. If x0 is one-hot encoded (i.e., only 0's and 1's),
        use theta_post instead.
        """
        t = t - 1
        smooth = 1e-8
        alphas_t = self.alphas[t][..., None, None, None]
        cumalphas_tm1 = self.cumalphas[t - 1][..., None, None, None, None]
        if self.dims == 3:
            alphas_t = alphas_t[..., None]
            cumalphas_tm1 = cumalphas_tm1[..., None]
        alphas_t[t == 0] = 0.0
        cumalphas_tm1[t == 0] = 1.0

        # We need to evaluate theta_post for all values of x0
        x0 = torch.eye(self.num_classes, device=xt.device)[None, :, :, None, None]
        if self.dims == 3: x0 = x0[..., None]
        # theta_xt_xtm1.shape == [B, C, H, W]
        # theta_xtm1_x0.shape == [B, C1, C2, H, W]
        if noise is None: 
            theta_xt_xtm1 = alphas_t * xt + (1 - alphas_t) / self.num_classes
            theta_xtm1_x0 = cumalphas_tm1 * x0 + (1 - cumalphas_tm1) / self.num_classes
            
            aux = theta_xt_xtm1[:, :, None] * theta_xtm1_x0
            # theta_xtm1_xtx0 == [B, C1, C2, H, W]
            theta_xtm1_xtx0 = (aux + smooth) / (aux.sum(dim=1, keepdim=True) + smooth)
            
            # theta_x0.shape = [B, C, H, W]
            out = torch.einsum("bcdlhw,bdlhw->bclhw", theta_xtm1_xtx0, theta_x0) if self.dims == 3 else\
                torch.einsum("bcdhw,bdhw->bchw", theta_xtm1_xtx0, theta_x0)
            
        else:
            theta_xt_xtm1 = alphas_t * xt + (1 - alphas_t) * noise
            theta_xtm1_x0 = cumalphas_tm1 * x0 + (1 - cumalphas_tm1) * noise

            aux = theta_xt_xtm1[:, :, None] * theta_xtm1_x0
            # theta_xtm1_xtx0 == [B, C1, C2, H, W]
            theta_xtm1_xtx0 = aux / (aux.sum(dim=1, keepdim=True) + smooth)
            
            # theta_x0.shape = [B, C, H, W]
            out = torch.einsum("bcdlhw,bdlhw->bclhw", theta_xtm1_xtx0, theta_x0) if self.dims == 3 else\
                torch.einsum("bcdhw,bdhw->bchw", theta_xtm1_xtx0, theta_x0)
            out = out / out.sum(1, keepdim=True)
            
        return out


class DenoisingModel(nn.Module):

    def __init__(self, diffusion: DiffusionModel, unet_config: dict=None, unet: nn.Module=None, step_T_sample:str = "majority"):
        super().__init__()
        self.diffusion = diffusion
        self.unet = unet
        if self.unet is None:
            assert unet_config is not None
            self.unet = instantiate_from_config(**unet_config)
        self.step_T_sample = step_T_sample

    @property
    def time_steps(self):
        return self.diffusion.time_steps

    def forward(self, x: Tensor, condition: Tensor, feature_condition: Tensor = None, t: Optional[Tensor] = None, label_ref_logits: Optional[Tensor] = None,
                validation: bool = False, context=None) -> Union[Tensor, dict]:

        if self.training:
            if not isinstance(t, Tensor):
                raise ValueError("'t' needs to be a Tensor at training time")
            if not isinstance(x, Tensor):
                raise ValueError("'x' needs to be a Tensor at training time")
            return self.forward_step(x, condition, feature_condition, t, context=context) 
        else:
            if validation:
                return self.forward_step(x, condition, feature_condition, t, context=context)
            if t is None:
                return self.forward_denoising(x, condition, feature_condition, label_ref_logits=label_ref_logits, context=context)

            return self.forward_denoising(x, condition, feature_condition, cast(int, t.item()), label_ref_logits, context=context)

    def forward_step(self, x: Tensor, condition: Tensor, feature_condition: Tensor, t: Tensor, context: Tensor=None) -> Tensor:
        return self.unet(x, condition, feature_condition=feature_condition, timesteps=t, context=context)

    def forward_denoising(self, x: Optional[Tensor], condition: Tensor, feature_condition: Tensor, init_t: Optional[int] = None,
                          label_ref_logits: Optional[Tensor] = None, context: Tensor=None) -> dict:
        if init_t is None:
            init_t = self.time_steps

        xt = x
        if label_ref_logits is not None:
            weights = self.guidance_scale_weights(label_ref_logits)
            label_ref = label_ref_logits.argmax(dim=1)

        shape = xt.shape
        if init_t > 10000:
            K = init_t % 10000
            assert 0 < K <= self.time_steps
            if K == self.time_steps:
                t_values = range(K, 0, -1)
            else:
                t_values = [round(t_val) for t_val in np.linspace(self.time_steps, 1, K)]
                LOGGER.warning(f"Override default {self.time_steps} time steps with {len(t_values)}.")
        else:
            t_values = range(init_t, 0, -1)
            
        for t in t_values:
            # Auxiliary values
            t_ = torch.full(size=(shape[0],), fill_value=t, device=xt.device)

            # Predict the noise of x_t
            ret = self.unet(xt, condition, feature_condition, t_.float(), context=context)
            x0pred = ret["diffusion_out"]
            probs = self.diffusion.theta_post_prob(xt, x0pred, t_)
            if label_ref_logits is not None:
                if self.guidance_scale > 0:
                    gradients = self.guidance_fn(probs, label_ref if self.guidance_loss_fn_name == 'CE' else label_ref_logits, weights)
                    probs = probs - gradients
            probs = torch.clamp(probs, min=1e-12)

            if t > 1:
                xt = OneHotCategoricalBCHW(probs=probs).sample()
            else:
                if self.step_T_sample is None or self.step_T_sample == "majority":
                    xt = OneHotCategoricalBCHW(probs=probs).max_prob_sample()
                elif self.step_T_sample == "confidence":
                    xt = OneHotCategoricalBCHW(probs=probs).prob_sample()

        ret = {"diffusion_out": xt}
        return ret