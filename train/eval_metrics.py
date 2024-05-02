import torch
import torch.nn as nn
import numpy as np

from einops import rearrange, repeat
from train.metrics.lpips import LPIPS
from train.metrics.fvd import Embedder, compute_fvd
from monai.networks.nets.unet import UNet
from scipy.ndimage import binary_dilation, distance_transform_edt


class pretrained_metrics:    
    def compute(self, x, y):
        raise NotImplementedError()
    
    def __call__(self, pred, gt):
        return self.compute(pred, gt)


class pretrained_lpips(pretrained_metrics):
    def __init__(self, device=0):
        super().__init__()
        self.net = LPIPS().to(device)
        
    def compute(self, x, y):
        b, c, *shp = x.shape
        perceptual = self.net.eval()
        x, y = repeat(x.argmax(1), 'b d h w -> b c d h w', c=3), repeat(y.argmax(1), 'b d h w -> b c d h w', c=3)
        if len(shp) == 3:
            lpips_x = perceptual(rearrange(x, "b c d h w -> (b d) c h w"),
                                rearrange(y, "b c d h w -> (b d) c h w")).mean()
            lpips_y = perceptual(rearrange(x, "b c d h w -> (b h) c d w"),
                                rearrange(y, "b c d h w -> (b h) c d w")).mean()
            lpips_z = perceptual(rearrange(x, "b c d h w -> (b w) c d h"),
                                rearrange(y, "b c d h w -> (b w) c d h")).mean()
            lpips = (lpips_x + lpips_y + lpips_z) / 3
        elif len(shp) == 2:
            lpips = perceptual(x, y)
        return lpips.item()


class pretrained_fvd(pretrained_metrics):
    def __init__(self, bs=8, device=0):
        super().__init__()
        self.bs = bs
        self.xs = []
        self.ys = []
        self.device = device
        self.embedder_real = Embedder()
        self.embedder_fake = Embedder()
        
    def compute(self, x, y):
        fvd = 0
        x, y = repeat(x.argmax(1), 'b d h w -> b c d h w', c=3), repeat(y.argmax(1), 'b d h w -> b c d h w', c=3)
        if len(self.xs) < self.bs:
            self.xs.append(x)
            self.ys.append(y)
        if len(self.xs) == self.bs:
            fvd = self._compute(torch.cat(self.xs, dim=0), torch.cat(self.ys, dim=0))
            self.xs = []
            self.ys = []
        return fvd
        
    def _compute(self, x, y):
        b, c, *shp = x.shape
        assert b > 1 and len(shp) == 3
        fvd = compute_fvd(self.embedder_real, self.embedder_fake, self.device,
                          rearrange(x, "b c d h w -> b d h w c").repeat(1, 1, 1, 1, 3),
                          rearrange(y, "b c d h w -> b d h w c").repeat(1, 1, 1, 1, 3))
        return fvd
    

class pretrained_fid(pretrained_metrics):
    def __init__(self):
        super().__init__()
        pass


class pretrained_seg(pretrained_metrics):
    def __init__(self, device=0, fn="distribution"):
        super().__init__()
        self.net = UNet(spatial_dims=3,
                        in_channels=1,
                        out_channels=6,
                        channels=(16, 32, 64, 128, 256),
                        strides=(2, 2, 2, 2),).to(device)
        self.fn = fn
        self.segment_x = None
        self.net.load_state_dict(torch.load("/mnt/workspace/dailinrui/data/pretrained/ccdm/segmentation/colon_segment/model/best_model_dice=0.79.ckpt", map_location="cpu")["model"])
    
    def compute(self, x, y):
        b, c, *shp = x.shape
        x, y = repeat(x.argmax(1), 'b d h w -> b c d h w', c=1), repeat(y.argmax(1), 'b d h w -> b c d h w', c=1)
        
        cx = (x == 9).float()
        tx = (x == 11).float().cpu().numpy()
        pred_cx = self.net(cx).argmax(1).cpu().numpy()
        self.segment_x = pred_cx.astype(np.uint8)
        cx = cx.cpu().numpy() > 0
        
        dx = torch.zeros((b, 5,))
        for ib in range(b):
            edt = distance_transform_edt(tx[ib, 0] == 0)
            dx[ib] = torch.tensor(np.array([np.mean(edt[pred_cx[ib] == i]) for i in range(1, 6)]))
        
        cy = (y == 9).float()
        ty = (y == 11).float().cpu().numpy()
        pred_cy = self.net(cy).argmax(1).cpu().numpy()
        cy = cy.cpu().numpy() > 0
        
        dy = torch.zeros((b, 5,))
        for ib in range(b):
            edt = distance_transform_edt(ty[ib, 0] == 0)
            dy[ib] = torch.tensor(np.array([np.mean(edt[pred_cy[ib] == i]) for i in range(1, 6)]))
        
        if self.fn == "distribution": dxy = nn.functional.kl_div(torch.log_softmax(dx, dim=1), torch.log_softmax(dy, dim=1), log_target=True, reduction="batchmean")
        elif self.fn == "argmax": dxy = nn.functional.cross_entropy(-dx, dy.argmin(1))
        return dxy.item()