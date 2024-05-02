import sys, os
sys.path.append("/mnt/workspace/dailinrui/code/multimodal/trajectory_generation/ccdm")
import torch
import torch.nn as nn
import torch.nn.functional as f

from tqdm import tqdm
from train.utils import check_loss, get_cls_from_pkg, dummy_context, clip_loss


class CrossEntropyLoss(nn.Module):
    def __init__(self, ignore_class=-100):
        super().__init__()
        self.ignore_class = ignore_class
    
    def forward(self, c_pred, c, **kwargs):
        loss = f.cross_entropy(c_pred, c, reduction="none", ignore_index=self.ignore_class)
        check_loss(loss)
        loss = loss.sum()
        return clip_loss(loss)


class DiffusionKLLoss(nn.Module):
    def __init__(self, attn_weight, diffusion_model):
        super().__init__()
        self.diffusion_model = diffusion_model
        self.attn_weight = attn_weight
    
    def forward(self, xt, x0, x0_pred, t, **kwargs):
        prob_xtm1_given_xt_x0 = self.diffusion_model.theta_post(xt, x0, t)
        prob_xtm1_given_xt_x0pred = self.diffusion_model.theta_post_prob(xt, x0_pred, t)
        
        loss = nn.functional.kl_div(
            torch.log(torch.clamp(prob_xtm1_given_xt_x0pred, min=1e-12)),
            prob_xtm1_given_xt_x0,
            reduction='none'
        )
        loss = loss.sum(dim=1) * self.attn_weight[x0.argmax(1)]
        check_loss(loss, is_kl=True)
        loss = loss.sum()
        return clip_loss(loss)
    

class DiceLoss(nn.Module):
    def __init__(self, n_classes=-1, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.n_classes = n_classes
        
    def _set_class_num(self, num):
        self.n_classes = num

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, inputs, target, mask=None):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(inputs * target * mask)
        y_sum = torch.sum(target * target * mask)
        z_sum = torch.sum(inputs * inputs * mask)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, mask=None, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        self.n_classes = inputs.size(1)
        if target.ndim != inputs.ndim:
            target = target.unsqueeze(1)
            target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        mask = inputs != self.ignore_index
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i], mask[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes
    
    
class LPIPS(nn.Module):
    backbones = {"lpips": {"target": "monai.networks.nets.dynunet.DynUNet",
                           "params": {"filters": (16, 32, 64, 128, 256),
                                      "strides": (1, 2, 2, 2, 2),
                                      "kernel_size": (3, 3, 3, 3, 3),
                                      "upsample_kernel_size": (2, 2, 2, 2,)},
                           "lpips": {"n_layers": (16, 32, 64, 128, 256)}}
                }
    def __init__(self, 
                 in_channels,
                 out_channels,
                 ndim=3,
                 model_path="", 
                 net_backbone=None,
                 is_training=False,
                 use_linear=True,
                 spatial_average=False):
        super().__init__()
        self.ndim = ndim
        self.model_path = model_path
        self.use_linear = use_linear
        self.is_training = is_training
        self.spatial_average = spatial_average
        
        net_backbone = self.backbones.get(net_backbone, self.backbones["lpips"])
        self.perceptual_net = get_cls_from_pkg(net_backbone,
                                               in_channels=in_channels, out_channels=out_channels, spatial_dims=ndim,
                                               deep_supervision=True, deep_supr_num=len(net_backbone["lpips"]["n_layers"]) - 2)
        if os.path.exists(model_path) and os.path.isfile(model_path):
            self.perceptual_net.load_state_dict(torch.load(model_path, map_location="cpu")["perceptual_backbone"], strict=False)
        
        if not self.is_training: self.eval()
        if self.use_linear: self.linear_layers = self.get_lin_layer(net_backbone["lpips"]["n_layers"])
    
    def get_lin_layer(self, in_channel, out_channel=1, dropout=0.):
        n = len(in_channel) - 1
        linear_layers = nn.ModuleList()
        in_channel = (1,) + in_channel[1:-1]
        conv_nd = nn.Conv3d if self.ndim == 3 else nn.Conv2d if self.ndim == 2 else nn.Conv1d
        dropout_nd = nn.Dropout3d if self.ndim == 3 else nn.Dropout2d if self.ndim == 2 else nn.Dropout1d
        for i in range(n):
            layer = nn.Module()
            layer.dropout = dropout_nd(dropout) if dropout > 0 else nn.Identity()
            layer.conv = conv_nd(in_channel[i], out_channel, 1)
            linear_layers.append(layer)
        return linear_layers
    
    @staticmethod
    def tensor_normalize(tensor):
        return tensor / ((tensor ** 2).sum(dim=1, keepdim=True) + 1e-8)

    def tensor_average(self, tensor, size=None):
        b, c, *shp = tensor.shape
        if not self.spatial_average: return tensor.mean(dim=[i for i in range(2, 2 + len(shp))], keepdim=True)
        else: return nn.Upsample(size, mode="bilinear", align_corners=False)(tensor)
        
    def forward(self, x0, x0_pred, is_training=False, **kwargs):
        lpips = []
        b, c, *shp = x0.shape
        is_training = is_training & self.is_training
        x0, x0_pred = x0.argmax(1, keepdim=True) / 255., x0_pred.argmax(1, keepdim=True) / 255.
        with torch.no_grad():
            i_embed, t_embed = self.perceptual_net(x0.float()), self.perceptual_net(x0_pred.float())

        with dummy_context() if is_training else torch.no_grad():
            for k in range(len(i_embed)):
                diff = (self.tensor_normalize(i_embed[k]) - self.tensor_normalize(t_embed[k])) ** 2 / torch.numel(i_embed[k])
                if self.use_linear:
                    diff = self.linear_layers[k].conv(self.linear_layers[k].dropout(diff))
                else: diff = diff.sum(dim=1, keepdim=True)
                diff = self.tensor_average(diff, size=shp)
                lpips.append(diff)
                
        lpips_metric = sum(lpips).sum()
        return lpips_metric
    
    def train(self, max_ep):
        from torch.optim import SGD
        from torch.utils.data import DataLoader
        from segmentation.colon_segmentation import ColonSegmentation
        
        trainset, valset = ColonSegmentation(patch=(64, 128, 128)), ColonSegmentation("val", patch=(64, 128, 128))
        trainloader, valloader = DataLoader(trainset, 12, shuffle=True, pin_memory=True, num_workers=16, persistent_workers=True), DataLoader(valset, 1, pin_memory=True, num_workers=1, persistent_workers=True)
        
        lr = 1e-3
        self.perceptual_net = self.perceptual_net.cuda()
        optimizer = SGD(self.perceptual_net.parameters(), lr=lr)
        
        best_loss = 1000.
        
        for _ in range(max_ep):
            iterator = tqdm(trainloader, desc=f"epoch {_} training progress")
            for i, batch in enumerate(iterator):
                data = batch["totalseg"].cuda() / 255.
                output = self.perceptual_net(data)
                output = output[0]
                
                optimizer.zero_grad()
                loss = ((output - data) ** 2).mean()
                loss.backward()
                optimizer.step()
                
                iterator.set_postfix(loss=f"{loss.item():.4f}", itr=_ * len(trainloader) + i)
                
            lr_ = lr * (1.0 - _ / max_ep)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            
            mean_val_loss = 0.
            for i in tqdm(valloader, desc=f"epoch {_} validation progress"):
                data = batch["totalseg"].cuda() / 255.
                data = self.perceptual_net.eval()(data)
                output = data[0]
                mean_val_loss += ((output - data) ** 2).mean().item()
            mean_val_loss /= len(valloader)
            
            if mean_val_loss < best_loss: 
                torch.save({"perceptual_backbone": self.perceptual_net.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "best_val_loss": best_loss}, "/mnt/workspace/dailinrui/data/pretrained/ccdm/lpips_backbone_best.ckpt")
                best_loss = mean_val_loss
            print(f"epoch {_} val mean loss {mean_val_loss:.4f}" + f"best loss {best_loss:.4f}")
            
            
if __name__ == "__main__":
    lpips = LPIPS(1, 1, is_training=True)
    lpips.train(1000)
    