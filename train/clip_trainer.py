import sys, os
sys.path.append("/mnt/workspace/dailinrui/code/multimodal/trajectory_generation/ccdm")

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import AdamW, SGD
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf
from accelerate import Accelerator
from torch.optim.lr_scheduler import LinearLR
from tensorboardX import SummaryWriter, GlobalSummaryWriter
from train.utils import default, identity, maybe_mkdir, get_cls_from_pkg, visualize, print_parameters, log_images

def norm(enc, smooth=1e-6):
    return enc / (enc.norm(dim=1, keepdim=True) + smooth)


class CLIPModel(nn.Module):
    def __init__(self, image_encoder_spec, text_encoder_spec, 
                 embed_dim=64, spatial_size=None, text_in_channels=None):
        super().__init__()
        self.image_encoder = get_cls_from_pkg(image_encoder_spec, spatial_size=spatial_size)
        self.text_encoder = get_cls_from_pkg(text_encoder_spec, in_channels=text_in_channels)
        
        self.image_proj = nn.Linear(self.image_encoder.last_layer_spatial_size, embed_dim)
        self.text_proj = nn.Linear(self.image_encoder.context_dim, embed_dim)
        self.temperature = nn.Parameter(torch.tensor((1.,)), requires_grad=True)

    def forward(self, i, t, context=None):
        image_enc = self.image_encoder(i, context=context)["encode_outputs"]
        text_enc = self.text_encoder(t)
        b = image_enc.shape[0]
        image_enc = norm(self.image_proj(image_enc.contiguous().view(b, -1)))  # b d
        text_enc = norm(self.text_proj(text_enc.contiguous().view(b, -1)))  # b d
        
        logits = torch.einsum("ij,kj->ik", image_enc, text_enc).contiguous() * torch.exp(self.temperature)  # b b
        target = torch.arange(b).to(logits.device)
        
        loss_i = nn.functional.cross_entropy(logits, target)
        loss_t = nn.functional.cross_entropy(logits.T, target)
        loss = (loss_i + loss_t) / 2
        return loss


class Trainer:
    def __init__(self, spec, *, 
                 val_device=None,
                 batch_size=4,
                 lr=1e-3,
                 max_epochs=100,
                 timesteps=1000,
                 snapshot_path=None,
                 restore_path=None,
                 parallel_validation=False) -> None:
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.timesteps = timesteps
        self.parallel_val = parallel_validation
        self.snapshot_path = maybe_mkdir(snapshot_path)
        
        self.best = 1e6
        self.spec = spec
        self.read_spec(spec)
        
        self.model_path = maybe_mkdir(os.path.join(self.snapshot_path, "model"))
        self.tensorboard_path = maybe_mkdir(os.path.join(self.snapshot_path, "logs"))
        self.visualization_path = maybe_mkdir(os.path.join(self.snapshot_path, "visual"))

        self.val_proc = None
        
        self.train_dl = DataLoader(self.train_ds,
                                   self.batch_size,
                                   shuffle=True, pin_memory=True, num_workers=8, collate_fn=self.train_ds.collate_fn)
        self.val_dl = DataLoader(self.val_ds,
                                 batch_size=1,
                                 pin_memory=True,
                                 num_workers=2, collate_fn=self.val_ds.collate_fn)
        
        self.optimizer = AdamW(self.model.parameters(), self.lr / self.batch_size)
        self.lr_scheduler = LinearLR(self.optimizer, start_factor=1, total_iters=self.max_epochs)
        if restore_path is not None: self.model.load_state_dict(torch.load(restore_path, map_location='cpu'))
        
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.val_device = torch.device("cuda", int(val_device)) if val_device is not None else self.device
        self.model, self.optimizer, self.train_dl, self.val_dl, self.lr_scheduler =\
            self.accelerator.prepare(self.model, self.optimizer, self.train_dl, self.val_dl, self.lr_scheduler)

        print_parameters(model=self.model)
        self.writer = SummaryWriter(getattr(self, "tensorboard_path",
                                                  "/mnt/workspace/dailinrui/data/pretrained/ccdm/leftover_collect"))
        
    def read_spec(self, specs):
        dataset_spec, model_spec = specs["dataset"], specs['model']
        
        self.train_ds = get_cls_from_pkg(dataset_spec["train"])
        self.val_ds = get_cls_from_pkg(dataset_spec["validation"])
        
        self.model = default(get_cls_from_pkg(model_spec,
                                              spatial_size=self.train_ds.spatial_size[::-1],
                                              text_in_channels=self.train_ds.collate_maxlen), identity)
        
    def train(self):
        self.model.train()
        if self.parallel_val: train_it = tqdm(range(self.max_epochs * len(self.train_dl)), desc="train progress")
        else: train_it = tqdm(range(self.max_epochs * (len(self.train_dl) + len(self.val_dl))), desc="train progress")
        
        for self.epoch in range(self.max_epochs):
            self._train(train_it)
            self.lr_scheduler.step()
            
            self.accelerator.wait_for_everyone()
            if self.epoch % 1 == 0 and self.accelerator.is_main_process:
                if self.parallel_val:
                    if self.val_proc is not None: 
                        try:
                            self.val_proc.join()
                        except Exception as e: print(e)  # noqa: may report error when parallel validating because tensorboard is not closed properly
                    self.val_proc = torch.multiprocessing.get_context("spawn").Process(target=self.val, args=(0, self.model.state_dict()))
                    self.val_proc.start()
                else:
                    self.val(train_it=train_it)
                    
        self._save(os.path.join(self.model_path, "last.ckpt"),
                   model=self.accelerator.unwrap_model(self.model).state_dict())
        self.accelerator.end_training()
    
    def _train(self, train_it, callbacks=None):
        self.model.train()
        for itr, batch in enumerate(self.train_dl):
            image, context = map(lambda attr: None if not batch.__contains__(attr) else batch[attr].to(self.device), ["mask", "context"])
            text = context
            loss = self.model(image, text, context)
            
            self.optimizer.zero_grad()
            self.accelerator.backward(loss)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                lr = self.lr_scheduler.get_last_lr()[0]
                self.lr_scheduler.step()
            else: lr = self.optimizer.defaults['lr']
            
            global_step = itr + len(self.train_dl) * self.epoch
            if self.accelerator.is_main_process:
                self.writer.add_scalar("train/lr", lr, global_step)
                self.writer.add_scalar("train/clip_loss", loss, global_step)
            train_it.set_postfix(itr=global_step, clip_loss=loss.item())
            train_it.update(1)
    
    def val(self, train_it=None):
        model = self.accelerator.unwrap_model(self.model)
        model.eval()
        v = 0
        val_it = tqdm(enumerate(self.val_dl), total=len(self.val_dl), desc='validation progress', position=1)
        
        for itr, batch in val_it:
            image, context = map(lambda attr: None if not batch.__contains__(attr) else batch[attr].to(self.device), ["mask", "context"])
            text = context
            loss = self.model(image, text, context)
            v += loss.item()
            val_it.set_postfix(clip_loss=loss.item(), val_itr=itr)
            if not self.parallel_val: train_it.update(1)
            
        self.writer.add_scalar(f'val/clip_loss', v / len(self.val_dl), self.epoch)
            
        if abs(new_best := v / len(self.val_dl)) < self.best:
            print(f"best lpips for epoch {self.epoch}: {new_best:.2f}")
            self.best = new_best
            self._save(os.path.join(self.model_path, f"best_model_cliploss.ckpt"),
                       best_lpips=new_best,
                       model=model.state_dict(),
                       optimizer=self.optimizer.state_dict())
            
    @staticmethod
    def _save(_save_path, /, **_save_dict):
        torch.save(_save_dict, _save_path)
        
    def _load(self, _load_path):
        obj = torch.load(_load_path, map_location="cpu")
        self.model.load_state_dict(obj["model"])
        self.optimizer.load_state_dict(obj["optimizer"])
        self.best = obj["current_best_dice"]


if __name__ == "__main__":
    spec = OmegaConf.to_container(OmegaConf.load("./run/train_ruijin_clip.yaml"))
    trainer = Trainer(spec, **spec["trainer"]["params"])
    trainer.train()
