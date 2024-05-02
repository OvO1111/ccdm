import sys, os
import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as f

from tqdm import tqdm
from omegaconf import OmegaConf
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import LinearLR, LambdaLR
from torch.utils.data import Dataset, DataLoader

sys.path.append("/mnt/workspace/dailinrui/code/multimodal/trajectory_generation/ccdm")
from train.utils import default, identity, maybe_mkdir, get_cls_from_pkg, visualize, print_parameters, BasicLogger, make_tabulate_data_from_nested_dict
from train.ccdm import CategoricalDiffusionModel, OneHotCategoricalBCHW
from train.loss import DiffusionKLLoss, CrossEntropyLoss, LPIPS
from train.eval_metrics import pretrained_lpips, pretrained_fvd, pretrained_seg


class Trainer:
    legends = ["background", "spleen", "kidney_left", "kidney_right", "liver", "stomach", "pancreas", "small_bowel",
               "duodenum", "colon", "uniary_bladder", "colorectal_cancer"]
    def __init__(self, spec, *, 
                 val_device=None,
                 batch_size=4,
                 lr=1e-3,
                 max_epochs=100,
                 timesteps=1000,
                 snapshot_path=None,
                 restore_path=None,
                 parallel_validation=False,
                 val_every=1, save_every=5,
                 save_n_train_image_per_epoch=10,
                 save_n_val_image_per_epoch=5) -> None:
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.timesteps = timesteps
        self.val_every = val_every
        self.save_every = save_every
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
                                   shuffle=True, pin_memory=True, num_workers=4, collate_fn=self.train_ds.collate_fn)
        self.val_dl = DataLoader(self.val_ds,
                                 batch_size=1,
                                 pin_memory=True,
                                 num_workers=2, collate_fn=self.val_ds.collate_fn)
        self.test_dl = DataLoader(self.test_ds,
                                  batch_size=1,
                                  pin_memory=True,
                                  num_workers=2, collate_fn=self.test_ds.collate_fn)
        
        self.logger = {"train": BasicLogger("train", 10, self.visualization_path),
                       "val":   BasicLogger("val", 10, self.visualization_path),
                       "test":  BasicLogger("test", 10, self.visualization_path), 
                       "nifti": BasicLogger("nifti", 10, self.visualization_path),
                       "tf":    BasicLogger("tboard", -1, self.snapshot_path, create_tensorboard=True),
                       "model": BasicLogger("checkpoint", 5, self.snapshot_path)}
        
        self.optimizer = AdamW(self.model.parameters(), self.lr / self.batch_size)
        self.lr_scheduler = LinearLR(self.optimizer, start_factor=1, total_iters=self.max_epochs)
        if restore_path is not None: self.model.load_state_dict(torch.load(restore_path, map_location='cpu'))
        
        self.accelerator = Accelerator(project_dir=self.snapshot_path, 
                                       kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
        self.device = self.accelerator.device
        self.val_device = torch.device("cuda", int(val_device)) if val_device is not None else self.device
        self.lpips = LPIPS(1, 1,
                           use_linear=False,
                           ndim=self.model.unet.dims,
                           model_path="/mnt/workspace/dailinrui/data/pretrained/ccdm/lpips_backbone_best.ckpt",
                           net_backbone="colon_segmentation")
        self.kl_loss = DiffusionKLLoss(attn_weight=torch.tensor(self.train_ds.cls_weight, device=self.device),
                                     diffusion_model=self.accelerator.unwrap_model(self.model).diffusion_model)
        self.model, self.optimizer, self.train_dl, self.val_dl, self.lr_scheduler, self.lpips =\
            self.accelerator.prepare(self.model, self.optimizer, self.train_dl, self.val_dl, self.lr_scheduler, self.lpips)
        
        self.loss = {CrossEntropyLoss(ignore_class=5): 0.,
                     self.kl_loss: 0.5,
                     self.lpips: 0.}
        print_parameters(model=self.model, lpips=self.lpips)
        self.train_vis_step = max(1, len(self.train_dl) // save_n_train_image_per_epoch)
        self.val_vis_step = max(1, len(self.val_dl) // save_n_val_image_per_epoch)
     
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
        
    def train(self):
        self.model.train()
        if self.parallel_val: self.train_it = tqdm(range(self.max_epochs * len(self.train_ds)), desc="train progress")
        else: self.train_it = tqdm(range(self.max_epochs * (len(self.train_ds) + len(self.val_ds))), desc="train progress")
        
        for self.epoch in range(self.max_epochs):
            self._train()
            self.lr_scheduler.step()
            
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                if self.epoch % self.val_every == 0:
                    if self.parallel_val:
                        if self.val_proc is not None: 
                            try:
                                self.val_proc.join()
                            except Exception as e: print(e)  # noqa: may report error when parallel validating because tensorboard is not closed properly
                        self.val_proc = torch.multiprocessing.get_context("spawn").Process(target=self.val, args=(0, self.model.state_dict()))
                        self.val_proc.start()
                    else:
                        self.val()
                
                if self.epoch & self.save_every == 0:
                    self.logger["model"](dict(lpips_model=self.accelerator.unwrap_model(self.lpips).state_dict(),
                                            model=self.accelerator.unwrap_model(self.model).state_dict(),
                                            optimizer=self.optimizer.state_dict(),
                                            epoch=self.epoch), f"checkpoint_ep{self.epoch}_gs{self.train_it.n}.ckpt")
                    
        self.logger["model"](dict(lpips_model=self.accelerator.unwrap_model(self.lpips).state_dict(),
                                    model=self.accelerator.unwrap_model(self.model).state_dict(),
                                    epoch=self.epoch), f"last.ckpt")
        self.accelerator.end_training()
            
    def _train(self, callbacks=None):
        self.model.train()
        for itr, batch in enumerate(self.train_dl):
            itr_loss = {}
            x0, condition, context, class_id = map(lambda attr: None if not batch.__contains__(attr) else batch[attr].to(self.device), ["mask", "image", "context", "class"])
            
            x0 = self.x_encoder(x0).float()
            t = torch.multinomial(torch.arange(self.timesteps + 1, device=self.device) ** 1.5, self.batch_size)
            xt = self.accelerator.unwrap_model(self.model).diffusion_model.q_xt_given_x0(x0, t).sample()
            ret = self.model(xt.contiguous(),
                             self.condition_encoder(None), None, t,
                             context=self.context_encoder(context))
            c_pred = ret["cond_pred_logits"]
            x0_pred = ret["diffusion_out"]
            if itr % self.train_vis_step == 0 and self.accelerator.is_main_process:
                self.logger["train"](dict(inputs=x0.argmax(1),
                                            xt=xt.argmax(1),
                                            t=str(t.cpu().numpy().tolist()),
                                            samples=x0_pred.argmax(1)), f"train_ep{self.epoch}_gs{self.train_it.n}.png")
            
            loss_diffusion = self.kl_loss(xt, x0, x0_pred, t)
            loss_ce = nn.functional.cross_entropy(c_pred, class_id)
            # loss_l1 = nn.functional.l1_loss(x0, x0_pred)
            loss = loss_diffusion
            itr_loss["DiffusionKLLoss"] = loss_diffusion
            itr_loss["CrossEntropyLoss"] = loss_ce
            
            self.optimizer.zero_grad()
            self.accelerator.backward(loss_diffusion)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                lr = self.lr_scheduler.get_last_lr()[0]
                self.lr_scheduler.step()
            else: lr = self.optimizer.defaults['lr']
            
            global_step = itr + len(self.train_dl) * self.epoch
            if self.accelerator.is_main_process:
                self.logger["tf"](ext="tf").add_scalar("train/lr", lr, global_step)
                self.logger["tf"](ext="tf").add_scalars("train/loss", {k: round(v.item(), 2) for k, v in itr_loss.items()}, global_step)
            self.train_it.set_postfix(itr=global_step, **{k: f"{v.item():.2f}" for k, v in itr_loss.items()}, debug=x0_pred.argmax(1).max().item())
            self.train_it.update(1)
    
    def val(self, state_dict=None):
        model = self.accelerator.unwrap_model(self.model)
        if state_dict is not None: model.load_state_dict(state_dict)
        model.eval()
        val_it = tqdm(enumerate(self.val_dl), total=len(self.val_dl), desc='validation progress', position=1)
        
        lpips = self.accelerator.unwrap_model(self.lpips)
        val_loss = {k.__class__.__name__: 0 for k in [lpips, self.kl_loss]}
        
        for itr, batch in val_it:
            x0, condition, context, class_id = map(lambda attr: None if not batch.__contains__(attr) else batch[attr].to(self.device), ["mask", "image", "context", "class"])
            
            x0 = self.x_encoder(x0)
            x0_shape = (x0.shape[0], model.diffusion_model.num_classes, *x0.shape[2:])
            xt = OneHotCategoricalBCHW(logits=torch.zeros(x0_shape, device=self.device)).sample()
            # ret = model.ddim_sampling(xt.contiguous(),
            #                         self.condition_encoder(None),
            #                         feature_condition=None,
            #                         context=self.context_encoder(context),
            #                         ref_class=class_id, 
            #                         loss_fn=nn.CrossEntropyLoss(ignore_index=5))
            # x0_pred, x0_denoise, ce_loss = ret
            ret = model(xt.contiguous(),
                        self.condition_encoder(None), None, None,
                        context=self.context_encoder(context))
            x0_pred = ret["diffusion_out"]
            
            if itr % self.val_vis_step == 0:
                self.logger["val"](dict(inputs=x0.argmax(1),
                                        xt=xt.argmax(1),
                                        samples=x0_pred.argmax(1),), f"val_ep{self.epoch}_gs{getattr(self.train_it, 'n', -1)}.png")
            
            val_it.set_postfix(**{k: v / (itr + 1) for k, v in val_loss.items()})
            if not self.parallel_val: self.train_it.update(1)
            
        self.logger["tf"](ext="tf").add_images("val/gen", visualize(x0_pred.argmax(1), n=len(self.legends) - 1), self.epoch)
        for k, v in val_loss.items(): self.logger["tf"](ext="tf").add_scalar(f'val/unnormalized_{k}', v / len(self.val_dl), self.epoch)
            
        if abs(new_best := val_loss["LPIPS"] / len(self.val_dl)) < self.best:
            print(f"best lpips for epoch {self.epoch}: {new_best:.2f}")
            self.best = new_best
            self.logger["model"](dict(lpips_model=self.accelerator.unwrap_model(self.lpips).state_dict(),
                                      model=self.accelerator.unwrap_model(self.model).state_dict(),
                                      optimizer=self.optimizer.state_dict(),
                                      epoch=self.epoch), f"best.ckpt")
        
    def _load(self, _load_path):
        obj = torch.load(_load_path, map_location="cpu")
        self.model.load_state_dict(obj["model"])
        self.optimizer.load_state_dict(obj["optimizer"])
        self.best = obj["current_best_dice"]
    
    @torch.no_grad()
    def _test(self, state_dict=None, callbacks=None):
        model = self.accelerator.unwrap_model(self.model)
        if state_dict is not None: model.load_state_dict(state_dict)
        model.eval()
        
        ret = {fn.__class__.__name__: 0 for fn in callbacks}
        test_it = tqdm(enumerate(self.test_dl), total=len(self.test_dl), desc='test progress', position=1)
        
        for itr, batch in test_it:
            x0, condition, context, class_id = map(lambda attr: None if not batch.__contains__(attr) else batch[attr].to(self.device), ["mask", "image", "context", "class"])
            
            x0 = self.x_encoder(x0)
            x0_shape = (x0.shape[0], model.diffusion_model.num_classes, *x0.shape[2:])
            xt = OneHotCategoricalBCHW(logits=torch.zeros(x0_shape, device=self.device)).sample()
            # r = model.ddim_sampling(xt.contiguous(),
            #                         self.condition_encoder(None),
            #                         feature_condition=None,
            #                         context=self.context_encoder(context),
            #                         ref_class=class_id, 
            #                         loss_fn=nn.CrossEntropyLoss(ignore_index=5))
            # x0_pred, x0_denoise, ce_loss = r
            r = model(xt.contiguous(),
                      self.condition_encoder(None), None, None,
                      context=self.context_encoder(context))
            x0_pred = r["diffusion_out"]
            self.logger["test"](dict(inputs=xt.argmax(1),
                                    #  denoise=torch.cat(x0_denoise["pred_x0"], dim=0).argmax(1),
                                     text="\n".join(batch["text"]),
                                     samples=x0_pred.argmax(1),), f"test_gb{test_it.n}.png")
            cb = {}
            for callback in callbacks:
                cb[callback.__class__.__name__] = callback(x0_pred, x0)
                ret[callback.__class__.__name__] += cb[callback.__class__.__name__] / len(self.test_ds)
            for b in range(x0_pred.shape[0]):
                nifti = x0_pred[b].argmax(0).cpu().numpy()
                nifti[nifti == 9] = callbacks[-1].segment_x[0, nifti == 9] + 100  # colon segments at 100-105, 100 for segmented bg
                self.logger["nifti"](nifti, f"test_gb{test_it.n}_bn{b}.nii.gz")
            test_it.set_postfix(debug=x0_pred.argmax(1).max().item(), **{k: v * len(self.test_ds) for k, v in cb.items()})
            
        make_tabulate_data_from_nested_dict(ret)
    
    @torch.no_grad()  
    def test(self, metrics=["lpips", "fvd", "seg"], path=None):
        callbacks = []
        state_dict = None
        if path is not None:
            state_dict = torch.load(path, map_location="cpu")["model"]
            
        if "lpips" in metrics: 
            callbacks.append(pretrained_lpips(self.device))
        if "fvd" in metrics: 
            callbacks.append(pretrained_fvd(bs=2))
        if "seg" in metrics: 
            callbacks.append(pretrained_seg(self.device))
        
        self._test(callbacks=callbacks, state_dict=state_dict)


if __name__ == "__main__":
    spec = OmegaConf.to_container(OmegaConf.load("./run/train_ruijin_ccdm.yaml"))
    trainer = Trainer(spec, **spec["trainer"]["params"])
    # trainer.train()
    trainer.test(["lpips", "seg"], path="/mnt/workspace/dailinrui/data/pretrained/ccdm/trainer_v2/dlc_test_multigpu/checkpoint/checkpoint_ep24_gs21550.ckpt")
            