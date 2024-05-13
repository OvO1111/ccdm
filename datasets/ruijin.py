
import re
import json

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import time
import torchio as tio
import SimpleITK as sitk
from einops import rearrange

import numpy as np
from tqdm import tqdm
import multiprocessing as mp

import torch
import torch.nn.functional as functional
from torch.utils.data import _utils, Dataset
from datasets.utils import load_or_write_split, conserve_only_certain_labels, TorchioForegroundCropper


def identity(x, *args, **kwargs):
    return x


def load_fn(x):
    return sitk.GetArrayFromImage(sitk.ReadImage(x))


class Ruijin_3D_Mask(Dataset):
    num_classes = 11  # not including crc mask
    cls_weight = [1,] + [1,] * 11
    def __init__(self,
                 split="train",
                 max_size=None,
                 collate_len=16,
                 use_summary_level="short",
                 spatial_size=(64, 128, 128),
                 text_encoder='CT_report_abstract_BLS_PULSE-20bv5_short',):
        
        with open('/ailab/user/dailinrui/data/records/dataset_crc_v2.json', 'rt') as f:
            data = json.load(f)
            self.data_keys = list(data.keys())
            self.data_keys.remove('RJ202302171638326937')  # which does not have features in BLS_PULSE-20bv5 extracted text features
            
        train_keys = self.data_keys[:round(len(self.data_keys) * 0.7)]
        val_keys = self.data_keys[round(len(self.data_keys) * 0.7): round(len(self.data_keys) * 0.8)]
        test_keys = self.data_keys[round(len(self.data_keys) * 0.8):]
        self.train_keys, self.val_keys, self.test_keys = load_or_write_split("/ailab/user/dailinrui/data/ccdm/", train=train_keys, val=val_keys, test=test_keys)
        
        self.collate_context_len = collate_len
        self.use_summary_level = use_summary_level
        if "PULSE" in text_encoder:
            if "short" in text_encoder: self.use_summary_level = "short"
            if "medium" in text_encoder: self.use_summary_level = "medium"
            if "long" in text_encoder: self.use_summary_level = "long"
            
        self.text_features = {name: value for name, value in np.load(f"/ailab/user/dailinrui/data/dependency/{text_encoder}.npz").items()}
        
        self.spatial_size = spatial_size
        self.transforms = dict(
            resize=tio.Resize(spatial_size) if spatial_size is not None else tio.Lambda(identity),
            crop=TorchioForegroundCropper(crop_level="mask_foreground", crop_anchor="mask", crop_kwargs=dict(foreground_mask_label=None,
                                                                                                             outline=(10, 10, 10))),
        )
        
        self.data = data
        self.split = split
        self.split_keys = getattr(self, f"{self.split}_keys")[:max_size]

    @staticmethod
    def _get_class(text):
        class_id = -1
        if "升结肠" in text: class_id = 0
        elif "横结肠" in text: class_id = 1
        elif "降结肠" in text: class_id = 2
        elif "乙状结肠" in text: class_id = 3
        elif "直肠" in text: class_id = 4
        else: class_id = 5
        return torch.tensor(class_id)

    def __len__(self):
        return len(self.split_keys)

    def __getitem__(self, idx):
        key = self.split_keys[idx] if isinstance(idx, int) else idx
        item = self.data[key]
        
        mask, crc_mask = map(lambda x: load_fn(item[x]), ["totalseg", "crcseg"])
        context = torch.tensor(self.text_features[key])
        spacing = torch.tensor(sitk.ReadImage(item["totalseg"]).GetSpacing())
        
        mask = conserve_only_certain_labels(mask)
        mask[crc_mask > 0] = 11
        
        subject = tio.Subject(mask=tio.LabelMap(tensor=mask[None]),)
        # crop
        subject = self.transforms["crop"](subject)
        # resize
        subject = self.transforms["resize"](subject)
        # random aug
        subject = self.transforms.get("augmentation", tio.Lambda(identity))(subject)
        
        if self.use_summary_level == "short": text = item.get("summary", "").split("；")[0]
        elif self.use_summary_level == "medium": text = item.get("summary", "")
        elif self.use_summary_level == "long": text = item.get("text", "")
        
        mask_one_hot = rearrange(torch.nn.functional.one_hot(subject.mask.data.long(), self.num_classes + 1), "1 h w d n -> n h w d")
        
        return {"image": mask[0:1], "mask": mask_one_hot, "text": text, "context": context,
                "class": self._get_class(item.get("summary", "").split("；")[0]),
                "spacing": spacing, "casename": key}
        
    def collate_fn(self, batch):
        context = [b["context"] for b in batch]
        for b in batch: del b["context"]
        collated = _utils.collate.default_collate(batch)
        longest_context = max([b.shape[0] for b in context]) if self.collate_context_len is None else self.collate_context_len
        collated_context = torch.cat([functional.pad(c, (0, 0, 0, longest_context - c.shape[0]), mode="constant", value=0) if c.shape[0] <= longest_context else c[:longest_context] for c in context], dim=0)
        collated["context"] = collated_context
        return collated
    
    def _preload_text_features(self, 
                               save_to="/mnt/workspace/dailinrui/data/pretrained/ccdm",
                               bert_ckpt="/mnt/data/oss_beijing/dailinrui/data/pretrained_weights/bert-ernie-health"):
        from ddpm.models import FrozenBERTEmbedder
        _frozen_text_embedder = FrozenBERTEmbedder(ckpt_path=bert_ckpt)
        feats = {}
        for c in tqdm(self.data_keys):
            feats[c] = _frozen_text_embedder([self.data[c]["text"]]).cpu().numpy()
        np.savez(os.path.join(save_to, bert_ckpt.split('/')[-1] + "_nogpt_extracted_features.npz"), **feats)
        
    def verify_dataset(self):
        from train.loss import TextRecoverModule
        recover_module = TextRecoverModule(5120, 1, 1024, device="cuda").cuda()
        
        iterator = tqdm(range(len(self.split_keys)))
        for idx in iterator:
            try:
                item = self.__getitem__(idx)
                loss, _ = recover_module(torch.randn((1, 4, 5120)).cuda(), [item["text"]])
                iterator.set_postfix(shape=item["mask"].shape, loss=loss, text=item["text"])
            except Exception as e:
                print(self.split_keys[idx], e, item["text"])


if __name__ == "__main__":
    x = Ruijin_3D_Mask()
    x.verify_dataset()