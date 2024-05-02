
import re
import json
import random
import pickle

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torchio as tio
from einops import rearrange
import nibabel
import SimpleITK as sitk
from datetime import datetime
from functools import reduce, partial

import numpy as np
import multiprocessing as mp
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
import torch.nn.functional as functional
from torch.utils.data import _utils, Dataset, IterableDataset
import torchvision.transforms.functional as tf
from torch.utils.data.dataset import Subset
from datasets.ruijin_config import abd_organ_classes


def conserve_only_certain_labels(label,
                                 designated_labels=[1, 2, 3, 5, 6, 10, 55, 56, 57, 104],
                                 preserve_original_indices=False):
    # as per TotalSegmentatorv1 conventions
    if designated_labels is None:
        return label.long()
    label_ = np.zeros_like(label, dtype=np.uint8)
    for il, l in enumerate(designated_labels):
        label_[label == l] = l if preserve_original_indices else il + 1
    return label_


def window_norm(image, window_pos=60, window_width=360):
    window_min = window_pos - window_width // 2
    image = (image - window_min) / window_width
    image[image < 0] = 0
    image[image > 1] = 1
    return image


def load_fn(n):
    return nibabel.load(n).dataobj[:]
    
    
def load_or_write_split(basefolder, force=False, **splits):
    splits_file = os.path.join(basefolder, "splits.json")
    if os.path.exists(splits_file) and not force:
        with open(splits_file, "r") as f:
            splits = json.load(f)
    else:
        with open(splits_file, "w") as f:
            json.dump(splits, f, indent=4)
    splits = list(splits.get(_) for _ in ["train", "val", "test"])
    return splits


with open('/mnt/data/oss_beijing/dailinrui/data/ruijin/records/dataset_crc_v2.json', 'rt') as f:
    data = json.load(f)
    data_keys = list(data.keys())
    data_keys.remove('RJ202302171638326937')  # which does not have features in BLS_PULSE-20bv5 extracted text features
    
train_keys = data_keys[:round(len(data_keys) * 0.7)]
val_keys = data_keys[round(len(data_keys) * 0.7):round(len(data_keys) * 0.8)]
test_keys = data_keys[round(len(data_keys) * 0.8):]
train_keys, val_keys, test_keys = load_or_write_split("/mnt/workspace/dailinrui/data/pretrained/ccdm/", train=train_keys, val=val_keys)
text_features = np.load("/mnt/workspace/dailinrui/data/pretrained/ccdm/CT_report_abstract_BLS_PULSE-20bv5_short.npz")
text_feature_cache = {k: text_features[k] for ik, k in enumerate(data_keys)}


class PretrainDataset(Dataset):
    num_classes = 11  # not including crc mask
    cls_weight = [1,] + [1,] * 11
    def __init__(self,
                 split="train",
                 use_summary=False,
                 spatial_size=(128, 128, 64),
                 max_size=None):
        self.spatial_size = spatial_size
        self.joined_transform = tio.Compose((
            tio.Resize(self.spatial_size),
            # tio.OneOf(spatial_transformations)
        ))
        
        self.data = data
        self.split = split
        self.use_summary = use_summary
        self.train_keys = train_keys[:max_size]
        self.val_keys = val_keys[:max_size]
        self.text_feature_cache = text_feature_cache
        self.split_keys = self.train_keys if self.split == "train" else self.val_keys

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
        key = self.split_keys[idx]
        item = self.data[key]
        report = item.get("report", item.get("summary", None))
        
        mask = load_fn(item["totalseg"])
        crc_mask = load_fn(item["crcseg"])
        context = self.text_feature_cache.get(key)[0]
        spacing = torch.tensor(sitk.ReadImage(item["totalseg"]).GetSpacing())
            
        mask = conserve_only_certain_labels(mask, preserve_original_indices=False)
        mask[crc_mask > 0] = mask.max() + 1
        start_layer, end_layer = np.where(mask.sum((0, 1)))[0][[0, -1]]
        _mask = tio.LabelMap(tensor=torch.tensor(mask[..., max(0, start_layer): end_layer + 1])[None])
        if self.joined_transform is not None:
            _mask = self.joined_transform(_mask)
            
        _mask = functional.one_hot(_mask.data.long(), num_classes=PretrainDataset.num_classes + 1)
        _mask = rearrange(_mask, "1 h w d c -> c d h w")
        
        image = _mask[0: 1].float()
        image[...] = 0
        
        return {"image": image,
                "mask": _mask,
                "text": report.split("；")[0],
                "class": self._get_class(report.split("；")[0]),
                "context": context,
                "spacing": spacing,
                "casename": key}
        
    def collate_fn(self, batch):
        context = [b["context"] for b in batch]
        for b in batch: del b["context"]
        collated = _utils.collate.default_collate(batch)
        longest_context = max([b.shape[0] for b in context])
        collated_context = torch.tensor(np.array([np.pad(c, ((0, longest_context - c.shape[0]), (0, 0)), mode="constant", constant_values=0) for c in context]))
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
        
        
class CLIPDataset(PretrainDataset):
    def __init__(self,
                 split="train",
                 use_summary=False,
                 spatial_size=(128, 128, 64),
                 max_size=None, force_collate_len=16):
        self.collate_maxlen = force_collate_len
        super().__init__(split, use_summary, spatial_size, max_size)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        _mask = item["mask"].argmax(0)[None]
        # _mask = conserve_only_certain_labels(_mask, designated_labels=[9, 11])
        item["mask"] = _mask
        return item
    
    def collate_fn(self, batch):
        context = [b["context"] for b in batch]
        for b in batch: del b["context"]
        collated = _utils.collate.default_collate(batch)
        longest_context = max([b.shape[0] for b in context]) if self.collate_maxlen is None else self.collate_maxlen
        collated_context = torch.tensor(np.array([np.pad(c, ((0, longest_context - c.shape[0]), (0, 0)), mode="constant", constant_values=0)
                                                  if c.shape[0] <= longest_context else c[:longest_context] for c in context]))
        collated = [collated["mask"].float(), collated_context]
        return collated
        
        
def training_dataset(toy=False):
    return PretrainDataset(max_size=None, split="train")


def validation_dataset(max_size=50):
    return PretrainDataset(max_size=max_size, split="val")


def get_ignore_class():
    return 0

def get_weights(*args, **kwargs):
    raw = torch.ones(get_num_classes())
    raw[-1] = 1
    return raw

def get_num_classes():
    return PretrainDataset.num_classes + 1

def train_ids_to_class_names():
    return {ic: c.label_name for ic, c in enumerate(abd_organ_classes)}


if __name__ == "__main__":
    x = PretrainDataset(cache_len=0)
    x._preload_text_features()