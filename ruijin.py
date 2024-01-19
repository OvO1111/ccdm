
import re
import json
import random
import pickle
from pathlib import Path
import torchio as tio
from einops import rearrange
import SimpleITK as sitk
from datetime import datetime
from functools import reduce, partial

import numpy as np
import multiprocessing as mp
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
import torch.nn.functional as f
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as tf
from torch.utils.data.dataset import Subset
from ddpm.models import FrozenBERTEmbedder


def conserve_only_certain_labels(label, designated_labels=[57]):
    # 6: stomach, 57: colon
    if designated_labels is None:
        return label.long()
    label_ = torch.zeros_like(label, dtype=torch.long)
    for il, l in enumerate(designated_labels):
        label_[label == l] = il + 1
    return label_


def window_norm(image, window_pos=60, window_width=360):
    window_min = window_pos - window_width // 2
    image = (image - window_min) / window_width
    image[image < 0] = 0
    image[image > 1] = 1
    return image


class PretrainDataset(Dataset):
    num_classes = 2  # not including crc mask
    # pretrain: CT cond on report -> totalseg organ mask
    def __init__(self, split="train", use_summary=False, max_size=None):
        with open('/mnt/data/oss_beijing/dailinrui/data/ruijin/dataset_crc.json', 'rt') as f:
            self.data = json.load(f)
            self.data_keys = list(self.data.keys())
            # self.data_keys.remove('RJ202302171638320174')  # which is to cause a "direction mismatch" in ct and mask
            
        self.use_summary = use_summary
        self.base_folder = "/mnt/data/oss_beijing/dailinrui/data/ruijin"
        self.load_fn = lambda x: sitk.GetArrayFromImage(sitk.ReadImage(x))
        # stage 1: train controlnet
        self.z = 32
        conserve_only_colon_label = partial(conserve_only_certain_labels, designated_labels=[57])
        self.mask_transform = tio.Lambda(conserve_only_colon_label)
        self.volume_transform = tio.Lambda(lambda x: window_norm(x))# tio.RescaleIntensity(out_min_max=(0, 1))
        self.joined_transform = tio.Compose((
            tio.CropOrPad((256, 256, self.z), mask_name="crc_mask", labels=[1,]),
            tio.Resize((128, 128, self.z)),
            # tio.OneOf(spatial_transformations)
        ))
        self.frozen_bert_embedder = FrozenBERTEmbedder()
        
        self.split = split
        train_portion = .9
        self.train_keys = self.data_keys[:round(len(self.data_keys) * train_portion)]
        self.val_keys = self.data_keys[round(len(self.data_keys) * train_portion):]
        if max_size is not None:
            self.train_keys = self.train_keys[:max_size]
            self.val_keys = self.val_keys[:max_size]
        self.split_keys = self.train_keys if self.split == "train" else self.val_keys
        
        cache_len = 500
        # n_procs = 16
        # pool = []
        # data_cache = mp.Manager().dict()
        
        # for p in range(n_procs):
        #     pool.append(mp.Process(target=self._preload, 
        #                            args=(self.split_keys[p:cache_len:n_procs], data_cache)))
        #     pool[-1].start()
            
        # for p in pool:
        #     p.join()
            
        # self.data_cache = dict(**data_cache)
        load_fn_im = lambda n: tio.ScalarImage(tensor=sitk.GetArrayFromImage(sitk.ReadImage(n))[None])
        load_fn_msk = lambda n: tio.LabelMap(tensor=sitk.GetArrayFromImage(sitk.ReadImage(n)).astype(np.uint8)[None])
        self.data_cache = {m: {"ct": load_fn_im(self.data[m]["ct"]), 
                               "totalseg": load_fn_msk(self.data[m]["totalseg"]), 
                               "crcseg": load_fn_msk(self.data[m]["crcseg"])} 
                           for im, m in tqdm(enumerate(self.train_keys if self.split == "train" else self.val_keys), 
                                             desc="image preload", 
                                             total=min(len(self.train_keys if self.split == "train" else self.val_keys), cache_len)) 
                           if im < cache_len}

    def __len__(self):
        return len(self.split_keys)
    
    def _preload(self, keys, cache):
        load_fn_im = lambda n: tio.ScalarImage(tensor=sitk.GetArrayFromImage(sitk.ReadImage(n))[None])
        load_fn_msk = lambda n: tio.LabelMap(tensor=sitk.GetArrayFromImage(sitk.ReadImage(n)).astype(np.uint8)[None])
        cache.update({m: [load_fn_im(self.data[m]["ct"]), 
                          load_fn_msk(self.data[m]["totalseg"]), 
                          load_fn_msk(self.data[m]["crcseg"])]
                          for _, m in tqdm(enumerate(keys), desc="data preload", total=len(keys))})
        return cache

    def __getitem__(self, idx):
        key = self.train_keys[idx] if self.split == "train" else self.val_keys[idx]
        item = self.data[key]  # {pacs, date, data, cond={<date>:<string>}}
        report = item["report"]
        data = Path(item["ct"])
        mask_name = Path(item["totalseg"])
        crc_mask_name = Path(item["crcseg"])
        
        # if len(cond) == 0: report = ""
        # else:
        #     report = re.sub(r"\\+[a-z]", "", reduce(lambda x, y: x + y, [f"{v}" for _, v in cond.items()], ""), 0, re.MULTILINE)
        #     report = re.sub(r" {2,}", " ", report, 0, re.MULTILINE)
        # mask_name = Path(self.base_folder, "totalseg_8k", data.parts[-3], data.parts[-2].split("_")[0], data.parts[-1].split(".")[0], "all.nii.gz")
        
        if self.data_cache.__contains__(data):
            image = self.data_cache[data][0]
            mask = self.data_cache[data][1]
            crc_mask = self.data_cache[data][2]
        else:
            image = tio.ScalarImage(data)
            mask = tio.LabelMap(mask_name)
            crc_mask = tio.LabelMap(crc_mask_name)
            
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        if self.volume_transform is not None:
            image = self.volume_transform(image)
        
        subject = tio.Subject(image=image, mask=mask, crc_mask=crc_mask)
        if self.joined_transform is not None:
            subject = self.joined_transform(subject)
            
        image = subject.image.data
        mask = subject.mask.data
        crc_mask = subject.crc_mask.data
            
        # for stage-1 training
        # random_slice = np.random.randint(self.z)
        # # random_slice = slice(random_slice, random_slice + 1)
        # image = image[..., random_slice]
        # mask = mask[0, ..., random_slice]
        # crc_mask = crc_mask[0, ..., random_slice]
        mask[crc_mask > 0] = mask.max() + 1
        mask = f.one_hot(mask.long(), num_classes=PretrainDataset.num_classes + 1)
        mask = rearrange(mask, "1 h w d c -> c d h w")
        image = rearrange(image, "1 h w d -> 1 d h w")
        
        return {"image": image, "mask": mask}#, "text": report}
        
        
def training_dataset(toy=False):
    return PretrainDataset(max_size=None, split="train")


def validation_dataset(max_size=None):
    return PretrainDataset(max_size=10, split="val")


def get_ignore_class():
    return 0


if __name__ == "__main__":
    x = PretrainDataset()
    x.test()