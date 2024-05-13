import os
import torch
import argparse

from tensorboardX import SummaryWriter
from omegaconf import OmegaConf
import torch.distributed
from train.utils import get_cls_from_pkg, make_tabulate_data_from_nested_dict
    

def main(spec, subjects):
    # torch.distributed.init_process_group("nccl")
    spec = OmegaConf.load(spec)
    new_spec = OmegaConf.from_cli(subjects)
    spec_ = OmegaConf.to_container(OmegaConf.merge(spec, new_spec))
    make_tabulate_data_from_nested_dict(spec_)
    trainer = get_cls_from_pkg(spec_["trainer"], spec=spec_)
    with open(os.path.join(trainer.snapshot_path, "conf.yaml"), "w") as w:
        OmegaConf.save(OmegaConf.create(spec), w)
    trainer.train()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", type=str, default="./configs/train_ruijin_ccdm.yaml")
    args, unknowns = parser.parse_known_args()
    main(args.cfg, unknowns)