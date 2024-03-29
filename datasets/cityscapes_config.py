
from collections import namedtuple

import torch
import numpy as np


CITYSCAPES_LABEL_COLORMAP = torch.tensor([
    [0, 0, 0],
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32]
])


CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                 'has_instances', 'ignore_in_eval', 'color'])
classes = [
    CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
    CityscapesClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
    CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
    CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
    CityscapesClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
    CityscapesClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
    CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
    CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
    CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
    CityscapesClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
    CityscapesClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
    CityscapesClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
    CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
    CityscapesClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
    CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
    CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
    CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
    CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
    CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
    CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
    CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
    CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
    CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
    CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
    CityscapesClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
    CityscapesClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
    CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
    CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
    CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
    CityscapesClass('license plate', -1, 255, 'vehicle', 7, False, True, (0, 0, 142)),
]

crc_classes = [
    CityscapesClass('unlabeled', 0, 255, 'void', 0, False, False, (0, 0, 0)),
    CityscapesClass('colon', 1, 255, 'void', 0, False, False, (255, 0, 0)),
    CityscapesClass('colorectal cancer', 2, 255, 'void', 0, False, False, (0, 255, 0)),
    CityscapesClass('gallbladder', 3, 255, 'void', 0, False, False, (0, 0, 255),),
    CityscapesClass('liver', 4, 6, 'void', 3, False, False, (250, 170, 30)),
    CityscapesClass('stomach', 5, 7, 'void', 3, False, False, (220, 220, 0)),
    CityscapesClass('pancreas', 6, 8, 'void', 4, False, False, (107, 142, 35)),
    CityscapesClass('small_bowel', 7, 9, 'void', 4, False, False, (152, 251, 152)),
    CityscapesClass('duodenum', 8, 10, 'void', 4, False, False, (0, 255, 255)),
    CityscapesClass('urinary_bladder', 9, 11, 'void', 5, False, False, (70, 130, 180)),
]

train_id_to_crc_color = torch.tensor(np.array([c.color for c in crc_classes]))

train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
train_id_to_color.append([0, 0, 0])
train_id_to_color = np.array(train_id_to_color)
train_id_to_color_th = torch.tensor(train_id_to_color)

train_id_to_mm_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
train_id_to_mm_color.append([84, 86, 22])
train_id_to_mm_color.append([167, 242, 242])
train_id_to_mm_color.append([242, 160, 19])
train_id_to_mm_color.append([30, 193, 252])
train_id_to_mm_color.append([46, 247, 180])
train_id_to_mm_color.append([0, 0, 0])
train_id_to_mm_color = np.array(train_id_to_mm_color)
train_id_to_mm_color_th = torch.tensor(train_id_to_mm_color)

id_to_train_id = np.array([c.train_id for c in classes])
id_to_train_id[id_to_train_id == 255] = 19

train_id_to_id = [c.id for c in classes if (c.train_id != -1 and c.train_id != 255)]
train_id_to_id.append(0)
train_id_to_id = np.array(train_id_to_id)
train_id_to_id_th = torch.tensor((train_id_to_id))


def encode_target(target):
    # NB: this converts target from context "labelId" to context "trainId"
    # NB: id_to_train_id: [0, 19] (20 classes)
    return id_to_train_id[np.array(target)]


def decode_target_to_color(target):
    target[target == 255] = 19
    if torch.is_tensor(target):
        return train_id_to_color_th[target.to(train_id_to_id_th.device)]
    return train_id_to_color[target]


def decode_target_to_mm_color(target):
    target[target == 255] = 24
    return train_id_to_mm_color_th[target.to(train_id_to_id_th.device)]


def decode_target_to_crc_color(target):
    target[target == 255] = 0
    return train_id_to_crc_color[target.to(train_id_to_id_th.device)]


def map_train_id_to_id(target):
    target[target == 255] = 19
    if torch.is_tensor(target):
        return train_id_to_id_th[target]
    return train_id_to_id[target]
