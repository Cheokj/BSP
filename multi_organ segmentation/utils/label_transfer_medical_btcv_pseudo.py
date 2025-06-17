from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    CenterSpatialCropd,
    Resized,
    SpatialPadd,
    apply_transform,
)

import collections.abc
import math
import pickle
import shutil
import sys
import tempfile
import threading
import time
from copy import copy, deepcopy
import cc3d
import argparse
import os
import h5py
import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.WARNING)

import numpy as np
import nibabel as nib
import torch
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

torch.multiprocessing.set_sharing_strategy('file_system')

from monai.data import DataLoader, Dataset, list_data_collate, DistributedSampler
from monai.config import DtypeLike, KeysCollection
from monai.transforms.transform import Transform, MapTransform
from monai.utils.enums import TransformBackends
from monai.config.type_definitions import NdarrayOrTensor


# 定义标签映射表
LABEL_MAP = {
    'LiTS': {1: 1, 2: 1},
    'KiTS': {1: 2, 2: 2},
    'pancreas': {1: 3, 2: 3},
    'spleen': {1: 4},
    'AMOS': {6: 1, 2: 2, 3: 2, 10: 3, 1: 4, 4: 5},
    'BTCV': {6: 1, 2: 2, 3: 2, 11: 3, 1: 4, 4: 5},
}

DATASET_NAME = "BTCV"
ORGAN_DATASET_DIR = '/public1/cuikangjun/Data_Seg/'
ORGAN_LIST = '/public1/cuikangjun/Data_Seg/BTCV/btcv_train.txt'
STEP = 5    # if DATASET_NAME == "BTCV", then need to set STEP, 指定哪些类可见
NUM_WORKER = 8
TRANSFER_LIST = ['01', '02', '03', '04', '05', '06']


def modify_name(name, step):
    parts = name.split('/')
    parts.insert(1, f'step{step}')
    return '/'.join(parts)

def create_one_hot(lbl, dataset_name):
    """
    Convert class index tensor to one hot encoding tensor.
    Args:
        lbl: A tensor of shape [bs, 1, *]
        dataset_name: dataset name
    Returns:
        A tensor of shape [bs, num_classes, *], mapped_labels
    """
    # 获取标签映射
    label_mapping = LABEL_MAP[dataset_name]

    # 创建一个与lbl相同形状的空数组用于存储映射后的标签
    mapped_labels = np.zeros_like(lbl)

    # 遍历标签映射字典，并应用映射
    for original_label, new_label in label_mapping.items():
        mapped_labels[lbl == original_label] = new_label
    
    # 确定类别数量
    num_classes = max(label_mapping.values()) + 1  # 加1，因为0通常表示背景

    # 将映射后的标签值大于step的设为0  BTCV用来做测试集
    if DATASET_NAME == "BTCV":
        mapped_labels[mapped_labels > STEP] = 0
        num_classes = STEP + 1
    
    # 初始化one-hot编码数组，形状为 [b, num_classes, h, w, d]
    one_hot_labels = np.zeros((lbl.shape[0], num_classes, lbl.shape[2], lbl.shape[3], lbl.shape[4]), dtype=np.float32)

    # 遍历每个类，并设置对应位置的值为1
    for i in range(num_classes):
        one_hot_labels[:, i, :, :, :] = (mapped_labels == i)

    return one_hot_labels, mapped_labels

label_process = Compose(
    [
        LoadImaged(keys=["image", "label", "label_raw"]),
        AddChanneld(keys=["image", "label", "label_raw"]),
        # Orientationd(keys=["image", "label", "label_raw"], axcodes="RAS"),
        # Spacingd(
        #     keys=["image", "label", "label_raw"], 
        #     pixdim=(1.5, 1.5, 1.5), 
        #     mode=("bilinear", "nearest", "nearest"),
        # ),
    ]
)

train_img = []
train_lbl = []
train_name = []

for line in open(ORGAN_LIST):
    train_img.append(ORGAN_DATASET_DIR + line.strip().split()[0])
    train_lbl.append(ORGAN_DATASET_DIR + line.strip().split()[1])
    train_name.append(line.strip().split()[1].split('.')[0])
data_dicts_train = [{'image': image, 'label': label, 'label_raw': label, 'name': name}
                    for image, label, name in zip(train_img, train_lbl, train_name)]
print('train len {}'.format(len(data_dicts_train)))

train_dataset = Dataset(data=data_dicts_train, transform=label_process)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKER, 
                          collate_fn=list_data_collate)

for index, batch in enumerate(train_loader):
    x, y, y_raw, name = batch["image"], batch["label"], batch['label_raw'], batch['name']
    print(f"x:{x.shape}, y:{y.shape}")
    one_hot_y, mapped_labels = create_one_hot(y, DATASET_NAME)
    print(f"one_hot_y:{one_hot_y.shape}, mapped_labels:{mapped_labels.shape}")
    print(f"unique(mapped_labels): {np.unique(mapped_labels)}")

    name = modify_name(batch['name'][0], STEP)
    print("Modified name:", name)

    post_dir = os.path.join(ORGAN_DATASET_DIR, *name.split('/')[:-1])
    if not os.path.exists(post_dir):
        os.makedirs(post_dir)
    print("post_dir:", post_dir)

    # 保存处理后的标签数据为NIfTI格式
    lbl = nib.load(ORGAN_DATASET_DIR + batch['name'][0] + ".nii.gz")
    output_label_path = os.path.join(post_dir, name.split('/')[-1] + '.nii.gz')
    label_nii = nib.Nifti1Image(mapped_labels[0, 0], lbl.affine, lbl.header)  # 使用原始标签的仿射矩阵
    print("output_path:", output_label_path)
    nib.save(label_nii, output_label_path)
    
    print(f'Saved processed label to {output_label_path}')

