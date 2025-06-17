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

# from utils.utils import get_key

# DATASET_NAME = 'KiTS'
# ORGAN_LIST = '/public1/cuikangjun/Data_Seg/02_KiTS19/data/kits_pseudo-train.txt'
DATASET_NAME = sys.argv[1]
ORGAN_LIST = sys.argv[2]
print(f"dataset:{DATASET_NAME}, organ_list:{ORGAN_LIST}")

ORGAN_DATASET_DIR = '/public1/cuikangjun/Data_Seg/'
NUM_WORKER = 8
TRANSFER_LIST = ['01', '02', '03', '04', '05', '06']
# LABEL_MAP = {
#     'LiTS':{1:1, 2:1},
#     'KiTS':{1:2, 2:2},
#     'pancreas':{1:3, 2:3},
#     'spleen':{1:4},
#     #'AMOS':{6:1, 2:2, 3:2, 10:3, 1:4, 4:5},
#     'AMOS':{4:5},
#     'BTCV':{6:1, 2:2, 3:2, 11:3, 1:4, 4:5},
# }
LABEL_MAP = {
    'LiTS':{1:1, 2:1},
    'spleen':{1:2},
    'pancreas':{1:3, 2:3},
    'KiTS':{4:4, 5:5},                            # 4:Right Kidney, 5:Left Kidney
    # 'AMOS':{6:1, 2:4, 3:5, 10:3, 1:2, 4:6},     # test
    'AMOS':{4:6},                                 # train
    'BTCV':{6:1, 2:4, 3:5, 11:3, 1:2, 4:6},
}

def get_key(name):
    ## input: name
    ## output: the corresponding key
    dataset_index = int(name[0:2])
    if dataset_index == 10:
        template_key = name[0:2] + '_' + name[17:19]
    else:
        template_key = name[0:2]
    return template_key


def npy_to_nii(input, output_nii_path):
    # 假设 one_hot_data 是你的 one-hot 编码数据，形状为 [H, W, D]
    # C 是类别数，H, W, D 分别是图像的高、宽和深度

    # 将 one-hot 编码的数据转换回单通道标签数据
    # input_data = np.argmax(input, axis=1)
    print('input:', input.shape)

    # 创建一个新的 NIfTI 图像
    # 假设你有原始图像的 NIfTI 文件，用于获取 affine 和 header 信息
    # 这里我们用一个示例 NIfTI 文件路径替换
    example_nii_path = '/public1/cuikangjun/Data_Seg/01_LiTS_downsampled/labels/segmentation-0.nii'
    example_nii = nib.load(example_nii_path)

    # 使用原始图像的 affine 和 header 创建新的 NIfTI 图像
    new_nii = nib.Nifti1Image(input.astype(np.uint8), affine=example_nii.affine, header=example_nii.header)
    
    # test_nii = nib.load(new_nii)
    print("new_nii.shape:", new_nii.shape)

    # 保存为新的 NIfTI 文件
    nib.save(new_nii, output_nii_path)

    print(f"Saved one-hot encoded data to NIfTI file at: {output_nii_path}")


def load_nifti(file_path):
    img = nib.load(file_path)
    data = img.get_fdata()
    return img, data

def save_nifti(data, affine, file_path):
    new_img = nib.Nifti1Image(data.astype(np.uint8), affine)
    nib.save(new_img, file_path)



def combine_labels(real_label, pseudo_label, dataset_name):
    """
    Convert class index tensor to one hot encoding tensor.
    Args:
        real_label, pseudo_label: A tensor of shape [bs, 1, *]
        dataset_name: dataset name
    Returns:
        A tensor of shape [bs, num_classes, *]
    """
    # 1.获取标签映射
    label_mapping = LABEL_MAP[dataset_name]

    # 2.创建一个与real_label相同形状的空数组用于存储映射后的标签
    mapped_labels = np.zeros_like(real_label)

    # 3.遍历标签映射字典，并应用映射
    for original_label, new_label in label_mapping.items():
        mapped_labels[real_label == original_label] = new_label

    # 4.合并真实标签和伪标签
    combined_labels = np.where(mapped_labels > 0, mapped_labels, pseudo_label)

    # example_nii_path = '/public1/cuikangjun/Data_Seg/02_KiTS19/data/imagesTr/imaging-00000.nii.gz'
    # example_nii = nib.load(example_nii_path)
    # save_nifti(combined_labels[0,0], example_nii.affine, "/public1/cuikangjun/ContinualLearning/pseudo-00000.nii.gz")
    
    # 5.确定类别数量
    num_classes = max(label_mapping.values()) + 1  # 加1，因为0通常表示背景
  
    # 6.初始化one-hot编码数组，形状为 [b, num_classes, h, w, d]
    one_hot_labels = np.zeros((real_label.shape[0], num_classes, real_label.shape[2], real_label.shape[3], real_label.shape[4]), dtype=np.float32)
    
    print(f"combined_labels:{combined_labels.shape}, one_hot_labels:{one_hot_labels.shape}")
    # 7.遍历每个类，并设置对应位置的值为1
    for i in range(num_classes):
        one_hot_labels[:, i, :, :, :] = (combined_labels == i)
    
    return one_hot_labels



label_process = Compose(
    [
        LoadImaged(keys=["image", "label", "label_raw", "pseudo_lbl"]),
        AddChanneld(keys=["image", "label", "label_raw", "pseudo_lbl"]),
        Orientationd(keys=["image", "label", "label_raw", "pseudo_lbl"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label", "label_raw", "pseudo_lbl"], 
            pixdim=(1.5, 1.5, 1.5), 
            mode=("bilinear", "nearest", "nearest", "nearest"),), # process h5 to here
    ]
)

train_img = []
train_lbl = []
pseudo_lbl = []
train_name = []

for line in open(ORGAN_LIST):
    # key = get_key(line.strip().split()[0])
    # if key in TRANSFER_LIST:
    train_img.append(ORGAN_DATASET_DIR + line.strip().split()[0])
    train_lbl.append(ORGAN_DATASET_DIR + line.strip().split()[1])
    pseudo_lbl.append(line.strip().split()[2])
    train_name.append(line.strip().split()[1].split('.')[0])
data_dicts_train = [{'image': image, 'label': label, 'label_raw': label, 'pseudo_lbl': pseudo_lbl, 'name': name}
            for image, label, pseudo_lbl, name in zip(train_img, train_lbl, pseudo_lbl, train_name)]
print('train len {}'.format(len(data_dicts_train)))


train_dataset = Dataset(data=data_dicts_train, transform=label_process)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKER, 
                            collate_fn=list_data_collate)

for index, batch in enumerate(train_loader):
    x, y, y_raw, pseudo_y, name = batch["image"], batch["label"], batch['label_raw'], batch['pseudo_lbl'], batch['name']
    # print(name)
    # print(f"x:{x.shape}, y:{y.shape}, pseudo_y:{pseudo_y.shape}")
    y = combine_labels(y, pseudo_y, DATASET_NAME)
    # print(f"y:{y.shape}")
    name = batch['name'][0].replace('label', 'post_label')
    post_dir = ORGAN_DATASET_DIR + '/'.join(name.split('/')[:-1])
    store_y = y.astype(np.uint8)
    if not os.path.exists(post_dir):
        os.makedirs(post_dir)
    with h5py.File(ORGAN_DATASET_DIR + name + '.h5', 'w') as f:
        f.create_dataset('post_label', data=store_y, compression='gzip', compression_opts=9)
        f.close()
