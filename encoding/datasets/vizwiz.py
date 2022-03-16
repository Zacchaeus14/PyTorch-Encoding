###########################################################################
# Created by: Yuchen Wang
# Email: yw3642@nyu.edu
# Copyright (c) 2022
###########################################################################

import os
import sys
import numpy as np
import random
import math
from PIL import Image, ImageOps, ImageFilter
from glob import glob

import torch
import torch.utils.data as data
import torchvision.transforms as transform

from .base import BaseDataset

class VIZWIZSegmentation(BaseDataset):
    BASE_DIR = 'VizWizGrounding2022'
    NUM_CLASS = 2
    def __init__(self, root=os.path.expanduser('~/.encoding/data'), split='train',
                 mode=None, transform=None, target_transform=None, **kwargs):
        super(VIZWIZSegmentation, self).__init__(
            root, split, mode, transform, target_transform, **kwargs)
        # assert exists and prepare dataset automatically
        root = os.path.join(root, self.BASE_DIR)
        assert os.path.exists(root), "Dataset folder not found!"
        self.images, self.masks = _get_wizviz_pairs(root, split)
        if split != 'test':
            assert (len(self.images) == len(self.masks))
        if len(self.images) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: \
                " + root + "\n"))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.masks[index])
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            mask = self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        return img, mask

    #def _sync_transform(self, img, mask):
    #    # random mirror
    #    if random.random() < 0.5:
    #        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    #        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    #    crop_size = self.crop_size
    #    # random scale (short edge)
    #    w, h = img.size
    #    long_size = random.randint(int(self.base_size*0.5), int(self.base_size*2.0))
    #    if h > w:
    #        oh = long_size
    #        ow = int(1.0 * w * long_size / h + 0.5)
    #        short_size = ow
    #    else:
    #        ow = long_size
    #        oh = int(1.0 * h * long_size / w + 0.5)
    #        short_size = oh
    #    img = img.resize((ow, oh), Image.BILINEAR)
    #    mask = mask.resize((ow, oh), Image.NEAREST)
    #    # pad crop
    #    if short_size < crop_size:
    #        padh = crop_size - oh if oh < crop_size else 0
    #        padw = crop_size - ow if ow < crop_size else 0
    #        img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    #        mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
    #    # random crop crop_size
    #    w, h = img.size
    #    x1 = random.randint(0, w - crop_size)
    #    y1 = random.randint(0, h - crop_size)
    #    img = img.crop((x1, y1, x1+crop_size, y1+crop_size))
    #    mask = mask.crop((x1, y1, x1+crop_size, y1+crop_size))
    #    # gaussian blur as in PSP
    #    if random.random() < 0.5:
    #        img = img.filter(ImageFilter.GaussianBlur(
    #            radius=random.random()))
    #    # final transform
    #    return img, self._mask_transform(mask)

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int64') - 1
        return torch.from_numpy(target)

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 1


def _get_wizviz_pairs(folder, split='train'):
    def get_path_pairs(img_folder, mask_folder):
        if not mask_folder:
            return glob(os.path.join(img_folder, '*.jpg')), None
        mask_paths = glob(os.path.join(mask_folder, '*.png'))
        names = [_get_name_from_path(path) for path in mask_paths]
        names = [x.split('.')[0] for x in names]
        img_paths = [os.path.join(img_folder, f'{name}.jpg') for name in names]
        assert all(os.path.isfile(img_path) for img_path in img_paths)
        return img_paths, mask_paths

    if split == 'train':
        img_folder = os.path.join(folder, 'train')
        mask_folder = os.path.join(folder, 'binary_masks_png/train')
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
        print('len(img_paths):', len(img_paths))
        assert len(img_paths) == 6494
    elif split == 'val':
        img_folder = os.path.join(folder, 'val')
        mask_folder = os.path.join(folder, 'binary_masks_png/val')
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
        assert len(img_paths) == 1131
    elif split == 'trainval':
        train_img_folder = os.path.join(folder, 'train')
        train_mask_folder = os.path.join(folder, 'binary_masks_png/train')
        val_img_folder = os.path.join(folder, 'val')
        val_mask_folder = os.path.join(folder, 'binary_masks_png/val')
        train_img_paths, train_mask_paths = get_path_pairs(train_img_folder, train_mask_folder)
        val_img_paths, val_mask_paths = get_path_pairs(val_img_folder, val_mask_folder)
        img_paths = train_img_paths + val_img_paths
        mask_paths = train_mask_paths + val_mask_paths
        assert len(img_paths) == 6494 + 1131
    else:
        assert split == 'test', 'split unknown'
        img_folder = os.path.join(folder, 'test')
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
        assert len(img_paths) == 8000
    return img_paths, mask_paths

def _get_name_from_path(path):
    return os.path.basename(os.path.normpath(path))

if __name__ == '__main__':
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt
    SPLIT = 'val'
    norm_mean= [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
    train_transform = [
                transforms.ToTensor(),
                # transforms.Normalize(norm_mean, norm_std),
            ]
    train_transform = transforms.Compose(train_transform)
    kwargs = {'root': '../../../datasets/', 
            'split': SPLIT, 
            'mode': 'train', 
            'transform': train_transform, 
            'base_size': 520,
            'crop_size': 480}
    trainset = VIZWIZSegmentation(**kwargs)
    img0, msk0 = trainset[0]
    plt.imshow(img0.permute(1, 2, 0))
    print(torch.unique(msk0))
    plt.show()
    plt.imshow(msk0)
    plt.show()