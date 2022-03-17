###########################################################################
# Created by: Yuchen Wang
# Email: yw3642@nyu.edu
# Copyright (c) 2022
###########################################################################

import json
import os
from glob import glob

import torch
from PIL import Image

from base import BaseDataset


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
        self.questions = []
        with open(os.path.join(root, f'{split}_grounding.json'), 'r') as f:
            grounding = json.load(f)
        for i, img_path in enumerate(self.images):
            self.questions.append(grounding[_get_name_from_path(img_path)]['question'])
        if split != 'test':
            assert len(set([len(self.images), len(self.masks), len(self.questions)])) == 1
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
        return img, mask, self.questions[index]

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
        img_paths, mask_paths = get_path_pairs(img_folder, None)
        assert len(img_paths) == 8000
    return img_paths, mask_paths

def _get_name_from_path(path):
    return os.path.basename(os.path.normpath(path))

if __name__ == '__main__':
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt
    SPLIT = 'train'
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
    img0, msk0, q0 = trainset[0]
    plt.imshow(img0.permute(1, 2, 0))
    print('mask contains', torch.unique(msk0))
    print('question', q0)
    plt.show()
    plt.imshow(msk0)
    plt.show()