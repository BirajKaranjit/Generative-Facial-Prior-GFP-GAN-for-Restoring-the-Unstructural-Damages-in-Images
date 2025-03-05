#Dataloader

from torch.utils.data import Dataset
import torch.distributed as dist


class DatasetBase(Dataset):
    def __init__(self,slice_id=0,slice_count=1,use_dist=False,**kwargs):

        if use_dist:
            slice_id = dist.get_rank()
            slice_count = dist.get_world_size()
        self.id = slice_id
        self.count = slice_count

    def __getitem__(self,i):
        pass
    

    def __len__(self):
        return 1000
    

#DataLoader
import os
from torchvision import transforms
import torchvision.transforms.functional as TF
import PIL.Image as Image
from torchvision.transforms.functional import  normalize
import random
import math
import pickle
import numpy as np
import torch
# from utils import *
import cv2

def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


class GFPData(DatasetBase):
    def __init__(self, slice_id=0, slice_count=1,dist=False, **kwargs):
        super().__init__(slice_id, slice_count,dist, **kwargs)
        self.eval = kwargs['eval']
        self.mean = kwargs['mean']
        self.std = kwargs['std']
        self.out_size = kwargs['size']
        print("Keys available in kwargs:", kwargs.keys())  # Check what keys are in kwargs

        if self.eval:
           # Validation mode: Use val_hq_root and val_lq_root
            self.hq_root = kwargs['val_hq_root']
            self.lq_root = kwargs['val_lq_root']

        else:
           # Training mode: Use train_hq_root and train_lq_root
            self.hq_root = kwargs['train_hq_root']
            self.lq_root = kwargs['train_lq_root']

        # Get image paths
        self.hq_paths = self.get_img_paths(self.hq_root)
        self.length = len(self.hq_paths)

        # Shuffle paths
        random.shuffle(self.hq_paths)

    def __getitem__(self,i):
        hq_path = self.hq_paths[i%self.length]
        lq_path = os.path.join(self.lq_root,os.path.basename(hq_path))

     # Check if paths exist
        if not os.path.exists(hq_path):
            print(f"Error: HQ path does not exist: {hq_path}")
            # Return a default black image tensor
            return torch.zeros((3, self.out_size, self.out_size), dtype=torch.float32), torch.zeros((3, self.out_size, self.out_size), dtype=torch.float32)

        if not os.path.exists(lq_path):
              print(f"Error: LQ path does not exist: {lq_path}")
              # Return a default black image tensor
              return torch.zeros((3, self.out_size, self.out_size), dtype=torch.float32), torch.zeros((3, self.out_size, self.out_size), dtype=torch.float32)

        # Load images, handle potential errors
        try:
            img_hq = cv2.imread(hq_path).astype(np.float32) / 255.0
            img_lq = cv2.imread(lq_path).astype(np.float32) / 255.0

            if img_hq is None or img_lq is None:
              print(f"Error loading image(s) at hq_path: {hq_path} or lq_path: {lq_path}")
              return torch.zeros((3, self.out_size, self.out_size), dtype=torch.float32), torch.zeros((3, self.out_size, self.out_size), dtype=torch.float32)
        except Exception as e:
            print(f"Error loading image(s) at hq_path: {hq_path} or lq_path {lq_path} - {e}")
            # Return a default black image tensor
            return torch.zeros((3, self.out_size, self.out_size), dtype=torch.float32), torch.zeros((3, self.out_size, self.out_size), dtype=torch.float32)

        # resize to original size
        img_lq = cv2.resize(img_lq, (512, 512), interpolation=cv2.INTER_LINEAR)
        img_hq = cv2.resize(img_hq, (1024, 1024))

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_hq, img_lq = img2tensor([img_hq, img_lq], bgr2rgb=True, float32=True)

        # round and clip
        img_lq = torch.clamp((img_lq * 255.0).round(), 0, 255) / 255.

        # normalize
        img_hq = normalize(img_hq, self.mean, self.std, inplace=True)
        img_lq = normalize(img_lq, self.mean, self.std, inplace=True)


        return img_lq, img_hq

    def get_img_paths(self, root):
            """Get paths of all images in the directory."""
            if root is None:
                print(f"Warning: Image root path is None")
                return []
            if not os.path.exists(root):
                print(f"Warning: Image root path does not exist: {root}")
                return []
            return [os.path.join(root, f) for f in os.listdir(root) if f.endswith(('.png', '.jpg', '.jpeg'))]


    def __len__(self):
        return self.length