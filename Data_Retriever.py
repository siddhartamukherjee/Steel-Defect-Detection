#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
import pandas as pd
import os
import cv2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from albumentations import (HorizontalFlip, Normalize, Compose, Resize)
from albumentations.pytorch import ToTensor


# In[2]:


class ImageData(Dataset):
    def __init__(self, df, data_folder, mean, std, phase):
        self.df = df
        self.root = data_folder
        self.mean = mean
        self.std = std
        self.phase = phase
        self.transforms = self.get_transforms()
        self.fnames = self.df.index.tolist()

    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, idx):
        image_id, mask = self.make_mask(idx)
        image_path = os.path.join(self.root, "train_images",  image_id)
        img = cv2.imread(image_path)
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask'] # 1x256x1600x4
        mask = mask[0].permute(2, 0, 1) # 4x256x1600
        return img, mask

        
    def make_mask(self, row_id):
        '''Given a row index, return image_id and mask (256, 1600, 4) from the dataframe `df`'''
        fname = self.df.iloc[row_id].name
        labels = self.df.iloc[row_id][:4]
        masks = np.zeros((256, 1600, 4), dtype=np.float32) # float32 is V.Imp
        # 4:class 1～4 (ch:0～3)
        
        for idx, label in enumerate(labels.values):
            if type(label) is str:
                label = label.split(" ")
                positions = map(int, label[0::2])
                length = map(int, label[1::2])
                mask = np.zeros(256 * 1600, dtype=np.uint8)
                for pos, le in zip(positions, length):
                    mask[pos:(pos + le)] = 1
                masks[:, :, idx] = mask.reshape(256, 1600, order='F')
        return fname, masks
    
    def get_transforms(self):
        list_transforms = []
        if self.phase == "train":
            list_transforms.extend(
                [
                    HorizontalFlip(p=0.5), # only horizontal flip as of now
                ]
            )
        list_transforms.extend(
            [
                Normalize(mean=self.mean, std=self.std, p=1),
                Resize(224,224),
                ToTensor(),
            ]
        )
        list_trfms = Compose(list_transforms)
        return list_trfms


# In[ ]:




