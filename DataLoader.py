import tifffile, os
import torch
from os.path import join
from skimage import exposure
import numpy as np
from torch import nn as nn
import torch

def MaxProject(img):
    imgs = []
    for z in range(2):
        for y in range(2):
            for x in range(2):
                imgs.append(img[z::2, y::2, x::2])
    img = np.max(imgs, axis=0)
    return img

class GetMultiTypeMemoryDataSetAndCropQxz:
    def __init__(self, path, txtName, imgSize):
        # self.backThre = 0.3
        self.imgSize = imgSize
        self.pSum = self.imgSize[0] * self.imgSize[1] * self.imgSize[2]
        self.imgPath = join(path, 'image')
        self.maskPath = join(path, 'mask')
        with open(join(path, txtName), 'r') as f:
            self.nameLs = f.read().strip().split('\n')

    def __len__(self): return len(self.nameLs)

    def __getitem__(self, ind):
        img = tifffile.imread(join(self.imgPath, self.nameLs[ind]))[:self.imgSize[2], :self.imgSize[1], :self.imgSize[0]].astype(np.float32)
        img = np.expand_dims(img, axis=0).astype(np.float32)
        img = torch.from_numpy(img)
        # img = (img - img.min()) / (img.max() - img.min())
        mask = tifffile.imread(join(self.maskPath, self.nameLs[ind]))[:self.imgSize[2], :self.imgSize[1], :self.imgSize[0]] / 255.0
        mask = np.expand_dims(mask, axis=0).astype(np.float32)
        return img, mask, self.nameLs[ind]
