"""
# @Time    : 2023/2/8 13:43
# @File    : dataloader.py
# @Author  : rezheaiba
"""
import csv
import os

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms


class Loader(Dataset):
    def __init__(self, img_path, label_path, root, transforms_=None):
        self.name = []
        self.label = []
        label_dict = {}
        with open(label_path, encoding='utf-8-sig') as f:
            for line in f.readlines():
                items = line.rstrip().split()
                label_dict[items[0]] = int(items[-1]) -1  # -1为了和2分类的室内外索引位置对齐
        with open(img_path, encoding='utf-8-sig') as f:
            for row in f.readlines():
                self.name.append(str(row.rstrip()))
                self.label.append(int(label_dict[str(row).split('/')[-2]]))
        self.root = root
        if transforms_ is not None:
            self.transform = transforms_

    def __getitem__(self, index):
        path = os.path.join(self.root, self.name[index])
        img = Image.open(path).convert("RGB")
        label = self.label[index]

        if self.transform is not None:
            img = self.transform(img)
        return {"image": img,
                "label": torch.tensor(label, dtype=torch.int64),
                }

    def __len__(self):
        return len(self.name)



