import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from random import sample, random
import scipy.io as sio
import codecs
import os
import os.path



def _dataset_info(txt_labels,folder_dataset):
    with open(txt_labels, 'r') as f:
        images_list = f.readlines()

    file_names = []
    labels = []
    for row in images_list:
        row = row.split(' ')
        file_name = folder_dataset+row[0]
        file_names.append(file_name)
        labels.append(int(row[1]))

    return file_names, labels


def get_split_dataset_info(txt_list,folder_dataset):
    names, labels = _dataset_info(txt_list,folder_dataset)
    return names, labels

class CustomDataset(data.Dataset):
    #自定义实现了torch.utils.data.Dataset类
    def __init__(self, names, labels, img_transformer=None,is_train=None):
        
        self.names = names
        self.labels = labels
        self.N = len(self.names)
        self._image_transformer = img_transformer #图像变换函数
        self.is_train = is_train #是否为训练集
     
    def __getitem__(self, index):
        framename = self.names[index]
        img = Image.open(framename).convert('RGB')#图片转RGB格式

   
        data,label = self._image_transformer(img,self.labels[index], self.is_train)
        return data,label

    def __len__(self):
        return len(self.names)