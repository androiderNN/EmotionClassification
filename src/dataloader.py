import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import ImageReadMode, read_image
import matplotlib.pyplot as plt

import config

class labelconverter():
    def __init__(self, labels):
        '''
        ラベルの変換を行うクラス
        ラベルの対応関係を保持しone-hot-encodingを行う
        '''
        labels = set(labels)
        labels = list(labels)
        labels.sort()
        self.num_label = len(labels)
        self.decode_dict = {i:labels[i] for i in range(self.num_label)}
        self.encode_dict = {value:key for key, value in self.decode_dict.items()}
    
    def encode(self, label_list):
        '''
        ラベル名のリストを渡すとencodingした(要素数,ラベル数)のndarrayを返す'''
        # label_array = np.zeros((len(label_list), self.num_label))
        # for i, label in enumerate(label_list):
        #     label_array[i, self.encode_dict[label]] = 1

        ##
        label_array = label_list
        for value, i in self.encode_dict.items():
            label_array = [i if l==value else l for l in label_array]

        return label_array

    def decode(self, label_array):
        '''
        encodeされたndarrayを渡すとラベル名のリストに変換する'''
        # label_array = np.argmax(label_array, axis=1)
        # label_list = [self.decode_dict(label_list[l]) for l in label_array]
        label_list = list(label_array)
        for i, value in self.encode_dict.items():
            label_list = [value if l==i else l for l in label_list]
        return label_list

class customDataset(Dataset):
    def __init__(self, image, label, index):
        self.image_array = [image for idx, image in enumerate(image) if idx in index]
        self.label_array = [label for idx, label in enumerate(label) if idx in index]

    def __len__(self):
        return len(self.label_array)
    
    def __getitem__(self, idx):
        image = self.image_array[idx]
        label = self.label_array[idx]
        # return {'image':image, 'label': label}
        return (image, label)

def get_dataloaders(train_size=0.7, valid_size=0.3, test_size=1e-10, batch_size=10):
    if train_size+valid_size+test_size>(1+1e-10):
        raise ValueError

    # 画像の読み込みと格納
    image_names = os.listdir(config.train_path)
    image_names.sort()
    image_array = [read_image(os.path.join(config.train_path, i), mode=ImageReadMode.GRAY)/255 for i in image_names]

    # ラベルの読み込みとone-hot encoding
    table = pd.read_table(os.path.join(config.data_path, 'train_master.tsv'))
    label = table['expression'].tolist()
    lc = labelconverter(label)
    label_array = lc.encode(label)

    # インデックス分割
    index = [i for i in range(len(image_names))]
    train_index, index = train_test_split(index, train_size=train_size, shuffle=True)
    valid_index, test_index = train_test_split(index, train_size=valid_size, test_size=test_size, shuffle=True)

    # dataloader作成
    index_array = [train_index, valid_index, test_index]
    datasets = [customDataset(image_array, label_array, i) for i in index_array]
    dataloaders = [DataLoader(ds, batch_size) for ds in datasets]

    return dataloaders
