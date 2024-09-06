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
        label_array = np.zeros((len(label_list), self.num_label))
        for i, label in enumerate(label_list):
            label_array[i, self.encode_dict[label]] = 1

        return label_array

    def decode(self, label_array):
        '''
        encodeされたndarrayを渡すとラベル名のリストに変換する'''
        label_array = np.argmax(label_array, axis=1)
        label_list = [self.decode_dict(label_list[l]) for l in label_array]
        return label_list

class customDataset(Dataset):
    def __init__(self, image, label, index):
        self.image_array = image[index]
        self.label_array = label[index]

    def __len__(self):
        return len(self.label_array)
    
    def __getitem__(self, idx):
        return {'image':self.image_array[idx], 'label': self.label_array[idx]}

def get_dataloaders(train_size=0.8, valid_size=0.1, test_size=0.1, batch_size=10):
    # 画像の読み込みと格納
    image_names = os.listdir(config.train_path)
    image_names.sort()
    image_array = [read_image(os.path.join(config.train_path, i), mode=ImageReadMode.GRAY)/255 for i in image_names]

    # ラベルの読み込みとone-hot encoding
    table = pd.read_table(os.path.join(config.data_path, 'train_master.tsv'))
    label_array = labelconverter(table['expression'].tolist())

    # インデックス分割
    index = [i for i in range(len(image_names))]
    train_index, index = train_test_split(index, train_size=train_size, shuffle=True)
    valid_index, test_index = train_test_split(index, train_size=valid_size, valid_size=test_size, shuffle=True)

    # dataloader作成
    index_array = [train_index, valid_index, test_index]
    dataloaders = [DataLoader(customDataset(image_array, label_array, i), batch_size=batch_size) for i in index_array]

    return dataloaders
