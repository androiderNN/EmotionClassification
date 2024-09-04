import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
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
