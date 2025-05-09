import os
from scipy.io import loadmat
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets.SequenceDatasets import dataset
from datasets.sequence_aug import *
from tqdm import tqdm
from itertools import islice
from collections import OrderedDict



#Digital data was collected at 12,000 samples per second
signal_size = 1024
work_condition=['_20_0.csv','_30_2.csv']
dataname= {0:[os.path.join('bearingset','health'+work_condition[0]),
              os.path.join('gearset','Health'+work_condition[0]),
              os.path.join('bearingset','ball'+work_condition[0]),
              os.path.join('bearingset','outer'+work_condition[0]),
              os.path.join('bearingset', 'inner' + work_condition[0]),
              os.path.join('bearingset', 'comb' + work_condition[0]),
              os.path.join('gearset', 'Chipped' + work_condition[0]),
              os.path.join('gearset', 'Miss' + work_condition[0]),
              os.path.join('gearset', 'Surface' + work_condition[0]),
              os.path.join('gearset', 'Root' + work_condition[0]),
              ],
         1:[os.path.join('bearingset','health'+work_condition[1]),
              os.path.join('gearset','Health'+work_condition[1]),
              os.path.join('bearingset','ball'+work_condition[1]),
              os.path.join('bearingset','outer'+work_condition[1]),
              os.path.join('bearingset', 'inner' + work_condition[1]),
              os.path.join('bearingset', 'comb' + work_condition[1]),
              os.path.join('gearset', 'Chipped' + work_condition[1]),
              os.path.join('gearset', 'Miss' + work_condition[1]),
              os.path.join('gearset', 'Surface' + work_condition[1]),
              os.path.join('gearset', 'Root' + work_condition[1]),
              ]
          }

# label = [i for i in range(0, 9)]
# 定义统一的多标签编码系统
fault_labels = OrderedDict([
    # 健康状态 - 全0编码
    ('health',  [0, 0, 0, 0, 0, 0, 0]),  # 轴承健康状态
    ('Health',  [0, 0, 0, 0, 0, 0, 0]),  # 齿轮健康状态
    
    # 轴承故障标签 [滚动体, 外圈, 内圈, 复合, 缺损, 断齿, 齿面, 齿根]
    ('ball',    [1, 0, 0, 0, 0, 0, 0]),  # 滚动体故障
    ('outer',   [0, 1, 0, 0, 0, 0, 0]),  # 外圈故障
    ('inner',   [0, 0, 1, 0, 0, 0, 0]),  # 内圈故障
    ('comb',    [0, 1, 1, 0, 0, 0, 0]),  # 内外圈复合故障
    
    # 齿轮故障标签
    ('Chipped', [0, 0, 0, 1, 0, 0, 0]),  # 齿轮缺损
    ('Miss',    [0, 0, 0, 0, 1, 0, 0]),  # 齿轮断齿
    ('Surface', [0, 0, 0, 0, 0, 1, 0]),  # 齿面磨损
    ('Root',    [0, 0, 0, 0, 0, 0, 1]),  # 齿根损坏
])


def get_files(root, N):
    '''
    This function is used to generate the final training set and test set.
    root: The location of the data set
    N: Working conditions to use
    '''
    data = []
    lab = []
    for k in range(len(N)):
        for n in tqdm(range(len(dataname[N[k]])), desc=f'Loading condition {N[k]}'):
            path1 = os.path.join(root, dataname[N[k]][n])
            # 从文件路径中提取故障类型名称
            fault_type = os.path.basename(path1).split('_')[0]
            # 从fault_labels中获取对应的多标签编码
            data1, lab1 = data_load(path1, fault_type)
            data += data1
            lab += lab1
    return [data, lab]


def data_load(filename, fault_type):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    axisname:Select which channel's data,---->"_DE_time","_FE_time","_BA_time"
    '''
    # 使用pandas读取csv文件，跳过前16行
    if "ball_20_0.csv" in filename:
        df = pd.read_csv(filename, skiprows=16, usecols=[1], header=None)
    else:
        df = pd.read_csv(filename, skiprows=16, sep='\t', usecols=[1], header=None)
    
    # 转换为numpy数组并reshape
    fl = df.values.reshape(-1)
    
    data = []
    lab = []
    
    # Get multi-label encoding for current fault type
    label = fault_labels[fault_type]

    start, end = int(fl.shape[0]/2), int(fl.shape[0]/2)+signal_size
    while end<=(int(fl.shape[0]/2)+int(fl.shape[0]/3)):
        x = fl[start:end]
        x = np.fft.fft(x)
        x = np.abs(x) / len(x)
        x = x[range(int(x.shape[0] / 2))]
        x = x.reshape(-1,1)
        data.append(x)
        lab.append(label)
        start +=signal_size
        end +=signal_size
    return data, lab

#--------------------------------------------------------------------------------------------------------------------
class Md(object):
    num_classes = 7
    inputchannel = 1
    def __init__(self, data_dir, transfer_task, normlizetype="0-1"):
        self.data_dir = data_dir
        self.source_N = transfer_task[0]
        self.target_N = transfer_task[1]
        self.normlizetype = normlizetype
        self.data_transforms = {
            'train': Compose([
                Reshape(),
                Normalize(self.normlizetype),
                # RandomAddGaussian(),
                # RandomScale(),
                # RandomStretch(),
                # RandomCrop(),
                Retype(),
                # Scale(1)
            ]),
            'val': Compose([
                Reshape(),
                Normalize(self.normlizetype),
                Retype(),
                # Scale(1)
            ])
        }

    def data_split(self, transfer_learning=True):
        if transfer_learning:
            # get source train and val
            list_data = get_files(self.data_dir, self.source_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            source_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            source_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])

            # get target train and val
            list_data = get_files(self.data_dir, self.target_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            target_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            target_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])
            return source_train, source_val, target_train, target_val
        else:
            #get source train and val
            list_data = get_files(self.data_dir, self.source_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            source_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            source_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])

            # get target train and val
            list_data = get_files(self.data_dir, self.target_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            target_val = dataset(list_data=data_pd, transform=self.data_transforms['val'])
            return source_train, source_val, target_val