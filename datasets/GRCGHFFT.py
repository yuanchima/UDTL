import os
from scipy.io import loadmat
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets.SequenceDatasets import dataset
from datasets.sequence_aug import *
from tqdm import tqdm

#Digital data was collected at 12,000 samples per second
signal_size = 1024
root1 = "/home/yuanchima/datasets"
dataname= {0:["GRC/Healthy/H1.mat", 
              "GRC/Healthy/H2.mat", 
              "GRC/Healthy/H3.mat", 
              "GRC/Healthy/H4.mat", 
              "GRC/Healthy/H5.mat", 
              "GRC/Healthy/H6.mat", 
              "GRC/Healthy/H7.mat", 
              "GRC/Healthy/H8.mat", 
              "GRC/Healthy/H9.mat", 
              "GRC/Healthy/H10.mat",
              "GRC/Damaged/D1.mat", 
              "GRC/Damaged/D2.mat", 
              "GRC/Damaged/D3.mat", 
              "GRC/Damaged/D4.mat", 
              "GRC/Damaged/D5.mat", 
              "GRC/Damaged/D6.mat", 
              "GRC/Damaged/D7.mat", 
              "GRC/Damaged/D8.mat", 
              "GRC/Damaged/D9.mat", 
              "GRC/Damaged/D10.mat",
              ],  # GRC
           1:[  "gh_data/yxzc-gszjx-s1015-1.csv",
                "gh_data/ysdc-gsz-s1135.csv",
                "gh_data/zcclx-gszjx-s1517-1.csv",
                "gh_data/yxzc-gszjx-s994.csv",
                "gh_data/zqqfdjldjxsd-qddjx-s1546.csv",
                "gh_data/yxzc-gszzx-s929.csv",
                "gh_data/zcclx-gszzx-s914.csv",
                "gh_data/yxzc-gszzx-s1316.csv",
                "gh_data/zqqclms-gsz-s1551.csv",
                "gh_data/zqqclms-gsz-s1404.csv",
                "gh_data/zcclx-gszzx-s1206.csv",
                "gh_data/zcclx-gszzx-s1501.csv",
                "gh_data/zqqfdjldjxsd-fqddjx-s1545.csv",
                "gh_data/zqqfdjldjxsd-qddjx-s1548.csv",
                "gh_data/zqqfdjldjxsd-fqddjx-s1525.csv",
                "gh_data/zqqclms-gsz-s1535.csv",
                "gh_data/zcclx-gszjx-s1223.csv",
                "gh_data/yxzc-gszzx-s1016.csv",
                "gh_data/zcclx-gszjx-s914.csv",
                "gh_data/zcclx-gszzx-s1518.csv",
                "gh_data/ysdc-gsz-s1104.csv",
                "gh_data/yxzc-gszzx-s1013.csv",
                "gh_data/zcclx-gszjx-s1510.csv",
                "gh_data/yxzc-gszzx-s1504.csv",
                "gh_data/yxzc-gszzx-s1016-1.csv",
                "gh_data/zcclx-gszzx-s1516.csv",
                "gh_data/zcclx-gszzx-s1521.csv",
                "gh_data/zcclx-gszjx-s1502.csv",
                "gh_data/yxzc-gszjx-s1502.csv",
                "gh_data/ysdc-gsz-s1128.csv",
                "gh_data/zcclx-gszjx-s1516.csv",
                "gh_data/yxzc-gszzx-s1014.csv",
                "gh_data/yxzc-gszzx-s1503.csv",
                "gh_data/yxzc-gszzx-s1016-2.csv",
                "gh_data/zqqclms-gsz-s1603.csv",
                "gh_data/ysdc-gsz-s1069.csv",
                "gh_data/zqqfdjldjxsd-fqddjx-s1554.csv",
                "gh_data/yxzc-gszzx-s1503-1.csv",
                "gh_data/zqqfdjldjxsd-qddjx-s1529.csv",
                "gh_data/zqqfdjldjxsd-qddjx-s908.csv",
                "gh_data/zqqclms-gsz-s1523.csv",
                "gh_data/zqqfdjldjxsd-qddjx-s1551.csv",
                "gh_data/zcclx-gszjx-s1518.csv",
                "gh_data/yxzc-gszjx-s1018.csv",
                "gh_data/zcclx-gszzx-s1512.csv",
                "gh_data/yxzc-gszjx-s1015-2.csv",
                "gh_data/zcclx-gszzx-s1316.csv",
                "gh_data/yxzc-gszjx-s1015.csv",
                "gh_data/yxzc-gszjx-s1303.csv",
                "gh_data/yxzc-gszzx-s1015-1.csv",
                "gh_data/zcclx-gszzx-s1525.csv",
                "gh_data/zcclx-gszjx-s1520.csv",
                "gh_data/zqqfdjldjxsd-fqddjx-s1530.csv",
                "gh_data/zcclx-gszjx-s1527.csv",
                "gh_data/zqqfdjldjxsd-qddjx-s1541.csv",
                "gh_data/zqqfdjldjxsd-qddjx-s1526.csv",
                "gh_data/zqqfdjldjxsd-qddjx-s1546-1.csv",
                "gh_data/zqqfdjldjxsd-qddjx-s1518.csv",
                "gh_data/zcclx-gszjx-s1283.csv",
                "gh_data/ysdc-gsz-s1152.csv",
                "gh_data/zcclx-gszjx-s1517.csv",
                "gh_data/yxzc-gszjx-s1016.csv",
                "gh_data/yxzc-gszzx-s1018.csv",
                "gh_data/zqqfdjldjxsd-qddjx-s1520.csv",
                "gh_data/ysdc-gsz-s1090.csv",
                "gh_data/zcclx-gszzx-s1508.csv",
                "gh_data/yxzc-gszjx-s1504.csv",
                "gh_data/ysdc-gsz-s1105.csv",
                "gh_data/yxzc-gszjx-s929.csv",
                "gh_data/zqqfdjldjxsd-qddjx-s1545.csv",
                "gh_data/yxzc-gszjx-s1014.csv",
                "gh_data/zcclx-gszzx-s1521-1.csv",
                "gh_data/ysdc-gsz-s1131.csv",
                "gh_data/yxzc-gszjx-s1505.csv",
                "gh_data/yxzc-gszjx-s1014-1.csv",
                "gh_data/yxzc-gszjx-s1013.csv",
                "gh_data/yxzc-gszzx-s1013-1.csv",
                "gh_data/zcclx-gszjx-s1515.csv",
                "gh_data/yxzc-gszzx-s1017.csv",
                "gh_data/yxzc-gszjx-s1017.csv",
                "gh_data/ysdc-gsz-s1153.csv",
                "gh_data/zcclx-gszzx-s931.csv",
                "gh_data/zcclx-gszjx-s928.csv",
                "gh_data/zqqclms-gsz-s949.csv",
                "gh_data/zqqfdjldjxsd-fqddjx-s1548.csv",
                "gh_data/zqqfdjldjxsd-fqddjx-s1539.csv",
                "gh_data/zcclx-gszzx-s1514.csv",
                "gh_data/zqqfdjldjxsd-fqddjx-s1549-1.csv",
                "gh_data/zcclx-gszjx-s1507.csv",
                "gh_data/zqqfdjldjxsd-fqddjx-s1546.csv",
                "gh_data/zqqfdjldjxsd-fqddjx-s1498.csv",
                "gh_data/zqqfdjldjxsd-fqddjx-s1549.csv",
                "gh_data/zcclx-gszzx-s1518-1.csv",
                "gh_data/zqqfdjldjxsd-fqddjx-s1549-2.csv",
                "gh_data/zqqfdjldjxsd-qddjx-s1553.csv",
                "gh_data/yxzc-gszzx-s1015.csv",
                "gh_data/zqqclms-gsz-s1381.csv"]  # RG
                } 

label = [i for i in range(0, 2)]

def get_files_GRC(root, N):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''
    data = []
    lab =[]
    for k in range(len(N)):
        for n in tqdm(range(len(dataname[N[k]]))):
            path1 =os.path.join(root, dataname[N[k]][n])
            if n<10:
                data1, lab1 = data_load_GRC(path1, label=label[0])
            else:
                data1, lab1 = data_load_GRC(path1, label=label[1])
            data += data1
            lab +=lab1

    return [data, lab]

def get_files_GH(root, N):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''
    data = []
    lab =[]
    for k in range(len(N)):
        for n in tqdm(range(len(dataname[N[k]]))):
            path1 =os.path.join(root, dataname[N[k]][n])
            split_list = path1.split('-')
            if "yxzc" in split_list:
                data1, lab1 = data_load_GH(path1, label=label[0])
            else:
                data1, lab1 = data_load_GH(path1, label=label[1])
            data += data1
            lab +=lab1

    return [data, lab]

def data_load_GRC(filename, label):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    axisname:Select which channel's data,---->"_DE_time","_FE_time","_BA_time"
    '''

    fl = loadmat(filename)
    # Separation and enhancement of gear and bearing signals for the diagnosis of wind turbine transmission systems
    data1 = fl["AN7"] # placed on the HSS bearing
    # data1 = np.concatenate(data1, axis=1)
    data = []
    lab = []
    start, end = 0, signal_size

    while end <= data1.shape[0]:
        x = data1[start:end]
        x = np.fft.fft(x)
        x = np.abs(x) / len(x)
        x = x[range(int(x.shape[0] / 2))]
        x = x.reshape(-1,1)
        data.append(x)
        lab.append(label)
        start += signal_size
        end += signal_size

    return data, lab

def data_load_GH(filename, label):
    data1 = pd.read_csv(filename)
    data1 = np.array(data1["Y"])
    data = []
    lab = []
    start, end = 0, signal_size

    while end <= data1.shape[0]:
        if not np.any(np.isnan(data1[start:end])):
            x = data1[start:end]
            x = np.fft.fft(x)
            x = np.abs(x) / len(x)
            x = x[range(int(x.shape[0] / 2))]
            x = x.reshape(-1,1)
            data.append(x)
            lab.append(label)
        start += signal_size
        end += signal_size

    return data, lab

#--------------------------------------------------------------------------------------------------------------------
class GRCGHFFT(object):
    num_classes = 2
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
            list_data = get_files_GRC(self.data_dir, self.source_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            source_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            source_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])

            # get target train and val
            list_data = get_files_GH(self.data_dir, self.target_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            target_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            target_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])
            return source_train, source_val, target_train, target_val
        else:
            #get source train and val
            list_data = get_files_GRC(self.data_dir, self.source_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            source_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            source_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])

            # get target train and val
            list_data = get_files_GH(self.data_dir, self.target_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            target_val = dataset(list_data=data_pd, transform=self.data_transforms['val'])
            return source_train, source_val, target_val


"""
    def data_split(self):

"""