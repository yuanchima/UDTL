import os
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from datasets.SequenceDatasets import dataset
from datasets.sequence_aug import *
from tqdm import tqdm


signal_size = 1024

#1 Undamaged (healthy) bearings(6X)
HBdata = ['K001',"K002",'K003','K004','K005','K006']

#2 Artificially damaged bearings(12X)
ADBdata = ['KA01','KA03','KA05','KA06','KA07','KA08','KA09','KI01','KI03','KI05','KI07','KI08']

#3 Bearings with real damages caused by accelerated lifetime tests(14x)
RDBdata = ['KA04','KA15','KA16','KA22','KA30','KB23','KB24','KB27','KI14','KI16','KI17','KI18','KI21']
#RDBdata = ['KA16','KA22','KA30','KB23','KB27','KI14','KI17','KI18']
# label3=[i for i in range(len(RDBdata))]

# 定义多标签编码映射
bearing_labels = {
    # 故障轴承（Faulty bearings） - 6个故障特征
    'KA04': [1, 0, 1, 0, 1, 0],  # [疲劳点蚀, 塑性变形, 外圈, 内圈, 单点损伤, 分布式损伤]
    'KA15': [0, 1, 1, 0, 1, 0],  
    'KA16': [1, 0, 1, 0, 1, 0],
    'KA22': [1, 0, 1, 0, 1, 0],
    'KA30': [0, 1, 1, 0, 0, 1],
    'KB23': [1, 0, 1, 1, 1, 0],  # 内外圈都标记为1
    'KB24': [1, 0, 1, 1, 0, 1],
    'KB27': [0, 1, 1, 1, 0, 1],
    'KI14': [1, 0, 0, 1, 1, 0],
    'KI16': [1, 0, 0, 1, 1, 0],
    'KI17': [1, 0, 0, 1, 1, 0],
    'KI18': [1, 0, 0, 1, 1, 0],
    'KI21': [1, 0, 0, 1, 1, 0],
    # 健康轴承 (Healthy bearings) - 所有故障特征为0
    'K001': [0, 0, 0, 0, 0, 0],  # [疲劳点蚀, 塑性变形, 外圈, 内圈, 单点损伤, 分布式损伤]
    'K002': [0, 0, 0, 0, 0, 0],
    'K003': [0, 0, 0, 0, 0, 0],
    'K004': [0, 0, 0, 0, 0, 0],
    'K005': [0, 0, 0, 0, 0, 0],
    'K006': [0, 0, 0, 0, 0, 0]
}

#working condition
WC = ["N15_M07_F10","N09_M07_F10","N15_M01_F10","N15_M07_F04"]
#state = WC[0] #WC[0] can be changed to different working states
def get_files(root, N):
    '''
    This function is used to generate the final training set and test set.
    root: The location of the data set
    N: List of working conditions to use
    '''
    data = []
    lab = []
    for i in range(len(N)):
        state = WC[N[i]]  # Get current working condition

        # Load healthy bearing data
        for bearing_code in tqdm(HBdata, desc='Loading healthy bearings'):
            for w1 in range(20):
                name = f"{state}_{bearing_code}_{w1+1}"
                path = os.path.join('/tmp', root, bearing_code, f"{name}.mat")
                data1, lab1 = data_load(path, name=name, bearing_code=bearing_code)
                data += data1
                lab += lab1

        # Load real damaged bearing data
        for bearing_code in tqdm(RDBdata, desc='Loading damaged bearings'):
            for w3 in range(20):
                name = f"{state}_{bearing_code}_{w3+1}"
                path = os.path.join('/tmp', root, bearing_code, f"{name}.mat")
                data3, lab3 = data_load(path, name=name, bearing_code=bearing_code)
                data += data3
                lab += lab3

    return [data, lab]

def data_load(filename, name, bearing_code):
    '''
    This function is mainly used to generate test data and training data.
    filename: Data location
    name: File name
    bearing_code: Bearing code for label lookup
    '''
    fl = loadmat(filename)[name]
    fl = fl[0][0][2][0][6][2]  # Take out the data
    fl = fl.reshape(-1,1)
    data = []
    lab = []
    start, end = 0, signal_size
    
    # Get the multi-label for this bearing from bearing_labels dictionary
    label = bearing_labels[bearing_code]
    
    while end <= fl.shape[0]:
        data.append(fl[start:end])
        lab.append(label)
        start += signal_size
        end += signal_size

    return data, lab

#--------------------------------------------------------------------------------------------------------------------
class PU(object):
    num_classes = 6
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

