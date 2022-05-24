import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from skimage import io
from torchvision import models, transforms


file_path = 'C:/Users/cuomo_g6z0s1c/PycharmProjects/BachelorThesis/Pictures/01_40mm/20220330-165552-image-0000000.BMP'
transform = transforms.ToTensor()

img = io.imread(file_path)
img = transform(img)

"""
conv1 = nn.Conv2d(3, 6, 5)
conv2 = nn.Conv2d(6, 12, 5)
conv3 = nn.Conv2d(12, 24, 5)
pool = nn.MaxPool2d(2, 2)

out = pool(conv1(img))
out = pool(conv2(out))
out = pool(conv3(out))
print(out.shape)
"""
########################################################################################################################
# creat index list
########################################################################################################################

case = 2

def makeSeven(word):
    """creat a strin with the number of the ind array that can be used for the file path of the picture"""
    for i in range(100):
        zero = '0'
        word = zero + word
        if len(word)==7:
            return word
"""
folder = 'C:/Users/cuomo_g6z0s1c/PycharmProjects/BachelorThesis/LabelDepth'
file = ['/Depth20mm.csv', '/Depth30mm.csv', '/Depth40mm.csv']


depth = pd.read_csv(folder+file[case], delimiter=',', skiprows=1, names=['Index','Depth'], index_col=0)
row, col = depth.shape
df = pd.DataFrame(columns=['Index', 'Depth'],index=np.arange(row))





frame_length = ['03_20mm', '02_30mm', '01_40mm']
zusatz = ['170814', '170208', '165552']
path_start = '/Users/cuomo_g6z0s1c/PycharmProjects/BachelorThesis/Pictures/'+frame_length[case]+'/20220330-'+zusatz[case]+'-image-'
path_end = '.BMP'


for i in range(row):
    df.iloc[i]['Index'] = path_start+makeSeven(str(int(depth.iloc[i]['Index'])))+path_end
    df.iloc[i]['Depth'] = depth.iloc[i]['Depth']

#print(len(df))


file = ['/20mm', '/30mm', '/40mm']
df.to_csv('C:/Users/cuomo_g6z0s1c/PycharmProjects/BachelorThesis/NewLabelDepth'+file[case])

df = pd.read_csv('C:/Users/cuomo_g6z0s1c/PycharmProjects/BachelorThesis/NewLabelDepth'+file[case], index_col=0)
print(len(df['Depth']))
"""
########################################################################################################################
# Length Data frame
########################################################################################################################

"""
folder = 'C:/Users/cuomo_g6z0s1c/PycharmProjects/BachelorThesis/LengthLabel'
file = ['/20mm.csv', '/30mm.csv', '/40mm.csv']

length = pd.read_csv(folder+file[case], delimiter=';', skiprows=1, names=['Index','Length'])
row, col = length.shape
df = pd.DataFrame(columns=['Index', 'Length'],index=np.arange(row))

frame_length = ['03_20mm', '02_30mm', '01_40mm']
zusatz = ['170814', '170208', '165552']
path_start = '/Users/cuomo_g6z0s1c/PycharmProjects/BachelorThesis/Pictures/'+frame_length[case]+'/20220330-'+zusatz[case]+'-image-'
path_end = '.BMP'

for i in range(row):
    df.iloc[i]['Index'] = path_start+makeSeven(str(int(length.iloc[i]['Index'])))+path_end
    df.iloc[i]['Length'] = length.iloc[i]['Length']



file = ['/20mm', '/30mm', '/40mm']
df.to_csv('C:/Users/cuomo_g6z0s1c/PycharmProjects/BachelorThesis/NewLengthLabel'+file[case])

df = pd.read_csv('C:/Users/cuomo_g6z0s1c/PycharmProjects/BachelorThesis/NewLengthLabel'+file[case], index_col=0)
print(df)
"""
########################################################################################################################
# create class
########################################################################################################################



class ImageSquats(Dataset):
    def __init__(self, df, transform=None, target_transform=None):
        self.img_labels = df['Depth']
        self.img_dir = df['Index']
        self.transform = transform
        self.target_transform = target_transform


    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, item):
        img_path = self.img_dir.iloc[item]
        img = io.imread(img_path)

        depth = self.img_labels.iloc[item]


        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            depth = self.target_transform(depth)

        return img, depth




