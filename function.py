import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from skimage import io
from torchvision import models, transforms
from torch.utils.data import Dataset

def get_time(sample):

    time_string = sample[11:]
    hours = time_string[:2]
    hours = float(hours)
    min = time_string[3:5]
    min = float(min)
    sec = time_string[6:]
    sec = float(sec)

    time = hours*3600 + min*60 +sec

    return time

def get_time_camera(sample):

    time_string = sample[11:]
    hours = time_string[:2]
    hours = float(hours)
    min = time_string[3:4]
    min = float(min)
    sec = time_string[5:]
    sec = float(sec)

    time = hours*3600 + min*60 +sec

    return time

def get_index(df):
    dist = df.Distance
    marks = dist < 4
    index = 0
    liste = []

    for i in marks:
        index += 1
        if i:
            liste.append(index)

    first_index_marks = [liste[0]]

    for i in range(len(liste) - 1):
        if liste[i] + 10000 < liste[i + 1]:
            first_index_marks.append(liste[i + 1])

    return first_index_marks

def step_size(df, u_ring):
    mark_index = get_index(df)

    distance = df['Distance']
    laser_full_rotation = distance[mark_index[0]:mark_index[-1]]
    size = u_ring / len(laser_full_rotation)

    return size

def get_overlap(u, frame, pic_number, extra):
    overlap = []
    for i in range(len(frame)):
        overlap.append(1-(u+extra[i]-frame[i])/(frame[i]*(pic_number[i]-1)))
    return overlap

def get_appropriate(data_length, start_window, picture_number, range_number):

    for i in range(range_number):
        if (data_length-(start_window+i))%(picture_number-1)==0:
            window = start_window+i
            laser_step = int((data_length-window)/(picture_number-1))
            return window, laser_step



def prototyp(data_length, full_data,window, laser_step,end,step_size):

    rows = int((data_length - window) / laser_step + 1)
    df_window = pd.DataFrame(columns=np.arange(0, window), index=np.arange(0, rows))

    for i in range(rows):
        df_window.iloc[i] = full_data[i * laser_step:window + i * laser_step]

        if i == rows-1:
            print('error', (end-(window + i * laser_step))*step_size)

    return df_window

def normalize(df):
    df_scaled = preprocessing.normalize(df)
    df_scaled = pd.DataFrame(df_scaled)
    return df_scaled

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def expand(df, ind):
    ad_col = ['min', 'max', 'mean', 'std']  # names of features that will be added
    ad_feature = pd.DataFrame(index=ind, columns=ad_col)

    for i in ind:
        for j in ad_col:
            if j == 'min':
                ad_feature.loc[i][j] = df.loc[i].min()
            if j == 'max':
                ad_feature.loc[i][j] = df.loc[i].max()
            if j == 'mean':
                ad_feature.loc[i][j] = df.loc[i].mean()
            if j == 'std':
                ad_feature.loc[i][j] = df.loc[i].std()

    return ad_feature

########################################################################################################################
# create class
########################################################################################################################

class Image(Dataset):
    def __init__(self, df, mean, std, transform=None):
        self.mean = mean
        self.std = std
        self.path = df['Index']
        self.label = df['Length']
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):

        img_path = self.path.iloc[item]
        img = io.imread(img_path)

        label = (torch.tensor(np.array(self.label.iloc[item]), dtype=torch.float32)-self.mean)/self.std
        #label = torch.tensor(np.array(self.label.iloc[item]), dtype=torch.float32)

        if self.transform:
            img = self.transform(img)

        return img, label