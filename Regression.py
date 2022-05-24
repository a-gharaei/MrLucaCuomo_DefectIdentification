import pandas as pd
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import shuffle
import torchvision
import torchvision.transforms as transforms
from skimage import io
from PIL import Image
from torchvision.utils import make_grid
from torchvision.io import read_image
from pathlib import Path

########################################################################################################################
# Only for case 40 mmm because the length label exists only for them now...!!!
########################################################################################################################

########################################################################################################################
# Extract features with convolutional layers
########################################################################################################################

class CNNPicture(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 12, 5)
        self.conv3 = nn.Conv2d(12, 24, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.linear1 = nn.Linear(24*32*116, 2320)
        self.linear2 = nn.Linear(2320, 580)
        self.linear3 = nn.Linear(580, 145)
        self.linear4 = nn.Linear(145, 1)

    def forward(self, img):
        out1 = self.pool(F.relu(self.conv1(img)))
        out2 = self.pool(F.relu(self.conv2(out1)))
        out3 = self.pool(F.relu(self.conv3(out2)))
        out4 = out3.view(-1, 24*32*116)
        out5 = F.relu(self.linear1(out4))
        out6 = F.relu(self.linear2(out5))
        out7 = F.relu(self.linear3(out6))
        out8 = self.linear4(out7)
        return out8

########################################################################################################################
# Hyper parameter
########################################################################################################################

h = 286 #height in pixel of the picture
l = 480 #length in pexel of the picture
lr_rate = 0.001
BATCH = 16
EPOCH = 20 # withc batch 16 and epoch 20 and a total of 388 pictures there are 68 left to test

########################################################################################################################
# Get the lables
########################################################################################################################

file_path = 'C:/Users/cuomo_g6z0s1c/PycharmProjects/BachelorThesis/LengthLabel/40mm.csv'

bad = [40, 194, 217, 230, 247, 257, 260, 270, 293, 294] # picture which are asumed to not work well for the ML
marks = [20, 21, 172, 173, 174, 175, 326, 327, 328, 329, 330, 429, 430, 431] # index of all marks
take_out = bad+marks

y_true = pd.read_csv(file_path, skiprows=1, delimiter=';', index_col=0, names=['Label']) # get the labeld data of the 40 mm frame length

for i in take_out:
    y_true = y_true.drop(index=i) # take out values that are not used

y_true = shuffle(y_true, random_state=42)
#print(y_true)
ind = np.array(y_true.index, dtype='str') # to later take out the picture with a file path, so I need strings
#print(ind[319])
########################################################################################################################
# Creat features
########################################################################################################################

transform = transforms.ToTensor()
# to cheack how it should look C:/Users/cuomo_g6z0s1c/PycharmProjects/BachelorThesis/Pictures/01_40mm/20220330-165552-image-0000000.BMP
file_path_start = 'C:/Users/cuomo_g6z0s1c/PycharmProjects/BachelorThesis/Pictures/01_40mm/20220330-165552-image-'
file_path_end = '.BMP'

def makeSeven(word):
    """creat a strin with the number of the ind array that can be used for the file path of the picture"""
    for i in range(100):
        zero = '0'
        word = zero + word
        if len(word)==7:
            return word

"""
# to test
img_test = io.imread('C:/Users/cuomo_g6z0s1c/PycharmProjects/BachelorThesis/Pictures/01_40mm/20220330-165552-image-0000000.BMP')
img_test = transform(img_test)
"""

criterion = nn.MSELoss()
model = CNNPicture()
optim = torch.optim.Adam(model.parameters(), lr=lr_rate)

for i in range(EPOCH):
    for j in range(BATCH):
        file_number = makeSeven(ind[i*BATCH+j])
        y = torch.tensor([y_true.iloc[i*BATCH+j]['Label']], dtype=torch.float32)
        #Many print functions to like about type and size as they are important for the torch functions to work
        #print('y_true', y.shape)
        #print('y_true.dtype', y.dtype)
        path = file_path_start+file_number+file_path_end

        img = io.imread(path)
        img = transform(img)

        y_hat = model.forward(img)
        y_hat = y_hat.view(1)
        #print('y_hat.shape', y_hat.shape)
        #print('y_hat.dtype', y_hat.dtype)
        loss = criterion(y_hat, y)
        #print(loss)

        loss.backward()
        optim.step()

        optim.zero_grad()

        if (i*BATCH+j)%30==0:
            print('Loss: ', loss.item())

# last picture is ind 319


y_test = np.array(y_true.iloc[320:])
y_test = y_test.reshape(68)

x_test = ind[320:]
y_pred = []

for i in x_test:
    file_number = makeSeven(i)
    path = file_path_start+file_number+file_path_end

    img = io.imread(path)
    img = transform(img)

    y = model.forward(img).detach()
    y = y.view(1)
    y = y.numpy()

    y_pred = np.append(y_pred, y)

x = np.arange(0, 68) # all pictures

plt.plot(x, y_test, 'b', marker='o', linestyle=None)
plt.plot(x, y_pred, 'ro', linestyle=None)
plt.show()
