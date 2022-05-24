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
from matplotlib import pyplot as plt
import cv2


########################################################################################################################
#Chose Case
########################################################################################################################

case = 2

########################################################################################################################
#Analyse Lables
########################################################################################################################


file_path = ['C:/Users/cuomo_g6z0s1c/PycharmProjects/BachelorThesis/Label/Label_20mm.csv',
             'C:/Users/cuomo_g6z0s1c/PycharmProjects/BachelorThesis/Label/Label_30mm.csv',
             'C:/Users/cuomo_g6z0s1c/PycharmProjects/BachelorThesis/Label/Label_40mm.csv']

df_lable = pd.read_csv(file_path[case], delimiter=';', index_col=0)

#picture numbers of squats and create df with the picture in the sector 1

start_squat = [103, 26, 22]
end_squat = [691, 230, 171]

df_squat = df_lable.loc[start_squat[case]:end_squat[case]]
print('Number of pictures in case',case, 'for the squat section', df_squat.shape[0])

#picture numbers of corrosion spots and create df with the picture in the sector 2

star_corrosion = [706, 236, 176]
end_corrosion = [1293, 439, 325]

df_corrosion = df_lable.loc[star_corrosion[case]:end_corrosion[case]]
print('Number of pictures in case',case, 'for the corrosion section', df_corrosion.shape[0])

#the same with no defect in the section 3

star_noDefect = [1312, 447, 331]
end_noDefect = [1698, 581, 428]

df_noDefect = df_lable.loc[star_noDefect[case]:end_noDefect[case]]
print('Number of pictures in case',case, 'for the no defect section', df_noDefect.shape[0])

df = pd.concat([df_squat,df_corrosion, df_noDefect], axis=0)

y1 = len(df[df['Label']==1]) #Number of squats pictures
y2 = len(df[df['Label']==2]) #Number of corrosion spots pictures
y3 = len(df[df['Label']==3]) #Number of no defect pictures
print('Case: ', case, 'squats: ', y1, 'corrosion spots: ', y2, 'no defects: ', y3)

x = np.array([1,2, 3])

# creat a plot to look for balance in the lables

# Make a random dataset:

y = np.array([y1, y2, y3])
x = np.array([1,2, 3])

bars = ('Squats', 'Corrosion Spots', 'No Defects')
plt.bar(x, y)
plt.xticks(x, bars)
plt.show()

########################################################################################################################
# pictures
########################################################################################################################

"""
Torchvision can not handle BMP...

img= Image.open('C:/Users/cuomo_g6z0s1c/PycharmProjects/BachelorThesis/Pictures/03_20mm/20220330-170814-image-0000000.BMP')
img.save('testPicture.jpeg')
transform = transforms.ToTensor()
img_test = io.imread('C:/Users/cuomo_g6z0s1c/PycharmProjects/BachelorThesis/Pictures/03_20mm/20220330-170814-image-0000000.BMP', pilmode='RGB')
print(img_test.shape)
img_test=transform(img_test)
print('img_test as torch tensor', img_test.shape)

img1 = torchvision.io.read_image('testPicture.jpeg')
print('Image shape in colour: ', img1.shape) # get size of image for CNN in color

img2 = torchvision.io.read_image('testPicture.jpeg')
gray = transforms.Grayscale(1)
img2 = gray.forward(img2)
print('After gray scaling', img2.shape)


imgS =  cv2.imread('C:/Users/cuomo_g6z0s1c/PycharmProjects/BachelorThesis/Pictures/02_30mm/20220330-170208-image-0000157.BMP')
imgC = cv2.imread('C:/Users/cuomo_g6z0s1c/PycharmProjects/BachelorThesis/Pictures/02_30mm/20220330-170208-image-0000376.BMP')

imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2GRAY)
imgC = cv2.cvtColor(imgC, cv2.COLOR_BGR2GRAY)

imgS = cv2.threshold(imgS,127,255,cv2.THRESH_BINARY)
imgC = cv2.threshold(imgC,127,255,cv2.THRESH_BINARY)

print(imgS)


#plt.imshow(imgS,'gray',vmin=0,vmax=255)

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Grayscale(num_output_channels=1),
                                transforms.GaussianBlur(kernel_size=15, sigma=50),
                                nn.Threshold(0.3, 0),
                                transforms.ToPILImage()])

transGray = transforms.Grayscale(num_output_channels=1)
transPIL = transforms.ToPILImage()
thresh = nn.Threshold(0.3, 0)
transTensor = transforms.ToTensor()

imgS = io.imread('C:/Users/cuomo_g6z0s1c/PycharmProjects/BachelorThesis/Pictures/02_30mm/20220330-170208-image-0000157.BMP')
imgC = io.imread('C:/Users/cuomo_g6z0s1c/PycharmProjects/BachelorThesis/Pictures/02_30mm/20220330-170208-image-0000376.BMP')


imgS = transform(imgS)
imgC = transform(imgC)


plt.imshow(imgC)
plt.show()

df = pd.read_csv('C:/Users/cuomo_g6z0s1c/PycharmProjects/BachelorThesis/LaserWindow/LaserWindow30mm.csv', index_col=0)
head = df.head()
head = np.array(head)

trans = transforms.ToTensor()
head = trans(head)
test = head
std, mean = torch.std_mean(head)
std, mean = std.item(), mean.item()

head = (head-mean)/(std + 1e-6)
print(head)
norm = transforms.Normalize(mean=mean, std=std)
test = norm(test)
print(test)


y1 = np.array([1, 2, 3])
y2 = np.array([0.5, 1.5, 2.5])
print(1/len(y1)*sum(y1-y2))
print(1/len(y1))
"""