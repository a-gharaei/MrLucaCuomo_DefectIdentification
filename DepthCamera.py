import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from skimage import io
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from sklearn.utils import shuffle
from torch.optim import lr_scheduler

########################################################################################################################
# Case (actually it is for 20 mm so case=0 should be fix)
########################################################################################################################

case = 0

########################################################################################################################
# Hyperparameter
########################################################################################################################

lr_rate = 0.001
BatchSize = 20
Epoch = 13
Iteration = 1

########################################################################################################################
# Get Data
########################################################################################################################

folder = 'C:/Users/cuomo_g6z0s1c/PycharmProjects/BachelorThesis/LabelDepth'
file = ['/Depth20mm.csv', '/Depth30mm.csv', '/Depth40mm.csv']

depth_label = pd.read_csv(folder+file[case], index_col=0)

########################################################################################################################
# Creat an cnn
########################################################################################################################

# Not enough pictures for training

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 10, 5)
        self.pool = nn.MaxPool2d(4, 4)
        self.lin1 = nn.Linear(10*16*28, 1120)
        self.lin2 = nn.Linear(1120, 280)
        self.lin3 = nn.Linear(280, 70)
        self.lin4 = nn.Linear(70, 1)

    def forward(self, img):
        out = self.pool(F.relu(self.conv1(img)))
        out = self.pool(F.relu(self.conv2(out)))
        out = out.view(-1, 10*16*28)
        out = F.relu(self.lin1(out))
        out = F.relu(self.lin2(out))
        out = F.relu(self.lin3(out))
        out = self.lin4(out)

        return out

########################################################################################################################
# Picture index
########################################################################################################################

def makeSeven(word):
    """creat a strin with the number of the ind array that can be used for the file path of the picture"""
    for i in range(100):
        zero = '0'
        word = zero + word
        if len(word)==7:
            return word

path_start = 'C:/Users/cuomo_g6z0s1c/PycharmProjects/BachelorThesis/Pictures/03_20mm/20220330-170814-image-'
path_end = '.BMP'
df = shuffle(depth_label, random_state=42)

x_train = df.iloc[0:260]['Index']
x_test = df.iloc[260:]['Index'] # 23% der daten
y_train = df.iloc[0:260]['Depth']
y_test = df.iloc[260:]['Depth']

transform = transforms.ToTensor()

########################################################################################################################
# Training
########################################################################################################################

#model = CNN()

model = models.resnet18(pretrained=True)
"""
# for fine tuning, only change last layer
for param in model.parameters():
    param.requires_grad = False
"""
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)

step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.1)

for i in range(Iteration):
    for j in range(Epoch):
        for z in range(BatchSize):

            #test = j*BatchSize + z

            path_ind = makeSeven(str(x_train.iloc[j*BatchSize + z]))
            path = path_start+path_ind+path_end
            img = io.imread(path)
            img = transform(img)
            img = img.unsqueeze(0)

            y_pre = model.forward(img)
            y_pre = y_pre.view(1)

            y = torch.tensor([y_train.iloc[j*BatchSize + z]], dtype=torch.float32)

            loss = criterion(y_pre,y)

            loss.backward()
            optimizer.step()

            optimizer.zero_grad()

            if (j*BatchSize + z + 1)%260==0:
                with torch.no_grad():
                    number = i+1
                    number = str(number)
                    path = 'C:/Users/cuomo_g6z0s1c/PycharmProjects/BachelorThesis/Model/DepthCamera/modelResnet18'+number+'.pth'
                    torch.save(model, path)

                    row, = x_test.shape
                    y_hat = np.zeros(row)
                    y_test = np.array(y_test)
                    x = np.arange(row)

                    counter = 0

                    for i in x_test:
                        path_ind = makeSeven(str(i))
                        path = path_start + path_ind + path_end
                        img = io.imread(path)
                        img = transform(img)
                        img = img.unsqueeze(0)

                        y_pred = model.forward(img)
                        y_hat[counter] = y_pred
                        counter += 1

                    # print(len(x), len(y_hat), len(y_test))
                    errorR2 = r2_score(y_test, y_hat)
                    MSE = 1/len(y_test)*sum(y_test-y_hat)**2

                    print('R2 Score: ', errorR2, 'MSE: ', MSE)

                    fig, ax = plt.subplots()
                    lin1, = ax.plot(y_test, label='Y Test')
                    lin2, = ax.plot(y_hat, label='Y Predicted')

                    ax.legend(handles=[lin1], loc='upper right')
                    ax.legend(handles=[lin2], loc='upper right')

                    ax.legend(bbox_to_anchor=(1, 1),
                              bbox_transform=fig.transFigure
                              )

                    plt.show()


