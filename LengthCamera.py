import pandas as pd
from function import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import DataLoader
from skimage import io
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from sklearn.utils import shuffle
from torch.optim import lr_scheduler

########################################################################################################################
# Get Length Data
########################################################################################################################

df = pd.read_csv('C:/Users/cuomo_g6z0s1c/PycharmProjects/BachelorThesis/NewLengthLabel/40mm', index_col=0)

#df = df.iloc[:141] # for squats
df = df.iloc[158:] # for corrosion

df = df[df['Length']>0] # I want only pictures with a defect on it
#print(len(df))




########################################################################################################################
# Hyperparameter
########################################################################################################################

BatchSizeTrain = 16
BatchSizeTest = 25
Epoch = 33
lr_rate = 0.001


########################################################################################################################
# Dataloading, split in test and train...
########################################################################################################################

mean = df['Length'].mean()
std = df['Length'].std()
split = int(0.8*len(df))

transform = transforms.ToTensor()


train = Image(df.iloc[:split], mean=mean, std=std, transform=transform)
test = Image(df.iloc[split:], mean=mean, std=std, transform=transform)

TrainDataloader = DataLoader(train, batch_size=BatchSizeTrain, shuffle=True, drop_last=True)
TestDataloader = DataLoader(test, batch_size=BatchSizeTest, shuffle=True, drop_last=True)

########################################################################################################################
# Forward
########################################################################################################################

model = models.resnet50(pretrained=True)

# training full model or only last fully connected layer
for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)

end = int(0.8*len(df)/BatchSizeTrain)

error = pd.DataFrame(index=np.arange(Epoch), columns=['R2', 'MSE'])
number = 0

step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=end, gamma=0.1)

for j in range(Epoch):
    for i,(featurs, labels) in enumerate(TrainDataloader):
        y_pre = model(featurs)
        y_true = torch.reshape(labels, (BatchSizeTrain,1))

        loss = criterion(y_pre, y_true)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (i+1)==end:
            with torch.no_grad():

                for z, (featurs, label) in enumerate(TestDataloader):
                    y_pre = model(featurs)
                    y_true = torch.reshape(label, (BatchSizeTest, 1))


                R2 = r2_score(y_true, y_pre)
                MSE = np.square(criterion(y_pre, y_true).item())
                error.iloc[number]['R2'] = R2
                error.iloc[number]['MSE'] = MSE.item()

                number += 1

                print('Number: ', number, 'R2: ', R2, 'MSE: ', MSE.item())

                fig, ax = plt.subplots()

                ax.set_title('Prediction Squats')
                ax.set_xlabel('Samples')
                ax.set_ylabel('Normalised Length')

                lin1, = ax.plot(y_true, label='Y True')
                lin2, = ax.plot(y_pre, label='Y Predicted')

                ax.legend(handles=[lin1], loc='upper right')
                ax.legend(handles=[lin2], loc='upper right')

                ax.legend(bbox_to_anchor=(1, 1),
                          bbox_transform=fig.transFigure
                          )

                plt.show()

MSEmin = error['MSE'].min()
R2max = error['R2'].max()
print('R2: ', R2max, 'MSE: ', MSEmin)
error.to_csv('error.csv')



