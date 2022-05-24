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
# Case
########################################################################################################################

case = 0

########################################################################################################################
# Hyperparameter
########################################################################################################################

lr_rate = 0.001
BatchSize = 20
Epoch = 13
Iteration = 8
transform = transforms.ToTensor()

########################################################################################################################
# Get Data
########################################################################################################################

folder = 'C:/Users/cuomo_g6z0s1c/PycharmProjects/BachelorThesis/NewLabelDepth'
file = ['/20mm', '/30mm', '/40mm']

df = pd.read_csv(folder+file[case], index_col=0)
df = shuffle(df, random_state=42)

df_train = df.iloc[:260]
df_test = df.iloc[260:]

########################################################################################################################
# And foooorwaaaard!!!;)
########################################################################################################################

model = models.resnet18(pretrained=True)

# training full model or only last fully connected layer
for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)

#step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.1)

for i in range(Iteration):

    for j in range(Epoch):

        y_pre = torch.tensor(np.zeros(BatchSize))
        y_true = torch.tensor(np.zeros(BatchSize))

        for z in range(BatchSize):
            path = df_train.iloc[j*BatchSize + z]['Index']
            img = io.imread(path)
            img = transform(img)
            img = img.unsqueeze(0)

            y_pre[z] = model(img)
            y_true[z] = df_train.iloc[j*BatchSize + z]['Depth']

        loss = criterion(y_pre, y_true)
        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

        if (j * BatchSize + z + 1) % 260 == 0:
            with torch.no_grad():

                y_pre = torch.tensor(np.zeros(len(df_test)))
                y_true = torch.tensor(np.zeros(len(df_test)))

                for i in range(len(df_test)):

                    path = df_test.iloc[i]['Index']
                    img = io.imread(path)
                    img = transform(img)
                    img = img.unsqueeze(0)

                    y_pre[i] = model(img)
                    y_true[i] = df_test.iloc[i]['Depth']


                R2 = r2_score(y_true=y_true, y_pred=y_pre)
                MSE = criterion(y_pre, y_true)

                y_pre = np.array(y_pre)
                y_true = np.array(y_true)

                print('R2 Score: ', R2, 'MSE: ', MSE.item())

                fig, ax = plt.subplots()
                lin1, = ax.plot(y_true, label='Y_True')
                lin2, = ax.plot(y_pre, label='Y_Predicted')

                ax.legend(handles=[lin1], loc='upper right')
                ax.legend(handles=[lin2], loc='upper right')

                ax.legend(bbox_to_anchor=(1, 1),
                          bbox_transform=fig.transFigure
                          )

                plt.show()



