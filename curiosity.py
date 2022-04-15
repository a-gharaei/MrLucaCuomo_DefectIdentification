import numpy as np
import pandas as pd
import function as fun
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Choose Case

case = 0

# Get Data

folder_Label = 'C:/Users/cuomo_g6z0s1c/PycharmProjects/Laser_Data_Windows/Input/Label/'
folder_DataFrames = 'C:/Users/cuomo_g6z0s1c/PycharmProjects/Laser_Data_Windows/Input/DataFrames_Windows/'

file_DataFrames = ['windows_20mm.csv', 'windows_30mm.csv', 'windows_40mm.csv']
file_Label = ['Label_20mm.csv', 'Label_30mm.csv', 'Label_40mm.csv']

df_train = pd.read_csv(folder_DataFrames + file_DataFrames[case], index_col=[0])
df_label = pd.read_csv(folder_Label + file_Label[case], index_col=[0], delimiter=';', names=['Label'], skiprows=1)

# Craete DataFrames to train the model

corrosion_start = [706, 236, 176]
corrosion_end = [1293, 439, 325]
defectfree_start = [1312, 447, 331]
defectfree_end = [1698, 581, 428]

x1 = df_train.loc[corrosion_start[case]:corrosion_end[case]]
x2 = df_train.loc[defectfree_start[case]:defectfree_end[case]]

x = pd.concat([x1, x2])
x_scaled = fun.normalize(x)

y1 = df_label.loc[corrosion_start[case]:corrosion_end[case]]
y2 = df_label.loc[defectfree_start[case]:defectfree_end[case]]

y = pd.concat([y1,y2], ignore_index=True)

X_train, X_test, Y_train, Y_test = train_test_split(x_scaled,y, test_size=0.7, shuffle=True)

Y_train = np.ravel(Y_train, order='C')
Y_test = np.ravel(Y_test, order='C')

# Train model and look what happens....

model = SVC().fit(X_train, Y_train)
print(model.score(X_test, Y_test))


