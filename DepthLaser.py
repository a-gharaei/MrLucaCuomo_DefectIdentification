import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

########################################################################################################################
# Case
########################################################################################################################

case = 0

########################################################################################################################
# Hyperparameter
########################################################################################################################

block_size = [20, 16, 26]
iteraton = [119, 223, 183]

########################################################################################################################
# get data frame
########################################################################################################################

# used to get the mean in the no defect area, so that the depth of the squats can be calculated
df_raw = pd.read_csv('C:/Users/cuomo_g6z0s1c/PycharmProjects/BachelorThesis/LaserDataRaw/LC_001_2022-03-30 17.25.59.602_A0000.csv')
df_raw = df_raw['Distance']

file_path_begin = 'C:/Users/cuomo_g6z0s1c/PycharmProjects/BachelorThesis/LaserWindow'
file_path_end = ['/LaserWindow20mm.csv', '/LaserWindow30mm.csv', '/LaserWindow40mm.csv']
df_window = pd.read_csv(file_path_begin+file_path_end[case], index_col=0)

file_path_begin = 'C:/Users/cuomo_g6z0s1c/PycharmProjects/BachelorThesis/Label'
file_path_end = ['/Label_20mm.csv', '/Label_30mm.csv', '/Label_40mm.csv']
df_label = pd.read_csv(file_path_begin+file_path_end[case], index_col=0, delimiter=';', skiprows=1, names=['Label'])

########################################################################################################################
# get mean of no defect section
########################################################################################################################

"""
#Get part to calculate the mean
no_defect_section = df_raw.iloc[625000:808000]
y = np.array(no_defect_section)
x = np.array(no_defect_section.index)

plt.plot(x,y)
#plt.axis('off')
plt.show()

"""
mean = df_raw.iloc[650000:800000].mean()
std = df_raw.iloc[650000:800000].std()
#print('mean', mean)
#print('std', std)

########################################################################################################################
# get squats windows and get depth of it
########################################################################################################################

df_squats = pd.concat([df_label, df_window], axis=1)
df_squats = df_squats[df_squats['Label']==1]

ind = np.array(df_squats.index)
test = 0

df_squats = df_squats.drop(['Label'], axis=1)
row, col = df_squats.shape

"""
# get batch size which fits

ind, col = df_squats.shape

div = 0
print(col)
for i in range(col):
    div += 1
    if col%div==0:
        print(div)
"""

depth = np.zeros(row)

for i in range(row):

    test += 3
    if test==2:
        break

    block = pd.DataFrame(np.zeros(block_size[case]))
    means = pd.DataFrame(np.zeros(iteraton[case]))
    #print(block)
    #print(means)

    for j in range(iteraton[case]):
        start = j*block_size[case]
        end = j*block_size[case] + block_size[case]
        block = df_squats.iloc[i][start:end]

        means.iloc[j] = block.mean()

    depth[i] = means.max() - mean

depth = pd.DataFrame(depth, columns=['Depth'])
depth = pd.concat([pd.DataFrame(np.array(df_squats.index), columns= ['Index']), depth], axis=1)
#print(df_squats.index)
#print(depth)

"""
# look into the results
control_point = 0
print(depth.iloc[control_point])
row, col = df_squats.shape

y = np.array(df_squats.iloc[control_point])
x = np.arange(0, col)

plt.plot(x, y)
plt.show()
"""

# produce files with the depth for the training of the CNN
"""
folder = 'C:/Users/cuomo_g6z0s1c/PycharmProjects/BachelorThesis/LabelDepth'
file = ['/Depth20mm.csv', '/Depth30mm.csv', '/Depth40mm.csv']

depth.to_csv(folder+file[case])
"""