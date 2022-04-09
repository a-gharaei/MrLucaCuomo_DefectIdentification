import numpy as np
import pandas as pd
import function as fun
import matplotlib.pyplot as plt
import sklearn
import matplotlib

# start point

u_ring = np.pi*1995
frame_length = [20, 30, 40]
pic_num = [1607, 561, 411] # 20mm (99 to 1706), 30mm (24 to 585), 40mm (20 to 431)
endpoint_difference = [0, 2.4264705882, 2.1857923497] # difference in length from one rotation
startpoint_difference = [0, 2.4264705882, 13.9890710383] # difference to the startpoint of the laser

# which frame length
i = 1


# prepare data

df = pd.read_csv('LC_001_2022-03-30 17.25.59.602_A0000.csv')
mark_index = fun.get_index(df)

step_size = fun.step_size(df, u_ring)
start = int(endpoint_difference[i]//step_size+1)

full_rotation = df['Distance']
full_rotation = full_rotation[(mark_index[0]-start):(mark_index[-1]-start)]
data_length = len(full_rotation)


star_window = int(frame_length[i]//step_size)
window, laser_step = fun.get_appropriate(data_length,star_window,pic_num[i], 10000)
#print('window', window, 'laser_step', laser_step)
#print('window', window*step_size, 'laser_step', laser_step*step_size)

#prototyp2

df_window = fun.prototyp(data_length, full_rotation,window, laser_step)


# test with defect, 30mm picture 157

y = df_window.iloc[161]
y = np.array(y)
x = np.arange(0, len(y))

plt.figure(figsize=(15, 12))
plt.plot(x,y)
plt.xlabel(xlabel='Samples')
plt.ylabel(ylabel='Distance')
plt.show()