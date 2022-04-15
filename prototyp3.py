import numpy as np
import pandas as pd
import function as fun
import matplotlib.pyplot as plt

# Set up (order allways from smallest frame to biggest) camera, laser and olga

d = 1995 # diameter of the ring
f = 1.61 # rotational speed
v_ring = np.pi*d*f/60 # velocity in mm/s

frame = [20, 30, 40]
frame_rate = [43.006253, 15.001865, 11.000904]
laser_rate = 20000

first_picture = [99, 24, 20]
last_picture = [1706, 584, 431]
startpoint_difference = [0, 2.4264705882, 13.9890710383] # difference to the startpoint of the laser in mm
endpoint_difference = [20, 23.8235294118, 28.1318681319] # difference in length from one rotation

# Get data

df = pd.read_csv('LC_001_2022-03-30 17.25.59.602_A0000.csv')
index_marks = fun.get_index(df)

# Choose frame

case = 0

# Preparation for splitting the laser data

df = pd.read_csv('LC_001_2022-03-30 17.25.59.602_A0000.csv')
mark_index = fun.get_index(df)

time_frames = 1/frame_rate[case]
time_laser = 1/laser_rate
laser_step = time_frames/time_laser
step_size = time_laser*v_ring


window = frame[case]/step_size
window_rounded = int(window+1)

start = int(mark_index[0]-startpoint_difference[case]//step_size+1)
end = int(mark_index[-1]+endpoint_difference[case]//step_size+1)
laser_data_length = len(df['Distance'][start:end])
laser_data = df['Distance'][start:]

# split laser data

picture_number = last_picture[case]-first_picture[case]
new_df = pd.DataFrame(columns=np.arange(0, window_rounded), index=np.arange(first_picture[case], last_picture[case]))

real_midle = window/2
midle = int(real_midle)
backward_step = midle
forward_step = int(window_rounded//2 + 1)

#print('window', window, 'window_rounded', window_rounded, 'midle', midle, 'backward_step', backward_step, 'forward_step', forward_step)


for i in range(picture_number):
    start = midle-backward_step
    end = midle+forward_step

    if i==(picture_number-1):
        print('Error Case',case,':', (end-laser_data_length)*step_size)

    new_df.iloc[i]=laser_data[start:end]

    real_midle = real_midle + laser_step
    midle = int(real_midle)

folder_DataFrames = 'C:/Users/cuomo_g6z0s1c/PycharmProjects/Laser_Data_Windows/Input/DataFrames_Windows/'
file_DataFrames = ['windows_20mm.csv', 'windows_30mm.csv', 'windows_40mm.csv']

new_df.to_csv(folder_DataFrames+file_DataFrames[case])

# Test case 1 picture 157

"""
y = new_df.iloc[157]
y = np.array(y)
x = np.arange(0, len(y))

plt.figure(figsize=(15, 12))
plt.plot(x,y)
plt.xlabel(xlabel='Samples')
plt.ylabel(ylabel='Distance')
plt.show()
"""

