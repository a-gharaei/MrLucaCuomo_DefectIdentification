import numpy as np
import pandas as pd
import function as fun
import matplotlib.pyplot as plt

# Chose case

case = 0

# Set up (order allways from smallest frame to biggest) camera, laser and olga

d = 1995 # diameter of the ring
f = 1.61 # rotational speed
v_ring = np.pi*d*f/60 # velocity in mm/s

frame = [20, 30, 40]
frame_rate = [43.006253, 15.001865, 11.000904]
laser_rate = 20000

first_picture = [99, 24, 20]
last_picture = [1706, 584, 431]
startpoint_difference = [0, 2.4264705882, 13.9890710383]
endpoint_difference = [20, 23.8235294118, 28.1318681319]

# Get data

df_Laser = pd.read_csv('LC_001_2022-03-30 17.25.59.602_A0000.csv')

# Prepare laser data

marks = fun.get_index(df_Laser)
df = df_Laser['Distance']
samples, = df.shape

start = marks[0]
BeginToStart = start
StartToEnd = samples-start

BeginToStart = np.arange(0, BeginToStart+1)*(-1)
BeginToStart = np.flip(BeginToStart)
BeginToStart = np.delete(BeginToStart, obj=-1)

StartToEnd = np.arange(0, StartToEnd)
ind = np.concatenate((BeginToStart, StartToEnd), axis=None)

df.index = ind

# Other preparations

LaserStepsBetween = (1/frame_rate[case])*laser_rate
LaserStep = 1/laser_rate*v_ring

L1 = int(startpoint_difference[case]//LaserStep+1)
L2 = int((frame[case]-startpoint_difference[case])//LaserStep+1)
window = L1 + L2
#print('L1: ', L1, 'L2: ', L2, 'window: ', window)

# Create new data frame

new_df = pd.DataFrame(index=np.arange(first_picture[case],last_picture[case]), columns= np.arange(0, window))

pictureNumber, windowSize = new_df.shape

for i in range(pictureNumber):
    new_middle = int(i*LaserStepsBetween)
    new_df.iloc[i]= df.loc[(new_middle-L1):(new_middle+L2-1)]


    if i == pictureNumber - 1:
        end = marks[-1]+endpoint_difference[case]/LaserStep
        error = (end-(new_middle+L2))*LaserStep
        print('Error in Last picture/window in Case', case, ': ', error)
