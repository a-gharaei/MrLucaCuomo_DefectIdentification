import numpy as np
import pandas as pd
import function as fun
import matplotlib.pyplot as plt

# Chose case

case = 2

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

df_Laser = pd.read_csv('LC_001_2022-03-30 17.25.59.602_A0000.csv')

folder_time_stamp = 'C:/Users/cuomo_g6z0s1c/PycharmProjects/Laser_Data_Windows/Input/Time_stamps_picture'
file = ['/timestamps_20mm.csv', '/timestamps_30mm.csv', '/timestamps_40mm.csv']

#Timesteps

time_stamps_laser = df_Laser['Protocol Time Stamp']
first_stamp_laser = fun.get_time(time_stamps_laser.iloc[0])*10**(5)
second_stamp_laser = fun.get_time(time_stamps_laser.iloc[1])*10**(5)

TimeStepLaser = (second_stamp_laser-first_stamp_laser)*10**(-5)

time_stamps_pic = pd.read_csv(folder_time_stamp+file[case], index_col=0)
deviceClock = time_stamps_pic[' DeviceClock']

TimeStepCamera = (deviceClock.iloc[1]-deviceClock.iloc[0])*10**(-7) # 10**(-7) to get seconds

TimeStepLaser = TimeStepLaser*10**5
TimeStepCamera = TimeStepCamera*10**5

#print(TimeStepLaser)
#print(TimeStepCamera)

# Prepare Laser Data

index_marks = fun.get_index(df_Laser)

row, col = df_Laser.shape
start_index = index_marks[0] #+510
end_index = index_marks[-1]

SamplesBeginToStart = start_index
SamplesStartToEnd = row-start_index

TimeStepBeginToStart = np.arange(0,SamplesBeginToStart+1)*(-TimeStepLaser)
TimeStepBeginToStart = np.flip(TimeStepBeginToStart)
TimeStepBeginToStart = np.delete(TimeStepBeginToStart, obj=-1)

TimeStepStartToEnd = np.arange(0, SamplesStartToEnd)*(TimeStepLaser)

time_index = np.concatenate((TimeStepBeginToStart,TimeStepStartToEnd), axis=None)

df_Laser['time_index']=time_index

# Create a brand brand new data frame!

dataLaser = df_Laser[['time_index','Distance']]

LaserStepSize = 1/laser_rate*v_ring

L1 = int(startpoint_difference[case]//LaserStepSize+1)
L2 = int((frame[case]-startpoint_difference[case])//LaserStepSize+1)
window = int(L1 + L2)
#print('L1:', L1, 'L2:', L2)

new_df = pd.DataFrame(index=np.arange(first_picture[case], last_picture[case]) , columns= np.arange(0,window))
pictureNumber, windowSize = new_df.shape

for i in range(pictureNumber):

    if i == 0:
        step = 0
    else:
        step = i*TimeStepCamera/10
        step = int(step)*10 + 5


    ind = dataLaser[dataLaser['time_index'] == step].index
    ind = ind[0]

    data = dataLaser['Distance']
    new_df.iloc[i] = data[ind - L1:ind + L2]

    if i == pictureNumber - 1:
        end = endpoint_difference[case] / LaserStepSize + index_marks[-1]
        error = (end - (ind + L2)) * LaserStepSize
        print('Error last frame in Case', case, ': ', error)
