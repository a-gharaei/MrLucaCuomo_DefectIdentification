import numpy as np
import pandas as pd

def get_index(df):
    dist = df.Distance
    marks = dist < 4
    index = 0
    liste = []

    for i in marks:
        index += 1
        if i:
            liste.append(index)

    first_index_marks = [liste[0]]

    for i in range(len(liste) - 1):
        if liste[i] + 10000 < liste[i + 1]:
            first_index_marks.append(liste[i + 1])

    return first_index_marks

def step_size(df, u_ring):
    mark_index = get_index(df)

    distance = df['Distance']
    laser_full_rotation = distance[mark_index[0]:mark_index[-1]]
    size = u_ring / len(laser_full_rotation)

    return size

def get_overlap(u, frame, pic_number, extra):
    overlap = []
    for i in range(len(frame)):
        overlap.append(1-(u+extra[i]-frame[i])/(frame[i]*(pic_number[i]-1)))
    return overlap

def get_appropriate(data_length, start_window, picture_number, range_number):

    for i in range(range_number):
        if (data_length-(start_window+i))%(picture_number-1)==0:
            window = start_window+i
            laser_step = int((data_length-window)/(picture_number-1))
            return window, laser_step



def prototyp(data_length, full_data,window, laser_step):

    rows = int((data_length - window) / laser_step + 1)
    df_window = pd.DataFrame(columns=np.arange(0, window), index=np.arange(0, rows))

    for i in range(rows):
        df_window.iloc[i] = full_data[i * laser_step:window + i * laser_step]

    return df_window
