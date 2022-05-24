import pandas as pd
import numpy as np
import function as fun
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score

########################################################################################################################
#Chose Case
########################################################################################################################

case = 2

########################################################################################################################
#Get Lables
########################################################################################################################

file_path = ['C:/Users/cuomo_g6z0s1c/PycharmProjects/BachelorThesis/Label/Label_20mm.csv',
             'C:/Users/cuomo_g6z0s1c/PycharmProjects/BachelorThesis/Label/Label_30mm.csv',
             'C:/Users/cuomo_g6z0s1c/PycharmProjects/BachelorThesis/Label/Label_40mm.csv']

df_lable = pd.read_csv(file_path[case], delimiter=';', index_col=0)

#picture numbers of squats and create df with the picture in the sector 1

start_squat = [103, 26, 22]
end_squat = [691, 230, 171]

df_squat = df_lable.loc[start_squat[case]:end_squat[case]]

#picture numbers of corrosion spots and create df with the picture in the sector 2

start_corrosion = [706, 236, 176]
end_corrosion = [1293, 439, 325]

df_corrosion = df_lable.loc[start_corrosion[case]:end_corrosion[case]]

#the same with no defect in the section 3

start_noDefect = [1312, 447, 331]
end_noDefect = [1698, 581, 428]

df_noDefect = df_lable.loc[start_noDefect[case]:end_noDefect[case]]

df = pd.concat([df_squat,df_corrosion, df_noDefect], axis=0)

########################################################################################################################
#Get Laser windows
########################################################################################################################

file_path =['C:/Users/cuomo_g6z0s1c/PycharmProjects/BachelorThesis/LaserWindow/LaserWindow20mm.csv',
            'C:/Users/cuomo_g6z0s1c/PycharmProjects/BachelorThesis/LaserWindow/LaserWindow30mm.csv',
            'C:/Users/cuomo_g6z0s1c/PycharmProjects/BachelorThesis/LaserWindow/LaserWindow40mm.csv']

df_laser = pd.read_csv(file_path[case], index_col=0) #get the laser windows

df_laser_squat = df_laser.loc[start_squat[case]:end_squat[case]] #laser windows from squats
df_laser_corrosion = df_laser.loc[start_corrosion[case]:end_corrosion[case]] #laser windows from corrosion spots
df_laser_noDefect = df_laser.loc[start_noDefect[case]:end_noDefect[case]] #laser windows from no defects

########################################################################################################################
#Training algorithm seperatly
########################################################################################################################
# for squats
"""
ind = np.arange(start_squat[case], end_squat[case]+1)
ad_feature = fun.expand(df_laser_squat, ind)
df_laser_squat = pd.concat([df_laser_squat, ad_feature], axis=1)

ind = np.arange(start_noDefect[case], end_noDefect[case]+1)
ad_feature1 = fun.expand(df_laser_noDefect, ind)
df_laser_noDefect = pd.concat([df_laser_noDefect, ad_feature1], axis=1)
"""
feature_squat = pd.concat([df_laser_squat, df_laser_noDefect], axis=0)
label_squat = pd.concat([df_squat, df_noDefect], axis=0)
# print(pd.unique(label_squat['Label'])) # control if I have the right labels and only those

X_train, X_test, y_train, y_test = train_test_split(feature_squat, label_squat, test_size=0.3, random_state=42)

y_train = y_train['Label']
y_test = y_test['Label']

# clf = LogisticRegression(class_weight='balanced', max_iter=1000, solver='saga').fit(X_train, y1) # does not converge
model = SVC(class_weight='balanced').fit(X_train, y_train) #score 0.9411764705882353
score_squat = model.score(X_test, y_test)
print('Score SVC squat: ', score_squat)

# for corrosion
"""
ind = np.arange(start_corrosion[case], end_corrosion[case]+1)
ad_feature = fun.expand(df_laser_corrosion, ind)
df_laser_corrosion = pd.concat([df_laser_corrosion, ad_feature], axis=1)

ind = np.arange(start_noDefect[case], end_noDefect[case]+1)
ad_feature1 = fun.expand(df_laser_noDefect, ind)
df_laser_noDefect = pd.concat([df_laser_noDefect, ad_feature1], axis=1)
"""
featur_corrosion = pd.concat([df_laser_corrosion, df_laser_noDefect], axis=0)
label_corrosion = pd.concat([df_corrosion, df_noDefect], axis=0)

# print(pd.unique(label_corrosion['Label'])) # control if I only have the right labels

X_train, X_test, y_train, y_test = train_test_split(featur_corrosion, label_corrosion, test_size=0.3, random_state=42)

y_train = y_train['Label']
y_test = y_test['Label']

#model = LogisticRegression(class_weight='balanced', max_iter=100, solver='saga').fit(X_train, y_train) #error that it does not converge but I get a result 0.5588235294117647 and with less iterations 0.45588235294117646
model = SVC(class_weight='balanced').fit(X_train, y_train) # score 0.6176470588235294
score_corrosion = model.score(X_test, y_test)
print('Score SVC corrosion: ', score_corrosion) #without expand 0.6176470588235294, with expand 0.6029411764705882

########################################################################################################################
#Training algorithm on one data set
########################################################################################################################
"""
ind = np.arange(start_squat[case], end_squat[case]+1)
ad_feature = fun.expand(df_laser_squat, ind)
df_laser_squat = pd.concat([df_laser_squat, ad_feature], axis=1)

ind = np.arange(start_corrosion[case], end_corrosion[case]+1)
ad_feature = fun.expand(df_laser_corrosion, ind)
df_laser_corrosion = pd.concat([df_laser_corrosion, ad_feature], axis=1)

ind = np.arange(start_noDefect[case], end_noDefect[case]+1)
ad_feature1 = fun.expand(df_laser_noDefect, ind)
df_laser_noDefect = pd.concat([df_laser_noDefect, ad_feature1], axis=1)
"""
feature_all = pd.concat([df_laser_squat, df_laser_corrosion, df_laser_noDefect], axis=0)
label_all = df

X_train, X_test, y_train, y_test = train_test_split(feature_all, label_all, test_size=0.2, random_state=42)

y_train = y_train['Label']
y_test = y_test['Label']

model = SVC(class_weight='balanced').fit(X_train, y_train) # score 0.7247706422018348 with feature expand 0.7339449541284404
y_pred = model.predict(X_test)
print('Score SVC all: ', model.score(X_test, y_test))
print('F1 Score SVC all: ', f1_score(y_pred, y_test, average=None, labels=[1, 2, 3]))
print('Mean of score of seperatly calculated models: ', (score_squat+score_corrosion)/2)
