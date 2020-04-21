from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import csv
import netCDF4 as nc
from datetime import datetime, timedelta
from netCDF4 import num2date, date2num
from mpl_toolkits.basemap import Basemap,cm
from paper.function import *


file = open('F://Jupyter/Data/1970-2018bst.txt',mode = 'r')
TpTrack = pd.read_csv('F://Jupyter/Data/1970_2018TyphoonTrack.csv') # 台风路径数据来源
TpDate = pd.read_csv('F://Jupyter/Data/1970-2018date.csv')         # 台风序列时间

#纬度
lat = TpTrack['Lat']
#经度
lon = TpTrack['Lon']
#台风ID
ID = TpDate['ID']
#每个台风持续时间（每6h一个记录值）
day = TpDate['day']

# 最大最小化处理
lat_nor = (lat-lat.min())/(lat.max()-lat.min())
lon_nor = (lon-lon.min())/(lon.max()-lon.min())


dataset = [lat_nor,lon_nor]
# 将1267个台风路径数据存在字典中，ID作为键,值为（lat,lon）路径
TPT = Datatodir(lat,lon,day,ID)

    
#选取2018年之前作为训练集 2018年数据作为测试集
p1 = {key: value for key, value in TPT.items() if int(key) < 1801 or int(key)>= 1900}
p2 = {key: value for key, value in TPT.items() if int(key) >= 1801 and int(key) < 1900}

trainingset = list(p1.values())
testset = list(p2.values())


# 观察一下 这些台风路径外观
# 训练集
#PrintTrackFig(trainingset,'1970-2017',lat,lon)
#测试集
#PrintTrackFig(testset,'2018',lat,lon)

#归一化数据作图
nTPT = Datatodir(lat_nor,lon_nor,day,ID)
np1 = {key: value for key, value in nTPT.items() if int(key) < 1801 or int(key)>= 1900}
np2 = {key: value for key, value in nTPT.items() if int(key) >= 1801 and int(key) < 1900}
nor_traindata = np.array(list(np1.values()))
nor_testdata = np.array(list(np2.values()))

def printfig(data):
    for i in data:
        X= []
        Y = []
        for j in i:
            X.append(j[1])
            Y.append(j[0])
        plt.plot(X,Y,'--')
        
    plt.show()

# printfig(nor_traindata)
# printfig(nor_testdata)


# 数据分为 训练集（训练+验证） 测试集（测试+验证）并进行0值填充使数据长度一致
train_length = len(nor_traindata)
data = list(nTPT)
BUFFER_SIZE = 10000

train_X ,train_Y = Divdata(nor_traindata,lenMax(nor_traindata),4)
test_X,test_Y = Divdata(nor_testdata,lenMax(nor_traindata), 4)
train_data = tf.data.Dataset.from_tensor_slices((train_X, train_Y))
train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(train_length).repeat()

train_Y1 = train_Y.reshape(len(train_Y),8)
test_Y1 = test_Y.reshape(len(test_Y),8)
val_data = tf.data.Dataset.from_tensor_slices((test_X,test_Y))
val_data = val_data.batch(train_length).repeat()

from keras.layers import Masking, Embedding
from keras.layers import LSTM

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Masking(mask_value = 0.))
model.add(tf.keras.layers.LSTM(108,return_sequences=True,input_shape = train_X.shape[-2:]))

model.add(tf.keras.layers.LSTM(108,return_sequences=True,input_shape = train_X.shape[-2:]))

model.add(tf.keras.layers.LSTM(54,activation='relu'))
model.add(tf.keras.layers.Dense(8))

model.compile(optimizer='adam', loss='mae')

loss = model.fit(train_X,train_Y1, epochs=2)

PrintLossFig(loss,'trainloss')

# pre = []
# for x, y in val_data.take(2):
#     pre.append(model.predict(x))
# pre = np.array(pre)

pre = model.predict(test_X)
pre = pre.reshape(len(pre),4,2)
# def Divdata2(data,val_time_step):   #分割数据 val_time_step 验证集长度
#     X = []
#     Y = []
    
#     for i in data:
#         X.append(np.array(i[:(len(i)-val_time_step)]))
#         Y.append(np.array(i[len(i)-val_time_step:]))
    
#     X,Y = np.array(X),np.array(Y)
    
#     return X,Y

# Xx,Yy = Divdata2(nor_testdata,1)

fig,ax = plt.subplots(5,6,figsize = (24,20))
PT = []
for i in range(len(nor_testdata)):
    x = 0
    y = 0
    
    X = []
    Y = []
    for j in Xx[i]:
        X.append(j[0])
        Y.append(j[1])
    a = pre[i][:,0]
    b = pre[i][:,1]
    c = test_Y[i][:,0]
    d = test_Y[i][:,1]
    PT.append([[X,Y],[a,b],[c,d]])

i = 0
for x in range(5):
    for y in range(6): 
        if i < 29:
            ax[x,y].plot(PT[i][0][0],PT[i][0][1])
            ax[x,y].plot(PT[i][1][0],PT[i][1][1],marker = 'x')
            ax[x,y].plot(PT[i][2][0],PT[i][2][1],marker = 'o')
        i += 1
        
    
    
# fig,ax = plt.subplots(5,6,figsize = (24,20))
# PT = []
# for i in range(len(nor_testdata)):
#     x = 0
#     y = 0
    
#     X = []
#     Y = []
#     for j in Xx[i]:
#         X.append(j[0])
#         Y.append(j[1])
#     a = pre[i,0]
#     b = pre[i,1]
#     c = test_Y[i][:,0]
#     d = test_Y[i][:,1]
#     PT.append([[X,Y],[a,b],[c,d]])

# i = 0
# for x in range(5):
#     for y in range(6): 
#         if i < 29:
#             ax[x,y].plot(PT[i][0][0],PT[i][0][1])
#             ax[x,y].plot(PT[i][1][0],PT[i][1][1],marker = 'x')
#             ax[x,y].plot(PT[i][2][0],PT[i][2][1],marker = 'o')
#         i += 1
# #plt.plot(b,a)





