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

#数据处理 1267个台风路径数据存在字典中，ID作为键,输出分别为lat,lon路径

def DataProg(lat,lon,day,ID):
    index = 0
    Lat = {}
    Lon = {}
    for i in range(len(day)):
        a = []
        o = []
        for j in range(day[i]):
            
            a.append(lat[index])
            o.append(lon[index])
            index+=1
        
        Lat.update({str(ID[i]):a})
        Lon.update({str(ID[i]):o})
        
    return Lat,Lon

Lat,Lon = DataProg(lat_nor,lon_nor,day,ID)


#选取2018年之前作为训练集 2018年数据作为测试集
p1 = {key: value for key, value in Lat.items() if int(key) < 1801 or int(key)>= 1900}
p2 = {key: value for key, value in Lat.items() if int(key) >= 1801 and int(key) < 1900}

train_Lat =  Dirtolist(p1.values())
test_Lat = Dirtolist(p2.values())
trainLat_x,trainLat_y = getdata(train_Lat,4)
testLat_x,testLat_y = getdata(test_Lat,4)

p3 = {key: value for key, value in Lon.items() if int(key) < 1801 or int(key)>= 1900}
p4 = {key: value for key, value in Lon.items() if int(key) >= 1801 and int(key) < 1900}


train_Lon =  Dirtolist(p3.values())
test_Lon = Dirtolist(p4.values())
trainLon_x,trainLon_y = getdata(train_Lon,4)
testLon_x,testLon_y = getdata(test_Lon,4)



trainLat_X,lat_len = paddatasingle(trainLat_x, 1)
trainLon_X,lon_len = paddatasingle(trainLon_x, 1)
trainLat_Y = paddatasingle2(trainLat_y, lat_len,1)
trainLon_Y = paddatasingle2(trainLon_y, lon_len,1)

testLat_X = paddatasingle2(testLat_x,lat_len, 1)
testLon_X = paddatasingle2(testLon_x,lon_len, 1)
testLat_Y = paddatasingle2(testLat_y, lat_len,1)
testLon_Y = paddatasingle2(testLon_y, lon_len,1)







from keras.layers import Masking, Embedding
from keras.layers import LSTM




model = tf.keras.models.Sequential([
    tf.keras.layers.Masking(mask_value=0.,
                                  input_shape=trainLat_X.shape[-2:]),
    tf.keras.layers.LSTM(108, input_shape=trainLat_X.shape[-2:],activation='relu'),
    tf.keras.layers.Dense(4)
])
model.compile(loss='mae', optimizer ='adam')
model.fit(trainLat_X,trainLat_Y[:,:4,:],epochs= 100)


preY = model.predict(testLat_X)

# model2 = model
# model2.compile(loss='mae', optimizer ='adam')
# model2.fit(trainLon_X,trainLon_Y[:,:4,:],epochs= 10)
# preX = model2.predict(testLon_X)

fig,ax = plt.subplots(5,6,figsize = (24,20))
PT = []
for i in range(len(test_Lon)):
    
    X = test_Lon[i]
    Y = test_Lat[i]
    PT.append([Y,testLat_y[i],preY[i]])

i = 0
for x in range(5):
    for y in range(6): 
        if i < 29:
            ax[x,y].plot(range(len(PT[i][0])),PT[i][0])
            ax[x,y].plot(range(len(PT[i][0]),len(PT[i][0])+4),PT[i][2])
        i += 1

