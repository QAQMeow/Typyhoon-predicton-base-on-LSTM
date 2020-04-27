from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os
import numpy as np
import pandas as pd
import time
import csv
import netCDF4 as nc
from datetime import datetime, timedelta
from netCDF4 import num2date, date2num
from paper.function import *
import random

path = "F://Jupyter/Data/CMABSTdata2"
files = os.listdir(path)
txts = []
t = []

#读取所有文件
for file in files:
    position = path+'\\'+file
    # print(postion)
    with open(position, "r",encoding='utf-8') as f:    #打开文件
        data = f.readlines()   #读取文件
        t.append(data)
        for i in range(len(data)):
            txts.append(data[i])



#按行读入并按空格分割
s = []
for i in txts:
    s.append(i.split())

#按年份读入并按空格分割
ss = []
for i in t:
    a = []
    for j in i:
        a.append(j.split())
    ss.append(a)


#测试集名称

TDtxts = []

#读取所有文件
f2 = open('F://Jupyter/Data/CMABSTdata/CH2018BST.txt','r')
TestD = f2.readlines()   #读取文件
for i in range(len(TestD)):
    TDtxts.append(TestD[i])
#按行读入并按空格分割
sr = []
for i in TDtxts:
    sr.append(i.split())
na = []
for i in sr:
    if i[0] == '66666':
        na.append(i[7])






Tp = {}                    #建立字典，键为ID，值为坐标序列
name1 = 70                 #第一年70
Name = []                  #ID序列
for y in ss:
    if name1 >99:
        name1 = 0
    i = 0
    for d in range(len(y)):
        if y[d][0] == '66666':
            i += 1
            name = ("%02d"%name1)+("%02d"%i)
            Name.append(name)
    name1+=1
       
i = 0 
re = []                    #所有路径序列序列（TXT文件删除头记录后留下的，以台风为单位）
ar = []                    #每个序列的步长
timeindex = []
for y in ss:
    k = 0
    for d in y:
        if d[0] != '66666' and d[0] != 'end':
            i+=1 
            re.append(d)
            timeindex.append(int(d[0]))
        else:
          
           ar.append(i)
           i=0

ar.remove(ar[0])          # 因为编程的原因（60-72行）所以删除第一个值


        

index = 0
for i in range(len(ar)):
    a = []
    for l in range(ar[i]):
        
        a.append([float(re[index][2])/10.,float(re[index][3])/10.,timeindex[index]])
        index+=1
    Tp.update({Name[i]:a})  # 全部数据储存在字典Tp中
    
    
value = list(Tp.values())  #台风路径数据
 
Lens = []                  #路径步长
lessthan8 = 0              #步长小于8步个数
rmIndex  =[]               #需要移除路径步长小于8的索引值（list下标）
for i in range(len(value)):
    if len(value[i])<0: #or len(value[i])>80:
        lessthan8+=1
        rmIndex.append(i)
        
    
print('移除的个数',lessthan8,'台风个数 ',len(value),'台风个数',len(Name))  #移除的个数，台风个数 ，台风个数
#分割数据 训练集（训练+验证）：1970-2017 测试集2018
DivIndex = None           #分割位置
for i in range(len(Name)):
    if Name[i][0:2] == '18':
        DivIndex = i
        break
def FindIndex(A,N):
    index = -1
    
    for j in range(len(A)):
        if A[j] == N:
            index = j
            break        
    return index
    
#二分法查找下标
def binary_search(A,key):
    left = 0
    right = len(A) - 1
    while left <= right:
        mid = (left + right) 
        if key > A[mid]:
            left = mid + 1
        elif key < A[mid]:
            right = mid - 1
        else:
            return mid
    else:
        return -1
    
def str2date(s):
    time = s
    Year = int(time[:4])
    month = int(time[4:6])
    day = int(time[6:8])
    hour = int(time[8:])
    TT = date2num(datetime(Year,month,day,hour),units=Time.units)
    return TT

#训练集
traindata = value[:DivIndex]
#测试集
testdata = value[DivIndex:]


TData = nc.MFDataset("F://Jupyter/Data/Temperature/*nc")
UData = nc.MFDataset("F://Jupyter/Data/u_wind/*nc")
VData = nc.MFDataset("F://Jupyter/Data/v_wind/*nc")
WData = nc.MFDataset("F://Jupyter/Data/w_wind/*nc")
PData = nc.MFDataset("F://Jupyter/Data/pre/*nc")

Tem = TData.variables['air']
Time = TData.variables['time']

Lon = TData.variables['lon']
Lat = TData.variables['lat']

Lev = TData.variables['level']
Uwind = UData.variables['uwnd']
Vwind = VData.variables['vwnd']
Wwind = WData.variables['omega']
Pre = PData.variables['hgt']


#选择500hPa等压面数据
layer= None
for i in range(len(Lev)):
    if Lev[i] == 500.:
        layer = i
l1 = 10.
l2 = 70.
l3 = 90.
l4 = 200.

#l2= latmin,l1 =latmax,l3=lonmin,l4 =lonmax
#经纬范围
latmin,latmax,lonmin,lonmax = 0,0,0,0
for i in range(len(Lon)):
    if Lon[i]==l3:
        lonmin = i
    if Lon[i]==l4:
        lonmax = i
for i in range(len(Lat)):
    if Lat[i]==l1:
        latmax = i
    if Lat[i]==l2:
        latmin = i

#
Trdata = []
R = list(Time[:])
Timeindex = []
for i in traindata:
    A = []
    for j in i:
        time = str(j[2])
        Year = int(time[:4])
        month = int(time[4:6])
        day = int(time[6:8])
        hour = int(time[8:])
        TT = date2num(datetime(Year,month,day,hour),units=Time.units)
        index = FindIndex(R,TT)
        Timeindex.append(index)
        A.append(Tem[index,layer,latmin:latmax,lonmin:lonmax])
    Trdata.append(np.array(A))        

Trdata = np.array(Trdata)
Tedata = []

for i in testdata:
    A = []
    for j in i:
        time = str(j[2])
        Year = int(time[:4])
        month = int(time[4:6])
        day = int(time[6:8])
        hour = int(time[8:])
        TT = date2num(datetime(Year,month,day,hour),units=Time.units)
        index = FindIndex(R,TT)
        A.append(Tem[index,layer,latmin:latmax,lonmin:lonmax])
    Tedata.append(A)
Tedata = np.array(Tedata)


Train = []
for i in traindata:
    a = []
    for j in i:
        a.append([j[0],j[1]])
    Train.append(np.array(a))
Test = []
for i in testdata:
    a = []
    for j in i:
        a.append([j[0],j[1]])
    Test.append(np.array(a))

seq = tf.keras.models.Sequential()
seq.add(tf.keras.layers.Masking(mask_value=0.))
seq.add(tf.keras.layers.ConvLSTM2D(filters=2,kernel_size=(3, 3)))

seq.compile(loss='mean_squared_error', optimizer='adadelta')
Trdata = tf.keras.preprocessing.sequence.pad_sequences(Trdata,padding = 'post')
a = Trdata
Trdata = tf.keras.backend.reshape(Trdata,[1521, 101, 24, 44,1])
Test = tf.keras.preprocessing.sequence.pad_sequences(Test,padding = 'post')
Train = tf.keras.preprocessing.sequence.pad_sequences(Train,padding = 'post')

#seq.fit(Trdata,Train)


