
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import netCDF4 as nc
from datetime import datetime, timedelta
from netCDF4 import num2date, date2num

file = open('F://Jupyter/Data/1970-2018bst.txt',mode = 'r')
TyphoonData = pd.read_csv('F://Jupyter/Data/1970-2018bst.csv')
TyphoonTime = pd.read_csv('F://Jupyter/Data/1970-2018date.csv')
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

def FindIndex(A,N):
    
    
    for i in range(len(A[:])):
        if A[i] == N:
            return i
            break
    return i

#数据
Date =TyphoonData['date']
TyphoonLat = TyphoonData['lat']
TyphoonLon = TyphoonData['lon']
TyphoonID = TyphoonTime['ID']
days = TyphoonTime['day']
len(TyphoonID)
len(days)
Date

def func(D,Lat,Lon):
    Date2 = []
    days = []
    lat = []
    lon = []
    k = 0
    
    for i in range(0,len(D)):
        if D[i] != 66666:
            timeStruct = time.strptime(str(D[i]),"%Y%m%d%H")
            strTime = time.strftime("%Y %m %d %H", timeStruct)
            t = strTime.split( ' ')
            T = datetime(int(t[0]),int(t[1]),int(t[2]),int(t[3]))
            unit = 'hours since 1800-01-01 00:00:0.0'
            a = date2num(T,units = unit)
            Date2.append(a)
            lat.append(Lat[i])
            lon.append(Lon[i])
            k+=1
        else:
            if(k!=0):
                days.append(k)
                k = 0
    return Date2,days,lat,lon

def func2(date,lat,lon,day,ID):
    Typhoon = []
    e = 0
    k = 0
    for j in day:
        t = []
        for i in range(e,e+j):
            t.append([date[i],lat[i]/10.0,lon[i]/10.0,ID[k]])
            e+=1
        k+=1
        Typhoon.append(t)
    return Typhoon

date,day ,lat ,lon= func(Date,TyphoonLat,TyphoonLon)


Typhoon  = func2(date,lat,lon,day,TyphoonID)
#
f2 = open('F://Jupyter/Data/list.txt','w')
for i in range(len(Typhoon)):
        for j in range(len(Typhoon[i])):
        
            f2.write(str(Typhoon[i][j][0])+" "+str(Typhoon[i][j][1])+" "+str(Typhoon[i][j][2])+" "+str(Typhoon[i][j][3]))
            f2.write('\n')

f2.close()






















