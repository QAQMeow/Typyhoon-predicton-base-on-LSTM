import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import csv
import netCDF4 as nc
from datetime import datetime, timedelta
from netCDF4 import num2date, date2num

file = open('F://Jupyter/Data/1970-2018bst.txt',mode = 'r')
TpTrack = pd.read_csv('F://Jupyter/Data/1970_2018TyphoonTrack.csv') # 台风路径数据来源
TpDate = pd.read_csv('F://Jupyter/Data/1970-2018date.csv')         # 台风序列时间

#1970-2018年 全球温度 气压 风速场再分析数据 NCEP-NCAR Reanalysis 1
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
    #寻找是相应时间在时间序列中的位置
    for i in range(len(A[:])):
        if A[i] == N:
            return i
            break
    return i


from mpl_toolkits.basemap import Basemap,cm
lon, lat = np.meshgrid(Lon,Lat)
latmin = 0
latmax = 70
lonmin = 90
lonmax = 190
im = []
DateTime = num2date(Time[1],units = Time.units)
DateTime.strftime("%Y%m%d%H")

def getData(TpTrack,TpDate):
    #将TyphoonTrack 中数据处理为单个台风为单位的数组 共1267组台风
    days = TpDate["day"]
    Tp = []
    n = 0
    for j in days:
        T = []
        for i in range(j):
            T.append([TpTrack["Date2"][n],TpTrack["Lat"][n],TpTrack["Lon"][n],TpTrack["ID"][n]])
            n=n+1
        Tp.append(T)
    return Tp
Tp = getData(TpTrack,TpDate)



m=Basemap(resolution='l',area_thresh=1000,projection='cyl',llcrnrlon=lonmin,urcrnrlon=lonmax,llcrnrlat=latmin,urcrnrlat=latmax)
x, y = m(lon, lat)
fig = plt.figure(figsize = (16,9))
ax = fig.add_axes([0.1,0.1,0.8,0.8])
clevls_hgt=[-80,-60,-40,-20,0,20,40,60,80]
parallels = np.arange(-90.,90,30.)
meridians = np.arange(0.,360.,60.)
  #  CS2 = m.contourf(x,y,w1,20)
    #CS2 = m.contour(x,y,w1,15,linewidths = 0.5)
    #CS2.clabel(fontsize = 20,colors= 'k',fmt='%.2f')
m.drawcoastlines(linewidth=0.1)
   # fig = plt.quiver(x,y,u1,v1,angles="uv",minshaft = 2,scale = 400,width = 0.0015,headwidth = 4)
for i in range(len(Tp)):
    X= []
    Y = []
    for j in Tp[i]:
        X.append(j[2])
        Y.append(j[1])
    fig = plt.plot(X,Y,'--')

plt.title('Typhoon Track 1970-2018',size=20)
plt.xlim(lonmin,lonmax)
plt.ylim(latmin,latmax)
     # Add Grid Lines
    # 绘制经纬线
m.drawparallels(np.arange(-90., 91., 20.), labels=[1,0,0,0], fontsize=20)
m.drawmeridians(np.arange(-180., 181., 40.), labels=[0,0,0,1], fontsize=20)
#m.scatter(X,Y,c = 'r')
# Add Coastlines, States, and Country Boundaries
m.drawcoastlines()
m.drawstates()
m.drawcountries()
plt.show()