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
latmin = -10
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






# 从再分析数据中提取台风数据 
print(TData)
print(VData)
print(UData)
print(WData)
print(PData)

print("\n")
print(TData.dimensions)
print("\n")
print(UData.dimensions)
print("\n")
print(PData.dimensions)
print("\n")
print(Lev[:])

#首次进行数据抽取时
#TpD = nc.Dataset("F://Jupyter/Datatest.nc","w",format = "NETCDF4")

# #设置维数 时间 高度 经度 维度
# time = TpD.createDimension('time',None)
# level = TpD.createDimension('level',None)
# lat = TpD.createDimension('lat',73)
# lon = TpD.createDimension('lon',144)

# #设置变量 温度 风速 气压 时间 高度 经度 维度 及其单位
# times = TpD.createVariable('time','f8',('time',))
# times.units = TData['time'].units
# levels = TpD.createVariable('level','i4',('level',))
# levels.units = TData['level'].units
# latitudes = TpD.createVariable('lat','f4',('lat',))
# latitudes.units = TData['lat'].units
# longitudes = TpD.createVariable('lon','f4',('lon',))
# longitudes.units = TData['lon'].units

# air = TpD.createVariable('air','f4',('time','level', 'lat', 'lon'))
# air.units = TData['air'].units
# uwnd = TpD.createVariable('uwnd','f4',('time','level', 'lat', 'lon'))
# uwnd.units = UData['uwnd'].units
# vwnd = TpD.createVariable('vwnd','f4',('time','level', 'lat', 'lon'))
# vwnd.units = VData['vwnd'].units
# wwnd = TpD.createVariable('wwnd','f4',('time','level', 'lat', 'lon'))
# wwnd.units = WData['omega'].units
# hgt = TpD.createVariable('hgt','f4',('time','level', 'lat', 'lon'))
# hgt.units = PData['hgt'].units

#print(TpTrack)
# TimeSeries= TpTrack['Date2'].astype(np.float64) #从台风路径数据中选择时间序列
# TimeIndex = []
# T = TData['time'][:]
##二分法查找下标
# def binary_search(A,key):
#     left = 0
#     right = len(A) - 1
#     while left <= right:
#         mid = (left + right) 
#         if key > A[mid]:
#             left = mid + 1
#         elif key < A[mid]:
#             right = mid - 1
#         else:
#             return mid
#     else:
#         return -1
    
# st = datetime.now()
# print(st)
# for i in range(0,46657):
    
#     TimeIndex.append(binary_search(T,TimeSeries[i]))
# et = datetime.now()
# print(et)
# print(et-st)
# t1 = datetime.now()
# index = [x for x in TData['time'] if x in TimeSeries[:]]
# t2 = datetime.now()
# print(t2-t1)


#数据已抽取完，不再使用"w",再次读取时用"r"
#TpD = nc.Dataset("F://Jupyter/Datatest.nc","r",format = "NETCDF4")

# List = [str(i) for i in TimeIndex]
# with open("F:\Jupyter\TimeIndex.txt","w") as f: # 将台风时间序列下标写入TimeIndex.txt
#     for i in List:
#         f.write(i+'\n')
# f.close()
# 文件已写好用如下
f = open("F:\Jupyter\TimeIndex.txt","r")
data  = f.readlines()
f.close()
TimeIndex = [int(x) for x in data]   

##将选取的数据写入 Datatest.nc
# longitudes[:] = TData['lon'][:]
# latitudes[:] = TData['lat'][:]
# levels[:] =[1000,850,500,250,100]
# t1 = datetime.now()
# print(t1)
# indx  = range(0,46657)
# index = [1,23,44,45,65,89,99,123,233,345,456,777]
# air[:,:,:,:] = TData['air'][TimeIndex,[1,2,5,8,11],:,:]
# t2 = datetime.now()
# print(t2)
# print(t2-t1)

# air
#TpD.close()