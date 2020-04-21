import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import csv
import netCDF4 as nc
from datetime import datetime, timedelta
from netCDF4 import num2date, date2num
from mpl_toolkits.basemap import Basemap,cm

# 将数据存入字典
def Datatodir(lat,lon,day,ID):
    t = 0
    Dir = {}
    for i in range(len(day)):
        a = []
        for j in range(day[i]):
            a.append([lon[t],lat[t]])
            t += 1
        Dir.update({ID[i]:a})
    return Dir

# 返回数据中最长序列长度
def lenMax(data):
    num_samples = len(data)          # 序列的个数。
    lengths = [len(s) for s in data] # 获取每个序列的长度]
    max_length = max(lengths)           # 最长序列的长度。
    return max_length

# 填充数据
def Paddingdata(data,lmax):
    #lmax 最长序列长度
    D = data
    for i in D:
        if len(i)<lmax:
            r = lmax -len(i)
            for j in range(r):
                i.insert(0,[0,0])
    return np.array(D)
    
def Padlen(ls,lamx):
    for i in range(lamx-len(ls)):
        ls.insert(0,[0,0])

   
def paddata(data,l,feature_dim):#填充
    num_samples = len(data)          # 序列的个数。                   # 如果是2维 （lat lon）feature_dim = 2
    padding_dataset = np.zeros([num_samples, l, feature_dim]) # 生成一个全零array来存放padding后的数据集
    
    for idx, seq in enumerate(data): # 将序列放入array中（相当于padding成一样长度）
        padding_dataset[idx, :len(seq), :] = seq
    return padding_dataset#,max_length

# 分割数据 
def Divdata(data,l,val_time_step):  
    #val_time_step 验证集长度 
    X = []
    Y = []
    
    for i in data:
        X.append(np.array(i[:(len(i)-val_time_step)]))
        Y.append(np.array(i[len(i)-val_time_step:]))
    
    X,Y = np.array(X),np.array(Y)
    
    return paddata(X,l,2),paddata(Y,lenMax(Y),2)
    

def testloss(pre,test_Y,i):
    A = tf.keras.losses.mean_squared_error(pre[:,:,:], test_Y[:,:,:])[i].numpy()
    return A
    
            
# 滑步预测4个时间步
def forecast(data,Model,steps):#steps : 用以计算的时间步 model2 6 model3 8 model4 4
    a = data
    for i in range(4):
        ss = Model.predict(a)
        for j in range(len(ss)):
            a[j] =  np.concatenate([a[j],ss[j:j+1]])[-steps:,:]
    return a[:,-4:,:]

# 滑步预测4个时间步
def forecast2(data1,data2,Model,steps):#steps : 用以计算的时间步 4,6,8....
    a = data1
    prd = np.zeros(data2.shape)
    for i in range(steps):
        ss = Model.predict(a)
        for j in range(len(data2)):
            a[j] =  np.concatenate([a[j],data2[j,i:i+1]])[-4:,:]
        for r in range(len(prd)):
            prd[r] =  np.concatenate([prd[r],ss[r:r+1]],axis = 0)[-steps:,:]
    return np.array(prd[:,:,:])



# 损失作图
def PrintLossFig(Loss,title):
    loss = Loss.history['loss']
    val_loss = Loss.history['val_loss']

    epochs = range(len(loss))

    plt.figure(figsize = (16,9))

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()
    plt.show()
    
    #plt.plot(epochs,(np.array(val_loss)-np.array(loss)))




# 绘制台风轨迹函数，输入 多组一连串坐标点和标题
def PrintTrackFig(data,title,lon,lat):
    
    latmin = -5
    latmax = 70
    lonmin = 90
    lonmax = 180
    
    m=Basemap(resolution='l',area_thresh=1000,projection='cyl',llcrnrlon=lonmin,urcrnrlon=lonmax,llcrnrlat=latmin,urcrnrlat=latmax)
    x, y = m(lon, lat)
    fig = plt.figure(figsize = (16,9))
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    clevls_hgt=[-80,-60,-40,-20,0,20,40,60,80]
    parallels = np.arange(-90.,90,30.)
    meridians = np.arange(0.,360.,60.)
    #CS2 = m.contourf(x,y,w1,20)
    #CS2 = m.contour(x,y,w1,15,linewidths = 0.5)
    #CS2.clabel(fontsize = 20,colors= 'k',fmt='%.2f')
    m.drawcoastlines(linewidth=0.1)
    #fig = plt.quiver(x,y,u1,v1,angles="uv",minshaft = 2,scale = 400,width = 0.0015,headwidth = 4)
    for i in data:
        X= []
        Y = []
        for j in i:
            X.append(j[1])
            Y.append(j[0])
        fig = plt.plot(X,Y,'--',alpha = 0.2)
    
    plt.title(title,size=20)
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
    
# 绘制结果 
def PlotAllmul(test_X,test_Y,pre,title,namelist):
    #pre 模型输出
    
    fig,ax = plt.subplots(5,7,figsize = (35,25))
    fig.suptitle(title,fontsize=50)
    PT = []
    for i in range(len(test_X)):
        x = 0
        y = 0
        
        X = []
        Y = []
        for j in test_X[i]:
            X.append(j[1])
            Y.append(j[0])
        a = pre[i,:,1]
        b = pre[i,:,0]
        c = test_Y[i][:,1]
        d = test_Y[i][:,0]
        PT.append([[X,Y],[a,b],[c,d]])
    
    i = 0
    for x in range(5):
        for y in range(7): 
            if i < len(test_X):
                ax[x,y].set_title(namelist[i])
                ax[x,y].plot(PT[i][0][0],PT[i][0][1])
                ax[x,y].plot(PT[i][1][0],PT[i][1][1],marker = 'x')
                ax[x,y].plot(PT[i][2][0],PT[i][2][1],marker = 'o')
            i += 1
                
    
    
def PlotAllsingle(test_X,test_Y,pre,title):
    fig,ax = plt.subplots(5,7,figsize = (35,25))
    fig.suptitle(title,fontsize=50)
    PT = []
    for i in range(len(test_X)):
        x = 0
        y = 0
        
        X = []
        Y = []
        for j in test_X[i]:
            X.append(j[1])
            Y.append(j[0])
        a = pre[i][1]
        b = pre[i][0]
        c = test_Y[i][1]
        d = test_Y[i][0]
        PT.append([[X,Y],[a,b],[c,d]])
    
    i = 0
    for x in range(5):
        for y in range(7): 
            if i < len(test_X):
                ax[x,y].plot(PT[i][0][0],PT[i][0][1])
                ax[x,y].plot(PT[i][1][0],PT[i][1][1],marker = '-x')
                ax[x,y].plot(PT[i][2][0],PT[i][2][1],marker = '-o')
            i += 1
                  
    
    
def PlotSingle(test_X,test_Y,pre,title):
    
    plt.figure(figsize=(11,8))
    plt.title(title)
    X = test_X[:,1]
    Y = test_X[:,0]
    a = pre[:,1]
    b = pre[:,0]
    c = test_Y[:,1]
    d = test_Y[:,0]
    
    plt.plot(X,Y)
    plt.plot(a,b,linestyle="--",label = 'predict')
    plt.plot(c,d,linestyle="-",label = 'true')
    plt.legend()
                  
      
    plt.show()
    



    
def Plotbtwsingle(test_X):
    fig,ax = plt.subplots(5,7,figsize = (35,25))
    #fig.suptitle(title,fontsize=50)
    PT = []
    for i in range(len(test_X)):
       
        
        X = []
        Y = []
        for j in test_X[i]:
            X.append(j[1])
            Y.append(j[0])
        
        PT.append([X,Y])
    
    i = 0
    for x in range(5):
        for y in range(7): 
            if i < len(test_X):
                ax[x,y].plot(PT[i][0],PT[i][1])
            i += 1 
    
    