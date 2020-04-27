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
from paper.Getdata import *
import random 
import math

path = "F://Jupyter/Data/CMABSTdata"
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
name1 = 49                 #第一年1949
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
for y in ss:
    k = 0
    for d in y:
        if d[0] != '66666' and d[0] != 'end':
            i+=1 
            re.append(d)
        else:
          
           ar.append(i)
           i=0

ar.remove(ar[0])          # 因为编程的原因（60-72行）所以删除第一个值


        

index = 0
for i in range(len(ar)):
    a = []
    for l in range(ar[i]):
        
        a.append([float(re[index][2])/10.,float(re[index][3])/10.])
        index+=1
    Tp.update({Name[i]:a})  # 全部数据储存在字典Tp中
    
    
value = list(Tp.values())  #台风路径数据
 
Lens = []                  #路径步长
lessthan8 = 0              #步长小于8步个数
rmIndex  =[]               #需要移除路径步长小于8的索引值（list下标）
for i in range(len(value)):
    if len(value[i])<1:#or len(value[i])>80:
        lessthan8+=1
        rmIndex.append(i)
        
    
print('移除的个数',lessthan8,'台风个数 ',len(value),'台风个数',len(Name))  #移除的个数，台风个数 ，台风个数
rmIndex = rmIndex[::-1]               #list反转，从后面向前开始移除
for i in rmIndex:
     del Tp[Name[i]]
     Name.remove(Name[i])
     value.remove(value[i])
   
    
lat = []                  #纬度
lon  = []                 #经度
for i in value:
    for j in i:
        lat.append(j[0])
        lon.append(j[1])




lat = np.array(lat)
lon = np.array(lon)


#初步查看数据
#PrintTrackFig(value,'1949-2018',lat,lon)


#台风序列长度分布
A = np.zeros([102])

for i in range(102):
    for j in value:
        if len(j)==i:
            A[i]+=1

#长度在8-60的序列占比
OF = sum(A[8:65])/sum(A)

plt.figure(figsize=(14,10))          
plt.bar(x = range(0,102),height = A)
plt.xlabel("Length",fontsize = 15)
plt.ylabel("Number of sequence",fontsize = 15)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.show()

#每年台风数变化
B = np.zeros(70)
for i in range(len(ss)):
    for j in ss[i]:
        if j[0]=='66666':
            B[i]+=1
C = np.arange(1949,2019,1)
P = []
for i in C:
    P.append(str(i))
plt.figure(figsize=(14,10))   
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12) 
plt.xlabel("Year",fontsize = 15)
plt.ylabel("number of Tropical cyclone",fontsize = 15)
plt.plot(C,B)
plt.show()

trainingdata,testdata,testname = Getdata(8)




M1 =np.array([[0.032131393579145275, 0.03136175263259146],
 [0.02583236592925257, 0.025241292616559396],
 [0.020503136018498076, 0.02119857455148465]])

M2 = np.array([[0.008333793047229187, 0.006972923687856107],
 [0.007895942828827161, 0.0067438682393771935],
 [0.007764942715080059, 0.006597981832392327]])

M3 = np.array([[0.008448939581439363, 0.007067060232265603],
 [0.008011814277422707, 0.006781950541466507],
 [0.007818672512086983, 0.006638097073642516]])

M4 = np.array([[0.00842812507969462, 0.007040250534989986],
 [0.008101919802666873, 0.0068311374655234776],
 [0.007883195184620273, 0.0066675954532375275]])

#trainingloss
X1 = ['model1','model2','model3','model4']
X = np.arange(4)
plt.figure(figsize=(14,10))  
plt.title("Loss of training set",fontsize = 25)
plt.bar(X-0.2,[M1[0][0],M2[0][0],M3[0][0],M4[0][0]],width = 0.18,label = '4 Nodes')
plt.bar(X,[M1[1][0],M2[1][0],M3[1][0],M4[1][0]],width = 0.18,label = '8 Nodes')
plt.bar(X+0.2,[M1[2][0],M2[2][0],M3[2][0],M4[2][0]],width = 0.18,label = '16 Nodes')
plt.xticks(X,X1,fontsize = 15)
plt.legend()
plt.show()
        
#valloss
plt.figure(figsize=(14,10))  
plt.title("Loss of validation set",fontsize = 25)
plt.bar(X-0.2,[M1[0][1],M2[0][1],M3[0][1],M4[0][1]],width = 0.18,label = '4 Nodes')
plt.bar(X,[M1[1][1],M2[1][1],M3[1][1],M4[1][1]],width = 0.18,label = '8 Nodes')
plt.bar(X+0.2,[M1[2][1],M2[2][1],M3[2][1],M4[2][1]],width = 0.18,label = '16 Nodes')
plt.xticks(X,X1,fontsize = 15)
plt.legend()
plt.show()
        
#traininglosschange

N = [1,2,3]
N1 = ['4 Nodes','8 Nodes','16 Nodes']
plt.figure(figsize=(14,10))  
plt.title("Changes in training set loss",fontsize = 25)
#plt.plot(N,[M1[0][0],M1[1][0],M1[2][0]],label = 'Model1')
plt.plot(N,[M2[0][0],M2[1][0],M2[2][0]],label = 'Model2')
plt.plot(N,[M3[0][0],M3[1][0],M3[2][0]],label = 'Model3')
plt.plot(N,[M4[0][0],M4[1][0],M4[2][0]],label = 'Model4')
plt.xticks(N,N1,fontsize = 15)
plt.legend()
plt.show()
        

plt.figure(figsize=(14,10))  
plt.title("Changes in validation set loss",fontsize = 25)
#plt.plot(N,[M1[0][0],M1[1][0],M1[2][0]],label = 'Model1')
plt.plot(N,[M2[0][1],M2[1][1],M2[2][1]],label = 'Model2')
plt.plot(N,[M3[0][1],M3[1][1],M3[2][1]],label = 'Model3')
plt.plot(N,[M4[0][1],M4[1][1],M4[2][1]],label = 'Model4')
plt.xticks(N,N1,fontsize = 15)
plt.legend()
plt.show()   
        
        
#CIMARON

m1 = [0.00547221 ,0.00172447 ,0.00059665, 0.0108586 ]
m2 = [0.0007653  ,0.0020859  ,0.00754    ,0.05596675]
m3 = [0.00090906 ,0.00116911 ,0.00732666 ,0.04023758]
m4 = [0.00020452 ,0.00134303 ,0.00965089 ,0.04231174]

x = ['step 1','step 2','step 3','step 4']
plt.figure(figsize=(14,10))
plt.title("Typhooon CIMARON 4 steps loss",fontsize = 25)  
plt.plot(x,m1,label = 'Model1',marker = 'o')
plt.plot(x,m2,label = 'Model2',marker = 'o')
plt.plot(x,m3,label = 'Model3',marker = 'o')
plt.plot(x,m4,label = 'Model4',marker = 'o')
plt.xticks(fontsize = 15)
plt.legend()S
plt.show() 



#

M5 = [0.008136027895463235, 0.006998909107178748]
M6 = [0.008589272367830369, 0.007501810539430538]
M7 = [0.009294320462764172, 0.008138845662202422]
R1 = [M2[2][0],M3[2][0],M4[2][0],M5[0],M6[0],M7[0],M1[2][0]]
R2 = [M2[2][1],M3[2][1],M4[2][1],M5[1],M6[1],M7[1],M1[2][1]]
r = ['model2(4)','model3(6)','model4(8)','model5(12)','model6(16)','model7(20)','model1(all)']
plt.figure(figsize=(14,10))
plt.plot(r,R1,label = 'Training set loss',marker = 'o')
plt.plot(r,R2,label = 'Validation set loss',marker = 'o')
plt.xticks(fontsize = 12)
plt.legend()
plt.show()









#


# for ind in [5,20,22,25]:
#     a = tf.keras.losses.mean_squared_error(pre[ind,:,:], test_Y[ind,:,:]).numpy()
#     print(a)
#     PlotSingle(np.array(tsX[ind]),test_Y[ind],pre[ind],modelname+testname[ind])



m20 = np.array([[0.0005156  ,0.00077321 ,0.00045377 ,0.00566213 ],
                [0.0022857  ,0.00549428 ,0.00134557 ,0.01526994],
                [0.00019745 ,0.00089148 ,0.01125346 ,0.04734518],
                [6.29748930e-05 ,5.02312266e-04 ,2.42994128e-03 ,7.33519906e-03]
                ])

m21 = np.array([[0.00051016 ,0.00229649 ,0.01031224 ,0.02626692],
                [0.00052495 ,0.00069127 ,0.00279989 ,0.032174  ],
                [0.00011027 ,0.00051803 ,0.00033403 ,0.01436079],
                [0.00022725 ,0.00248366 ,0.00839304 ,0.01729226]
                ])

m22 = np.array([[0.00052798 ,0.0003791  ,0.00260917 ,0.00157912],
                [0.00031719 ,0.00050895 ,0.00228689 ,0.00683135],
                [0.00011893 ,0.00012404 ,0.00108369 ,0.010082  ],
                [0.00013597 ,0.00046813 ,0.00023456 ,0.00112126]
                ])


testname.remove(testname[30])

NN= [testname[5],testname[20],testname[22],testname[25]]

fig,ax = plt.subplots(1,4,figsize = (20,5   ))

labels = ['model2','model2-1','model2-2']
for i in range(4):
    ax[i].plot(m20[i],label = labels[0])
    ax[i].plot(m21[i],label = labels[1])
    ax[i].plot(m22[i],label = labels[2])
    ax[i].set_title(NN[i])

fig.legend(labels)










