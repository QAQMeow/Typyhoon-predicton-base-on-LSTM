
import os
import numpy as np
import pandas as pd
from paper.function import *
import random 



    
def Getdata(minLens): 
      
    
   
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
        if len(value[i])<minLens or len(value[i])>80:
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
     
    #分割数据 训练集（训练+验证）：1949-2017 测试集（测试+验证）：2018
    DivIndex = None           #分割位置
    for i in range(len(Name)):
        if Name[i][0:2] == '18':
            DivIndex = i
            break
        
    trainlatmax = max(lat[:DivIndex])
    trainlatmin = min(lat[:DivIndex])
    trainlonmax = max(lon[:DivIndex])
    trainlonmin = min(lon[:DivIndex])
    
    
    testlatmax = max(lat[DivIndex:])
    testlatmin = min(lat[DivIndex:])
    testlonmax = max(lon[DivIndex:])
    testlonmin = min(lon[DivIndex:])
    
    trainlatmean = np.mean(lat[:DivIndex])
    trainlonmean = np.mean(lon[:DivIndex])
    trainlatstd = np.std(lat[:DivIndex])
    trainlonstd = np.std(lon[:DivIndex])
    
    testlatmean = np.mean(lat[DivIndex:])
    testlonmean = np.mean(lon[DivIndex:])
    testlatstd = np.std(lat[DivIndex:])
    testlonstd = np.std(lon[DivIndex:])
        
    #最大最小标准化
    # for i in range(len(value)):
    #     for j in value[i]:
    #         if i< DivIndex:
    #             j[0] = (j[0]-trainlatmin+1e-20)/(trainlatmax-trainlatmin)
    #             j[1] = (j[1]-trainlonmin+1e-20)/(trainlonmax-trainlonmin)
    #         else:
    #             j[0] = (j[0]-testlatmin+1e-20)/(testlatmax-testlatmin)
    #             j[1] = (j[1]-testlonmin+1e-20)/(testlonmax-testlonmin)
    
    #z-score 标准化
    for i in range(len(value)):
        for j in value[i]:
            if i< DivIndex:
                j[0] = (j[0]-trainlatmean)/trainlatstd
                j[1] = (j[1]-trainlonmean)/trainlonstd
            else:
                j[0] = (j[0]-testlatmean)/testlatstd
                j[1] = (j[1]-testlonmean)/testlonstd
    
    testdata = value[DivIndex:]   #测试集
    traindata = np.array(value[:DivIndex])  #训练集
    
     
    return traindata,testdata,na






def antitestnor(lat,lon):
    for i in lat:
        i = i*testlatstd+testlatmean
    for j in lon:
        j = j*testlonstd+testlonmean
