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
from dtw import dtw
import matplotlib.pyplot as plt
print('test')

trainingdata,testdata,testname = Getdata(8)
testname.remove(testname[30])    
# def DTWModel(test,train):
    
#     A = []
#     for i in range(len(train)):
#         A.append([dtw(test,train[i])[0],i])
#     A.sort()
#     ind = int(len(train)*0.33)
    
#     traindata = []
#     for i in range(ind):
#         traindata.append(train[A[i][1]])
#     return traindata



def DTW(l1,l2):#D 距离矩阵 A 累计矩阵
    D = np.zeros((len(l2), len(l1)))
    for i in range(len(l2)):
        for j in range(len(l1)):
            #D[i,j] = (l2[j]-l1[i])**2 
            D[i,j] = (l1[j][1]-l2[i][1])**2 + (l1[j][0]-l2[i][0])**2
    A = np.zeros((len(l2), len(l1)))
    A[0,0] = 0
    for i in range(1, len(l2)):
        A[i,0] = D[i, 0] + A[i-1, 0] 
    for i in range(1, len(l1)):
        A[0,i] = D[0,i] + A[0, i-1] 
    for i in range(1, len(l2)):
        for j in range(1, len(l1)):
            A[i, j] = min(A[i-1, j-1], A[i-1, j], A[i, j-1]) + D[i, j]
    
    path = [[len(l1)-1, len(l2)-1]]
    i = len(l2)-1
    j = len(l1)-1
    while i>0 and j>0:
        if i==0:
            j = j - 1
        elif j==0:
            i = i - 1
        else:
            if A[i-1, j] == min(A[i-1, j-1], A[i-1, j], A[i, j-1]):
                i = i - 1#来自于左边
            elif A[i, j-1] == min(A[i-1, j-1], A[i-1, j], A[i, j-1]):
                j = j-1#来自于下边
            else:
                i = i - 1#来自于左下边
                j= j- 1
        path.append([j, i])
    path.append([0,0])
    a = A[-1,-1]
    # for i in range(len(path)):
    #     a+=A[path[i][1],path[i][0]]
    
    return [D,A,np.array(path),np.array(a)]
    
def plotDTW(D,path):
    plt.figure(figsize=(12,14))
    plt.imshow(D, interpolation='nearest', cmap='binary') 
    plt.plot(path[:,0],path[:,1])
    plt.gca().invert_yaxis()#倒转y轴，让它与x轴的都从左下角开始
    plt.xlabel("X")
    plt.ylabel("Y")
    #plt.grid()
    plt.colorbar(fraction=0.045, pad=0.1)
   
    
def plotdtw(D,path):
    plt.figure(figsize=(12,14))
    plt.imshow(D, interpolation='nearest', cmap='binary') 
    plt.plot(path[0],path[1])
    plt.gca().invert_yaxis()#倒转y轴，让它与x轴的都从左下角开始
    plt.xlabel("X")
    plt.ylabel("Y")
    #plt.grid()
    plt.colorbar(fraction=0.045, pad=0.1)

    
def dtwprog(data1,seq):
    s = []
    for i in range(len(data1)):
        a = DTW(seq,data1[i])[3]
        s.append([a,i])
        s.sort()
    traindata = []
    index = int(len(s)*0.33)
    for i in range(index):
        traindata.append(data1[s[i][1]])
    return traindata

def plotseq(data,title):
    fig,ax = plt.subplots(1,5,figsize = (15,3))
    fig.suptitle(title,fontsize=15)
    PT = []
    for i in range(len(data)):
        X = []
        Y = []
        for j in data[i]:
            X.append(j[1])
            Y.append(j[0])
        
        PT.append([X,Y])
    
    i = 0
#for x in range(1):
    for y in range(5): 
        if i < len(data):
            ax[y].plot(PT[i][0],PT[i][1])
            ax[y].set_xticks([])
            ax[y].set_yticks([])
        i += 1 
 
# for i in [5,20,22,25]:
#     A = []
#     preseq = testdata[i]
#     A.append(preseq)
#     traindata = dtwprog(trainingdata,preseq)[:4]
#     for j in traindata:
#         A.append(j)
#     plotseq(A,testname[i])

# seq = testdata[5]
# A = DTW(seq,testdata[26])[1]
# pa = DTW(seq,testdata[26])[2]
# plotDTW(A,pa)

# tr = dtwprog(trainingdata,seq)
# A1 = DTW(seq,tr[0])[1]
# pa1 = DTW(seq,tr[0])[2]
# plotDTW(A1,pa1)

# PlotTytr(seq,testname[5])
# PlotTytr(testdata[26],testname[26])

