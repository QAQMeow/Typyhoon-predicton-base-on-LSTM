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

print('model3 8步滑动窗口')

trainingdata,testdata,testname = Getdata(12)
testname.remove(testname[30])
def dtwprog(data1,seq):
    s = []
    for i in range(len(data1)):
        a = dtw(seq,data1[i])[0]
        s.append([a,i])
        s.sort()
    traindata = []
    index = int(len(s)*0.33)
    for i in range(index):
        
        traindata.append(data1[s[i][1]])
    
    return traindata
    
for l in range(len(testdata)):
    preseq = testdata[l]
    traindata = dtwprog(trainingdata,preseq)
    
    train_X = []
    train_Y = []
    for i in traindata:
        for j in range(len(i)-8):
            train_X.append(i[j:j+8])
            train_Y.append(i[j+8])
    
    train_X = np.array(train_X)
    train_Y = np.array(train_Y)
        
    
    tsX =np.array(preseq[:-4])
    test_X = Paddingdata(preseq[-12:-4],lenMax(preseq[-12:-4]))
    test_Y = preseq[-4:]
    
    test_X = np.expand_dims(np.array(test_X),0)
    test_Y = np.array(test_Y)
    
    models = tf.keras.models.Sequential([
        
    
        tf.keras.layers.LSTM(12,
                            #  input_shape=padtrain_X.shape[-2:],
                              use_bias = True,
                              bias_initializer="zeros",
                              kernel_initializer = tf.keras.initializers.glorot_normal(), #权值初始化
                              recurrent_initializer = tf.keras.initializers.glorot_normal(), #循环层状态权值初始化
                              activity_regularizer = tf.keras.regularizers.l2(0.),
                              activation='tanh', 
                              recurrent_activation = "sigmoid",
                              ), 
        
        tf.keras.layers.Dense(2,activation='linear'),
      
    ])
    
    models.compile(loss='mse', optimizer =tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999))

    history = models.fit(train_X,train_Y,epochs=60,
                                              
                                              validation_split = 0.2,
                                              
                                              )
    modelname = 'models steps (12) '
    
    #PrintLossFig(history,modelname+'Training and validation loss')
    
    preY = models.predict(test_X)
    
    pre = forecast(test_X,models,8)
    
    PlotSingle(tsX,test_Y,pre,modelname+testname[l])


