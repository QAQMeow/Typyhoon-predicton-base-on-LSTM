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
def DTWModel(test,train):

    A = []
    for i in range(len(train)):
        A.append([dtw(test,train[i])[0],i])
    A.sort()
    ind = int(len(train)*0.33)
    
    traindata = []
    for i in range(ind):
        traindata.append(train[A[i][1]])
    return traindata

# for i in testdata:
preseq = testdata[22]
traindata = DTWModel(preseq,trainingdata)
train_X = []
train_Y = []


for i in traindata:
    for j in range(len(i)-4):
        train_X.append(i[j:j+4])
        train_Y.append(i[j+4])

train_X = np.array(train_X)
train_Y = np.array(train_Y)
    
test_X = preseq[-8:-4]
test_Y = preseq[-4:]

test_X = np.expand_dims(np.array(test_X),0)
test_Y = np.expand_dims(np.array(test_Y),0)

modelr = tf.keras.models.Sequential([
    
    
    # tf.keras.layers.LSTM(16,
    #                     # input_shape=padtrain_X.shape[-2:],
    #                       return_sequences=True, 
    #                       use_bias = True,
    #                       bias_initializer="zeros",
    #                       kernel_initializer = tf.keras.initializers.glorot_normal(), #权值初始化
    #                       recurrent_initializer = tf.keras.initializers.glorot_normal(), #循环层状态权值初始化
    #                       activity_regularizer = tf.keras.regularizers.l2(0.),#输出层激活函数正则化
    #                       activation='tanh', 
    #                       recurrent_activation = "sigmoid",
    #                       ),

    tf.keras.layers.LSTM(16,
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
      # tf.keras.layers.Reshape((4,2))
])

modelr.compile(loss='mse', optimizer =tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999))

history = modelr.fit(train_X,train_Y,epochs=200,
                                             
                                          validation_split = 0.2,
                                          
                                          )
modelname = 'modelr 8steps 4'

PrintLossFig(history,modelname+'Training and validation loss')

pre = modelr.predict(test_X)
#pre = forecast(test_X,modelr,4)
pre2 = forecast2(test_X,test_Y,modelr,4)
PlotSingle(np.array(preseq)[:-4],test_Y[0],pre2[0],modelname)

