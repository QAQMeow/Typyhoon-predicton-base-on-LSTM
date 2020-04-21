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

print('model6 16步滑动窗口预测4步')

trainingdata,testdata,testname = Getdata(8)

testname.remove(testname[30])    

train_X = []
train_Y = []
for i in trainingdata:
    if len(i)<20:
        Padlen(i,20)
    for j in range(len(i)-20):
        train_X.append(i[j:j+16])
        train_Y.append(i[j+16:j+20])

train_X = np.array(train_X)
train_Y = np.array(train_Y)
    
test_X = []
test_Y = []
tsX = []
for i in testdata:
    tsX.append(i[:-4])
    if len(i)<20:
        Padlen(i,20)
    test_X.append(i[-20:-4])
    test_Y.append(i[-4:])

#test_X = Paddingdata(test_X,lenMax(test_X))
test_X = np.array(test_X)
test_Y = np.array(test_Y)

model6 = tf.keras.models.Sequential([
    
    
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

    tf.keras.layers.LSTM(8,
                        #  input_shape=padtrain_X.shape[-2:],
                          use_bias = True,
                          bias_initializer="zeros",
                          kernel_initializer = tf.keras.initializers.glorot_normal(), #权值初始化
                          recurrent_initializer = tf.keras.initializers.glorot_normal(), #循环层状态权值初始化
                          activity_regularizer = tf.keras.regularizers.l2(0.),
                          activation='tanh', 
                          recurrent_activation = "sigmoid",
                          ), 
    
      tf.keras.layers.Dense(8,activation='linear'),
      tf.keras.layers.Reshape((4,2))
])

model6.compile(loss='mse', optimizer =tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999))

history = model6.fit(train_X,train_Y,epochs=200,
                                               
                                         validation_split = 0.2,
                                          
                                          )
modelname = 'model6 16steps 8'
PrintLossFig(history,modelname+' Training and validation loss')

pre = model6.predict(test_X)

#pre = forecast(test_X,model5,4)

PlotAllmul(tsX,test_Y,pre,modelname,testname)







