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
from paper.DTW import *
import random 
import math

print('modeldtw 4步滑动窗口预测1步')

trainingdata,testdata,testname = Getdata(8)
testname.remove(testname[30])    




   
modeld = tf.keras.models.Sequential([
    
    
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
      #tf.keras.layers.Reshape((4,2))
])

modeld.compile(loss='mse', optimizer =tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999))

history = modeld.fit(train_X,train_Y,epochs=100,
                                               
                                          validation_split = 0.2,
                                          
                                          )
modelname = 'modeldtw 4stepsto1 16'
#PrintLossFig(history,modelname+' Training and validation loss')

pre = modeld.predict(test_X)
#pre = forecast(test_X,modeld,len(test_Y))
pre = forecast2(test_X,np.expand_dims(np.array(test_Y),0),modeld,len(test_Y))
PlotSingle(np.array(tsX),test_Y,pre[0],modelname+ ' Typohoon '+testname[ind])


