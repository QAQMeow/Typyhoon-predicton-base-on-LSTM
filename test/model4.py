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

print('model4 8步滑动窗口预测4步')

trainingdata,testdata,testname = Getdata(12)
testname.remove(testname[30])

train_X = []
train_Y = []
for i in trainingdata:
    for j in range(len(i)-12):
        train_X.append(i[j:j+8])
        train_Y.append(i[j+8:j+12])

train_X = np.array(train_X)
train_Y = np.array(train_Y)
    

test_X = []
test_Y = []
tsX = []
for i in testdata:
    tsX.append(i[:-4])
    test_X.append(i[-12:-4])
    test_Y.append(i[-4:])

test_X = paddata(test_X,lenMax(test_X),2)
test_Y = np.array(test_Y)

# Nodes = [4,8,16]

# for Node in Nodes:
Node = 16
Loss = []
#for Node in Nodes:
model4 = tf.keras.models.Sequential([
    
    
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

    tf.keras.layers.LSTM(Node,
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

model4.compile(loss='mse', optimizer =tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999))

history = model4.fit(
                        train_X,train_Y,epochs=150,
                        validation_split = 0.2,
                                          
                                          )


loss = np.average(history.history['loss'][-10:])
valloss = np.average(history.history['val_loss'][-10:])
Loss.append([loss,valloss])
modelname = 'model4 8steps '+str(Node)

#PrintLossFig(history,modelname+'Training and validation loss')

pre = model4.predict(test_X)
#pre = forecast(test_X,model4,8)

PlotAllmul(tsX,test_Y,pre,modelname,testname)



ind = 5
a = tf.keras.losses.mean_squared_error(pre[ind,:,0], test_Y[ind,:,0]).numpy()
b = tf.keras.losses.mean_squared_error(pre[ind,:,1], test_Y[ind,:,1]).numpy()
print('lat:',a,'lon:',b)
PlotSingle(np.array(tsX[ind]),test_Y[ind],pre[ind],modelname+testname[ind])


