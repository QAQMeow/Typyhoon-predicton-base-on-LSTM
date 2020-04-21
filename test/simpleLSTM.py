from __future__ import absolute_import,division,print_function,unicode_literals
import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

mpl.rcParams['figure.figsize'] = (8,6)
mpl.rcParams['axes.grid'] = False
zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path)

df = pd.read_csv(csv_path)
def univariate_data(dataset,start_index,end_index,history_size,target_size):
    data = []
    labels = []
    
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size
    
    for i in range(start_index,end_index):
        indices = range(i- history_size,i)
        data.append(np.reshape(dataset[indices],(history_size,1)))
        labels.append(dataset[i+target_size])
    return np.array(data),np.array(labels)

TRAIN_SPLIT = 300000
tf.random.set_seed(13)

uni_data = df['T (degC)']
uni_data.index = df['Date Time']
uni_data.head()
#uni_data.plot(subplots=True)

uni_data = uni_data.values

uni_train_mean = uni_data[:TRAIN_SPLIT].mean() #均值
uni_train_std = uni_data[:TRAIN_SPLIT].std() #标准差
# 标准化数据
uni_data = (uni_data-uni_train_mean)/uni_train_std


univariate_past_history = 20
univariate_future_target = 0

x_train_uni,y_train_uni = univariate_data(uni_data, 0,\
                                          TRAIN_SPLIT, univariate_past_history, univariate_future_target)
x_val_uni,y_val_uni = univariate_data(uni_data,\
                                          TRAIN_SPLIT,None, univariate_past_history, univariate_future_target)


def create_time_steps(length):
    return list(range(-length,0))

def show_plot(plot_data,delta,title):
    labels = ['History','True Funture','Model Prediction']
    markers = ['.-','rx','go']
    time_steps = create_time_steps(plot_data[0].shape[0])
    if delta:
        future = delta
    else:
        future = 0
    plt.title(title)
    for i,x in enumerate(plot_data):
        if i:
            plt.plot(future,plot_data[i],markers[i],markersize = 10,label = labels[i])
        else:
            plt.plot(time_steps,plot_data[i].flatten(),markers[i],markersize = 10,label = labels[i])
    plt.legend()
    plt.xlim([time_steps[0],(future+5)*2])
    plt.xlabel('Time-Step')
    return plt

def baseline(history):
    return np.mean(history)

#show_plot([x_train_uni[0], y_train_uni[0]], 0,'Baseline Prediction Example')
#show_plot([x_train_uni[0], y_train_uni[0], baseline(x_train_uni[0])], 0,'Baseline Prediction Example')

BATCH_SIZE = 256
BUFFER_SIZE = 10000

# train_unvariate = tf.data.Dataset.from_tensor_slices((x_train_uni,y_train_uni))
# train_unvariate = train_unvariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

# val_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni,y_train_uni))
# val_univariate = val_univariate.batch(BATCH_SIZE).repeat()

# simlpe_lstm_model = tf.keras.models.Sequential([tf.keras.layers.LSTM(8,input_shape = x_train_uni.shape[-2:]),\
#                                                 tf.keras.layers.Dense(1)])
# simlpe_lstm_model.compile(optimizer = 'adam',loss = 'mae')
            
            
# for x,y in val_univariate.take(1):
#     print(simlpe_lstm_model.predict(x).shape)
            
            
# EVALUATION_INTERVAL = 200
# EPOCHS = 10
# simlpe_lstm_model.fit(train_unvariate,epochs = EPOCHS,steps_per_epoch = EVALUATION_INTERVAL,\
#                       validation_data = val_univariate,validation_steps = 50)        
            
            
# for x, y in val_univariate.take(3):
#   plot = show_plot([x[0].numpy(), y[0].numpy(),simlpe_lstm_model.predict(x)[0]], 0, 'Simple LSTM model')
#   plot.show()
            
            
# 2 预测多变量
features_considered = ['p (mbar)', 'T (degC)', 'rho (g/m**3)']
features = df[features_considered]
features.index = df['Date Time']
features.head()

#标准化数据 TRAIN_SPLIT = 300000
dataset = features.values
data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
data_std = dataset[:TRAIN_SPLIT].std(axis=0)     
dataset = (dataset-data_mean)/data_std



#2.1单步模型
def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i, step)
    data.append(dataset[indices])

    if single_step:
      labels.append(target[i+target_size])
    else:
      labels.append(target[i:i+target_size])

  return np.array(data), np.array(labels)
            
            
past_history = 720 #每小时采样的 720 个观测值
future_target = 72 #72（12+6） 观测后的温度。
STEP = 6

# x_train_single, y_train_single = multivariate_data(dataset, dataset[:, 1], 0,
#                                                     TRAIN_SPLIT, past_history,
#                                                     future_target, STEP,
#                                                     single_step=True)
# x_val_single, y_val_single = multivariate_data(dataset, dataset[:, 1],
#                                                 TRAIN_SPLIT, None, past_history,
#                                                 future_target, STEP,
#                                                 single_step=True)      
            

# train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
# train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

# val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
# val_data_single = val_data_single.batch(BATCH_SIZE).repeat()


# single_step_model = tf.keras.models.Sequential()
# single_step_model.add(tf.keras.layers.LSTM(32,
#                                             input_shape=x_train_single.shape[-2:]))
# single_step_model.add(tf.keras.layers.Dense(1))

# single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')

# single_step_history = single_step_model.fit(train_data_single, epochs=EPOCHS,
#                                             steps_per_epoch=EVALUATION_INTERVAL,
#                                             validation_data=val_data_single,
#                                             validation_steps=50)

# def plot_train_history(history, title):
#     loss = history.history['loss']
#     val_loss = history.history['val_loss']
    
#     epochs = range(len(loss))
#     plt.figure()
  
#     plt.plot(epochs, loss, 'b', label='Training loss')
#     plt.plot(epochs, val_loss, 'r', label='Validation loss')
#     plt.title(title)
#     plt.legend()  
#     plt.show()
  
# plot_train_history(single_step_history,
#                     'Single Step Training and validation loss')


# for x, y in val_data_single.take(3):
#   plot = show_plot([x[0][:, 1].numpy(), y[0].numpy(),
#                     single_step_model.predict(x)[0]], 12,
#                     'Single Step Prediction')
#   plot.show()




#多步进模型

future_target = 72
x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 1], 0,
                                                  TRAIN_SPLIT, past_history,
                                                  future_target, STEP)
x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 1],
                                              TRAIN_SPLIT, None, past_history,
                                              future_target, STEP)

train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()

# def multi_step_plot(history, true_future, prediction):
#   plt.figure(figsize=(12, 6))
#   num_in = create_time_steps(len(history))
#   num_out = len(true_future)

#   plt.plot(num_in, np.array(history[:, 1]), label='History')
#   plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo',
#            label='True Future')
#   if prediction.any():
#     plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',
#              label='Predicted Future')
#   plt.legend(loc='upper left')
#   plt.show()


# for x, y in train_data_multi.take(1):
#   multi_step_plot(x[0], y[0], np.array([0]))



multi_step_model = tf.keras.models.Sequential()
multi_step_model.add(tf.keras.layers.LSTM(32,
                                          return_sequences=True,
                                          input_shape=x_train_multi.shape[-2:]))
multi_step_model.add(tf.keras.layers.LSTM(16, activation='relu'))
multi_step_model.add(tf.keras.layers.Dense(72))

multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')

for x, y in val_data_multi.take(1):
  print (multi_step_model.predict(x).shape)



# multi_step_history = multi_step_model.fit(train_data_multi, epochs=EPOCHS,
#                                           steps_per_epoch=EVALUATION_INTERVAL,
#                                           validation_data=val_data_multi,
#                                           validation_steps=50)


# for x, y in val_data_multi.take(3):
#   multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0])







