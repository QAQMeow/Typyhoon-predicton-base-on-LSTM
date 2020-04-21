from __future__ import absolute_import,division,print_function,unicode_literals

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

#导入MNIST数据
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
#标签
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',\
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#查看图片
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()    

#归一化

train_images = train_images/255.0
test_images = test_images/255.0

#显示训练集中前25个

plt.figure(figsize = (10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i],cmap = plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

#建立模型
model = keras.Sequential(
    [keras.layers.Flatten(input_shape = (28,28)),
     keras.layers.Dense(128,activation = 'relu'),
     keras.layers.Dense(10)
     ]
    )

#编译模型
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#训练模型
model.fit(train_images,train_labels,epochs = 10)

#评估模型
test_loss,test_acc = model.evaluate(test_images,test_labels,verbose = 2)
print('\nTest accuracy:',test_acc)

#预测
probability_model = tf.keras.Sequential(
    [model,
     tf.keras.layers.Softmax()
     ]
    )

predictions = probability_model.predict(test_images)





















