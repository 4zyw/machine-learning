#数据获取
import tensorflow as tf
path=tf.keras.utils.get_file('cats_and_dogs_filtered.zip',origin='https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip')
print(path)

#文件解压
import zipfile

local_zip = path
zip_ref = zipfile.ZipFile(local_zip, 'r')

zip_ref.extractall('C:\\Users\\Administrator\\PycharmProjects\\pythonProject1\\tensorfolw\\data\\archive')
zip_ref.close()

#划分训练集和测试集
import os
base_dir = 'C:/Users/Administrator/PycharmProjects/pythonProject1/tensorfolw/data/archive/cats_and_dogs_filtered'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

train_cat_fnames = os.listdir( train_cats_dir )
train_dog_fnames = os.listdir( train_dogs_dir )

print(train_cat_fnames[:10])
print(train_dog_fnames[:10])

#查看数据集的大小
print('total training cat images :', len(os.listdir(train_cats_dir)))
print('total training dog images :', len(os.listdir(train_dogs_dir)))

print('total validation cat images :', len(os.listdir(validation_cats_dir)))
print('total validation dog images :', len(os.listdir(validation_dogs_dir)))

#画图
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

nrows = 4
ncols = 4

pic_index = 0

fig = plt.gcf()
fig.set_size_inches(ncols*4, nrows*4)

pic_index+=8

next_cat_pix = [os.path.join(train_cats_dir, fname)
                for fname in train_cat_fnames[ pic_index-8:pic_index]
               ]

next_dog_pix = [os.path.join(train_dogs_dir, fname)
                for fname in train_dog_fnames[ pic_index-8:pic_index]
               ]

for i, img_path in enumerate(next_cat_pix+next_dog_pix):
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off')
  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()

#模型建立
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()#观察神经网络的参数

#模型编译
from tensorflow.keras.optimizers import RMSprop

model.compile(optimizer=RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics = ['acc'])
#数据预处理
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#标准化到[0,1]
train_datagen = ImageDataGenerator( rescale = 1.0/255. )
test_datagen  = ImageDataGenerator( rescale = 1.0/255. )

#批量生成20个大小为150x150的图像及其标签用于训练
train_generator = train_datagen.flow_from_directory(train_dir,batch_size=20,class_mode='binary',target_size=(150, 150))
#批量生成20个大小为150x150的图像及其标签用于验证
validation_generator =  test_datagen.flow_from_directory(validation_dir,batch_size=20,class_mode  = 'binary',
                                                         target_size = (150, 150))

#模型训练
history = model.fit_generator(train_generator,validation_data=validation_generator,steps_per_epoch=100,
                              epochs=15,validation_steps=50,verbose=2)

#查看预测结果
import tkinter as tk
from tkinter import filedialog
'''打开选择文件夹对话框'''
root = tk.Tk()
root.withdraw()

Filepath = filedialog.askopenfilename() #获得选择好的图片

import numpy as np
from keras.preprocessing import image

path = Filepath
img = image.load_img(path, target_size=(150, 150))

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
images = np.vstack([x])

classes = model.predict(images, batch_size=10)
print(classes[0])

if classes[0] > 0:
    print("This is a dog")
else:
    print("This is a cat")

