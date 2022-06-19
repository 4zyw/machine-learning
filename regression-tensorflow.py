import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras
import warnings
warnings.filterwarnings("ignore")


#读取数据，观察数据
features=pd.read_csv('temps.csv')
print(features.head())
print('数据维度',features.shape)
#处理时间数据
import datetime
#分别得到年月日
years=features['year']
months=features['month']
days=features['day']
#datetime的格式
dates=[str(int(year))+'-'+str(int(month))+'-'+str(int(day)) for year,month,day in zip(years,months,days)]
dates=[datetime.datetime.strptime(date,'%Y-%m-%d') for date in dates]

#画图展示
#设计布局
fig,ax1=plt.subplots()
fig.autofmt_xdate(rotation=45)
#设计标签值
ax1.plot(dates,features['actual'])
ax1.set_xlabel('Date');ax1.set_ylabel('Temperature');ax1.set_title('Max Temp')

fig,ax2=plt.subplots()
fig.autofmt_xdate(rotation=45)
ax2.plot(dates,features['temp_1'])
ax2.set_xlabel('Date');ax2.set_ylabel('Temperature');ax2.set_title('Previous Max Temp')

fig,ax3=plt.subplots()
fig.autofmt_xdate(rotation=45)
ax3.plot(dates,features['temp_2'])
ax3.set_xlabel('Date');ax3.set_ylabel('Temperature');ax3.set_title('Two Days Prior Max Temp')

plt.show()

#预处理
features = pd.get_dummies(features)
print(features.head(5))

#标签
labels=np.array(features['actual'])
features=features.drop('actual',axis=1)
features=features.drop('forecast_noaa',axis=1)
features=features.drop('forecast_acc',axis=1)
features=features.drop('forecast_under',axis=1)
feature_list=list(features.columns)
features=np.array(features)
print(features.shape)
#数值标准化
from sklearn import preprocessing
input_features=preprocessing.StandardScaler().fit_transform(features)

#构建网络模型
model = tf.keras.Sequential()
model.add(layers.Dense(16,kernel_initializer='random_normal',kernel_regularizer=tf.keras.regularizers.l2(0.03)))
model.add(layers.Dense(32,kernel_initializer='random_normal',kernel_regularizer=tf.keras.regularizers.l2(0.03)))
model.add(layers.Dense(1,kernel_initializer='random_normal',kernel_regularizer=tf.keras.regularizers.l2(0.03)))

#训练
model.compile(optimizer=tf.keras.optimizers.SGD(0.001),loss='mean_squared_error')
model.fit(input_features,labels,validation_split=0.25,epochs=1000,batch_size=64)

#预测模型结果
predict=model.predict(input_features)
print(predict)


