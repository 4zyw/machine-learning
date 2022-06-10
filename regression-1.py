import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#读取文件
data=pd.read_csv('ex1data1.txt',names=['population','profit'])

#数据准备
data.plot.scatter('population','profit',label='population')
print(plt.show())
data.insert(0,'ones',1)
print(data.head())
X=data.iloc[:,0:-1]
y=data.iloc[:,-1]
print(X.head())
print(y.head())
X=X.values
print(X.shape)
y=y.values
print(y.shape)
y=y.reshape(97,1)
print(y.shape)

#损失函数的定义
def LossFuction(X,y,theta):
    inner=np.power(X@theta-y,2)
    return np.sum(inner)/(2*len(X))
theta=np.zeros((2,1))
print(theta.shape)
Loss_init=LossFuction(X,y,theta)
print(Loss_init)

#梯度下降
def gradientDescent(X,y,theta,alpha,iters):
    Losses=[]

    for i in range(iters):
        theta=theta-(X.T@(X@theta-y))*alpha/len(X)
        Loss=LossFuction(X,y,theta)
        Losses.append(Loss)

        print(Loss)

    return theta,Losses

alpha=0.01
iters=2000

theta,Losses=gradientDescent(X,y,theta,alpha,iters)


#可视化损失函数
fig,ax=plt.subplots()
ax.plot(np.arange(iters),Losses)
ax.set(xlabel='iters',ylabel='Loss',title='Loss vs iters')
plt.show()

#可视化拟合函数
x=np.linspace(y.min(),y.max(),100)
y_=theta[0,0]+theta[1,0]*x

fig,ax=plt.subplots()
ax.scatter(X[:,1],y,label='training data')
ax.plot(x,y_,'r',label='predict')
ax.legend()
ax.set(xlabel='population',ylabel='profit')
plt.show()

print(theta[0,0])
print(theta[1,0])


