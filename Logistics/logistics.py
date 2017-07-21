import numpy as np
import random
from numpy.linalg import inv

# m denotes the number of examples here, not the number of features
def gradientDescent(x, y, theta, alpha, m, numIterations):
    '''
    :param x: 训练数据
    :param y: Label
    :param theta: theta向量，欲求的参数
    :param alpha: learning rate
    :param m: 实例的个数
    :param numIterations: 迭代次数
    :return:
    '''
    xTrans = x.transpose() #转置
    for i in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y #估计值和实际值的差
        # avg cost per example (the 2 in  2*m doesn't really matter here.
        # But to be consistent with the gradient, I include it)
        cost = np.sum(loss ** 2) / (2 * m)
        print("Iteration %d | Cost: %f" % (i, cost))
        # avg gradient per example
        gradient = np.dot(xTrans, loss) / m
        # update
        theta = theta - alpha * gradient
    return theta


def genData(numPoints, bias, variance):
    '''
    :param numPoints: 实例的个数，多少行
    :param bias: 偏置
    :param variance: 方差
    :return: X,y
    '''
    x = np.zeros(shape=(numPoints, 2))
    y = np.zeros(shape=numPoints)
    # basically a straight line
    for i in range(numPoints):
        # bias feature
        x[i][0] = 1 #第一列全是1
        x[i][1] = i #第二列是i
        # our target variable
        y[i] = (i + bias) + random.uniform(0, 1) * variance
        #random.uniform(a, b)返回一个随机浮点数，取值在0,1之间
    return x, y

# gen 100 points with a bias of 25 and 10 variance as a bit of noise
x, y = genData(100, 25, 10)
m, n = np.shape(x) #维度
numIterations= 100000
alpha = 0.0005
theta = np.ones(n)
theta = gradientDescent(x, y, theta, alpha, m, numIterations)
print('用梯度下降法求解theta为：')
print(theta)

print('直接对矩阵求导，解出theta：')
theta_2=np.dot(np.dot(inv(np.dot(x.T,x)),x.T),y)

print(theta_2)