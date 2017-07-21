import numpy as np
import math


def computeCorrelation(X, Y):
    #计算皮尔斯系数
    xBar = np.mean(X)
    yBar = np.mean(Y)
    cov = 0
    varX = 0
    varY = 0
    for i in range(0, len(X)):
        diffXXBar = X[i] - xBar
        diffYYBar = Y[i] - yBar
        cov += (diffXXBar * diffYYBar)
        varX += diffXXBar ** 2
        varY += diffYYBar ** 2

    standard_deviation = math.sqrt(varX * varY)
    return cov / standard_deviation

def ployfix(x,y,degree):
    '''
    :param x: 一维向量
    :param y: 一维向量
    :param degree: 最高次数
    :return:
    '''
    results={}
    coefficients=np.polyfit(x,y,deg=degree)
    results['coefficient']=coefficients.tolist()
    p=np.poly1d(coefficients)
    y_hat=p(x) #代入x计算结果，即y的预测值.返回一维向量
    y_bar=np.sum(y)/len(y) #平均值
    ssres=np.sum((y-y_hat)**2)
    sstot=np.sum((y-y_bar)**2)
    results['determination']=1-ssres/sstot
    return results



testX = [1, 3, 8, 7, 9]
testY = [10, 12, 24, 21, 34]

print('Pearson correlation coefficient:',computeCorrelation(testX, testY))

print('决定系数为',computeCorrelation(testX,testY)**2)

print(ployfix(testX,testY,1)['determination'])