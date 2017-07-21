import numpy as np

def fix(x,y):
    n=len(x)
    numerator=0 #分子
    dinominator=0 #分母
    for v in zip(x,y):
        numerator+=(v[0]-np.mean(x))*(v[1]-np.mean(y))
        dinominator+=(v[0]-np.mean(x))**2
    b1=numerator/dinominator
    b0=np.mean(y)-b1*np.mean(x)
    return b0,b1

def predict(x,b0,b1):
    return b1*x+b0

x = [1, 3, 2, 1, 3]
y = [14, 24, 18, 17, 27]

b0,b1=fix(x,y)

print('截距intercept:',b0,'斜率slope:',b1)

print('当x为6时预测结果为:',predict(6,b0,b1))