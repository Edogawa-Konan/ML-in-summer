from numpy import genfromtxt
from sklearn import linear_model


file_path=r'E:\pycharm\ML\regression\delivery.csv'

data=genfromtxt(file_path,delimiter=',') #读取csv文件，返回ndarray

X=data[:,:-1]
y=data[:,-1]

print(X)
print(y)

reg=linear_model.LinearRegression()

reg.fit(X,y)

print('coefficient:',reg.coef_)

print('intercept:',reg.intercept_)

print('x为[102,6]预测结果为',reg.predict([[102,6]]))

