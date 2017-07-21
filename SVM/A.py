import numpy as np
import pylab as pl

from sklearn import svm

np.random.seed(0) #固定随机化种子

#生成数据
X=np.r_[np.random.randn(20,2)-[2,2],np.random.randn(20,2)+[2,2]]
Y=[0]*20+[1]*20

#拟合模型
classifier=svm.SVC(kernel='linear')
classifier.fit(X,Y)

#超平面
w=classifier.coef_[0] #系数w0 w1
k=-w[0]/w[1] #斜率
xx=np.linspace(-5,5)
yy=k*xx-(classifier.intercept_[0]/w[1])

#上下两条平行线
b=classifier.support_vectors_[0]
yy_down=k*xx+(b[1]-k*b[0])

b=classifier.support_vectors_[-1]
yy_up=k*xx+(b[1]-k*b[0])

#画图
pl.plot(xx,yy,'k-')
pl.plot(xx,yy_down,'k--')
pl.plot(xx,yy_up,'k--')

pl.scatter(classifier.support_vectors_[:,0],classifier.support_vectors_[:,1],s=80)
pl.scatter(X[:,0],X[:,1],c=Y,cmap=pl.cm.Paired)

# pl.axis('tight')
pl.show()








