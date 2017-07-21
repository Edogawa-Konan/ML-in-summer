from sklearn.datasets import load_digits
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from ANN.Neutral import NeuralNetwork


digits=load_digits()
X=digits.data
y=digits.target #label
#把X的所有值转化到0~1之间
X-=X.min()
X/=X.max()

print(digits.data.shape)

nn=NeuralNetwork([64,100,10],activation='logistic') #每个图片8*8，输出是10个数字一样。隐藏层可以适当灵活一些
X_train,X_test,y_train,y_test=train_test_split(X,y)

labels_train=LabelBinarizer().fit_transform(y_train)
labels_test=LabelBinarizer().fit_transform(y_test)

print('开始训练神经网络')

nn.fit(X_train,labels_train,epochs=3000)
prediction=[]

for line in X_test:
    o=nn.predict(line)
    prediction.append(np.argmax(o))


print(confusion_matrix(y_test,prediction))
print(classification_report(y_test,prediction))

