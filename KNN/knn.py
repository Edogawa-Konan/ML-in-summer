from sklearn import datasets
from sklearn import neighbors


K=neighbors.KNeighborsClassifier()#实现了knn算法的分类器
iris=datasets.load_iris() #鸢尾(花)数据集

K.fit(iris['data'],iris['target'])

print(iris.target_names) #也可以用iris['target_names']，如上

predicted_lable=K.predict([[0.1,0.2,0.3,0.4]])
print(predicted_lable)