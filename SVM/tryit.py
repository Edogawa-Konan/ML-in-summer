from sklearn import svm

X=[[2,0],[1,1],[2,3]]
y=[0,0,1]

classifier=svm.SVC(kernel='linear')

classifier.fit(X,y)

print('支持向量的索引：\n',classifier.support_)
print('支持向量:\n',classifier.support_vectors_)

print('支持向量的个数:\n',classifier.n_support_) #Number of support vectors for each class.

print(classifier.predict([[2,0]]))

