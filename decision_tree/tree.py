import csv
from pprint import pprint
from sklearn import preprocessing,tree
from sklearn.feature_extraction import DictVectorizer

featureList=[]#字典的列表，每个字典对应一个实例
lableList=[]#标题的列表

with open(r'E:\pycharm\ML\computer.csv','r') as f:
    r=csv.reader(f)
    header=next(r) #标题行
    for line in r:
        lableList.append(line[-1])
        dic={}
        for i in range(1,len(line)-1):
            dic[header[i]]=line[i]
        featureList.append(dic)

# pprint(featureList)

vec=DictVectorizer()#进行特征向量的变换

dummyX=vec.fit_transform(featureList).toarray()

# print(dummyX.shape)
# print(dummyX)

print(vec.get_feature_names())
# print(dummyX)

dummyY=preprocessing.LabelBinarizer().fit_transform(lableList)


classifier=tree.DecisionTreeClassifier(criterion='entropy')#利用信息熵时，用entropy（熵）。默认是gini
classifier.fit(dummyX,dummyY)
'''
根据训练集(X,Y)建立决策树，X的维度为[n_samples, n_features]，Y的维度是shape = [n_samples] or [n_samples, n_outputs]，代表标记
return:self
'''


with open('picture.dot','w') as f: #写入dot文件
    f=tree.export_graphviz(classifier,feature_names=vec.get_feature_names(),out_file=f)

Row=dummyX[0,:]
Row[0]=1
Row[2]=0
print('new Row:',str(Row))

predict=classifier.predict(Row)
print('预测结果为',str(predict))

