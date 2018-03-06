# 暑期实习报告

前言：本文是我@prime在暑期2017.7.17-2017.8.6机器学习的笔记。其主要包含两个部分，首先基础篇，因为都比较简单，所以主要把概念理清，以及对用到的库进行记录，代码比较短小；第二部分进阶篇，由于整个进阶篇都是神经网络的方方面面，所以，我把它们整理到了一起，一气呵成，其主要是手动实现，所以我的公式推导过程也就是代码含义之所在，后面用到了卷积神经网络，我在本机上用GTX860M加速，完成了该实验。

[TOC]



## 基础篇

### 监督学习——分类

#### 决策树

判定树是一个类似于流程图的树结构：其中，每个内部结点表示在一个属性上的测试，每个分支代表一个属性输出，而每个树叶结点代表类或类分布。树的最顶层是根结点。

------

**熵的概念**：

[wiki](https://en.wikipedia.org/wiki/Entropy_(information_theory))

熵的概念最早起源于物理学，用于度量一个热力学系统的无序程度。在信息论里面，熵是**对不确定性的测量**，**熵**是接收的每条消息中包含的信息的平均量，单位通常为**比特**。在信息世界，熵越高，则能传输越多的信息，不确定性越大，熵越低，则意味着传输的信息越少，不确定性越低。

我们不知道某事物具体状态，却知道它有几种可能性时，显然，可能性种类愈多，不确定性愈大。不确定性愈大的事物，我们最后确定了、知道了，这就是说我们从中得到了愈多的信息，也就是信息量大。*所以，熵、不确定性、信息量，这三者是同一个数值。*

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/67841ec4b4f7e6ab658842ef2f53add46a2debbd)

在这里b是对数所使用的底，通常是2,自然常数e，或是10。当b = 2，熵的单位是bit；当b = e，熵的单位是nat；而当b = 10,熵的单位是Hart。

1. 构造决策树的基本算法——决策树归纳算法（Iterative Dichotomiser 3简称ID3算法）

   思路**关键就是选择哪个属性作为结点**

   信息获取量(Information Gain)：Gain(A) = Info(D) - Infor_A(D)

   即用先不考虑属性A的信息熵-考虑属性A后的信息熵

   每次选择一个Gain最大的作为结点。

2. 优缺点

   - 优点

     直观，便于理解

   - 缺点

     处理连续变量不好（连续变量必须离散化）

     类别较多时，错误增加的比较快

     小规模数据集有效

以下是代码实现，Graphviz可以将dot文件转换成pdf，其即是可视化的文件。Graphviz安装后，把其bin目录添加到环境变量中，用`dot -Tpdf 输入文件名.dot -o 输出文件名.pdf`即可。

本次实验用到的数据集如下：

![snipaste_20170806_092023](http://on7mhq4kh.bkt.clouddn.com//%E6%9A%91%E6%9C%9F%E5%AE%9E%E4%B9%A02017/snipaste_20170806_092023.png)

```python
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
```

代码运行之后，决策树如下所示：

![snipaste_20170806_092105](http://on7mhq4kh.bkt.clouddn.com//%E6%9A%91%E6%9C%9F%E5%AE%9E%E4%B9%A02017/snipaste_20170806_092105.png)

#### KNN算法（k-Nearest Neighbor）

1. 基本流程

   - 为了判断未知实例的类别，以所有已知类别的实例作为参照
   - 选择参数K
   - 计算未知实例与所有已知实例的距离
   - 选择最近K个已知实例
   - 根据少数服从多数的投票法则(majority-voting)，让未知实例归类为K个最邻近样本中最多数的类别

   针对距离的衡量，选择**Euclidean distance，即两点间的直线距离**。

2. 算法优缺点

   - 优点

     简单

     易于理解

     容易实现

     通过对K的选择可具备丢弃噪音数据的健壮性

   - 缺点

     需要大量空间储存所有已知实例

     算法复杂度高（需要比较所有已知实例与要分类的实例）

     当其样本分布不平衡时，比如其中一类样本过大（实例数量过多）占主导的时候，新的未知实例容易被归类为这个主导样本，因为这类样本实例的数量过大，但这个新的未知实例实际并木接近目标样本

3. 改进

   引入距离作为权重。

**数据集iris介绍：**

```python
sklearn.datasets.load_iris(return_X_y=False)
#return_X_y : boolean, default=False.If True, returns (data, target) instead of a Bunch object。
#return:
'''
data : Bunch
Dictionary-like object, the interesting attributes are: ‘data’, the data to learn, ‘target’, the classification labels, ‘target_names’, the meaning of the labels, ‘feature_names’, the meaning of the features, and ‘DESCR’, the full description of the dataset.
(data, target) : tuple if return_X_y is True
'''
```

该数据集包括3个类别，每个类50个实例，共计150个实例。

**KNN算法的封装**：

```python
class sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1, **kwargs)
```

更多内容见[详细文档](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

这里用到了其`fit(X,Y)`方法，用X作为训练数据，Y是target。<u>大部分情况下，fit基本上就是建立模型,其有拟合之意。</u>

关于测试用了`predict(X)`方法，这里选择了参数X的维度是 (n_query, n_features)，返回一个数组，每个值都代表Class lable。

以下是利用库方法实现KNN算法：

```python
from sklearn import datasets
from sklearn import neighbors


K=neighbors.KNeighborsClassifier()#实现了knn算法的分类器
iris=datasets.load_iris() #鸢尾(花)数据集

K.fit(iris['data'],iris['target'])

print(iris.target_names) #也可以用iris['target_names']，如上

predicted_lable=K.predict([[0.1,0.2,0.3,0.4]])
print(predicted_lable)
```

封装的很好，很简洁。

------

手动实现如下 

先补充一些知识：

```python
operator.itemgetter(item)
operator.itemgetter(*items)
```

它们都是返回一个可调用对象，该对象会对其参数调用`__getitem__()`方法，也就是`[]`运算符。给定多个参数，就返回元组，示例如下

- After `f = itemgetter(2)`, the call `f(r)` returns `r[2]`.
- After `g = itemgetter(2, 5, 3)`, the call `g(r)` returns `(r[2], r[5], r[3])`.

```python
sorted(iterable, *, key=None, reverse=False)
#Return a new sorted list from the items in iterable!
```

可以利用reverse进行相反排序，key参数可以传递排序方法。**特别注意返回的是一个list。**

```python
import csv
import random
from math import sqrt
from operator import itemgetter

def loadDataset(filename,split,trainingSet=[],testSet=[]):
    '''
    :param filename: 数据集文件名，建议绝对路径
    :param split: 区分训练集和数据集的一个浮点数
    :param trainingSet: 训练集
    :param testSet: 测试集
    :return: none
    '''
    with open(filename,'r') as f:
        dataSet=csv.reader(f)
        lines=list(dataSet)
        for line in lines:
            for x in range(4):
                line[x]=float(line[x])
            if random.random()<split:
                trainingSet.append(line)
            else:
                testSet.append(line)

def euclideanDistance(position1,position2,dim):
    '''
    :param position1: 起点坐标
    :param position2: 终点坐标
    :param dim: 维度
    :return: 距离
    '''
    distance=0
    for i in range(dim):
        distance+=(position1[i]-position2[i])**2
    return sqrt(distance)

def getNeighbors(trainSet,test,K):
    '''
    :param trainSet:训练集
    :param test: 待测试的用例
    :param K: 代表选择多少个最近的点
    :return: K个最近的用例
    '''
    distances=[]#tuple的list，保存test到每个train的distance
    dim=len(test)-1#不算lable
    for train in trainSet:
        distances.append((train,euclideanDistance(test,train,dim)))
    distances.sort(key=itemgetter(1))
    neighbors=[]#K个最近的train的列表
    for i in range(K):
        neighbors.append(distances[i][0])
    return neighbors

def getResponse(neighbors):
    '''
    :param neighbors:K个最近的点的列表
    :return: 占多数的类别名，str类型
    '''
    Class={}
    for point in neighbors:
        lable=point[-1]
        if lable in Class:
            Class[lable]+=1
        else:
            Class[lable]=1
    return sorted(Class,key=itemgetter(1),reverse=True)[0]#对字典按键值排序，返回列表。dict也是iterable的

def getAccuracy(testSet,correctSet):
    '''
    :param testSet: 测试集
    :param correctSet: 正确的结果
    :return: 测试集正确的比例
    '''
    correct_count=0
    for i in range(len(testSet)):
        if testSet[i][-1]==correctSet[i]:
            correct_count+=1
    return correct_count/len(testSet)*100.0

if __name__ == '__main__':
    trainSet=[]
    testSet=[]

    split=0.67
    loadDataset(r'E:\pycharm\ML\KNN\iris.data.txt',split,trainSet,testSet)
    print('trainset个数:',len(trainSet))
    print('testset个数:',len(testSet))

    correctSet=[]
    K=3
    for test in testSet:
        neighbors=getNeighbors(trainSet,test,K)
        result=getResponse(neighbors)
        correctSet.append(result)
        print('预测为:',result,'实际为:',test[-1])
    print('精确度为:',getAccuracy(testSet,correctSet),'%')
```

代码执行结果如下：

```
D:\Anaconda\python.exe E:/pycharm/ML/KNN/Implement.py
trainset个数: 108
testset个数: 42
预测为: Iris-setosa 实际为: Iris-setosa
预测为: Iris-setosa 实际为: Iris-setosa
预测为: Iris-setosa 实际为: Iris-setosa
预测为: Iris-setosa 实际为: Iris-setosa
预测为: Iris-setosa 实际为: Iris-setosa
预测为: Iris-setosa 实际为: Iris-setosa
预测为: Iris-setosa 实际为: Iris-setosa
预测为: Iris-setosa 实际为: Iris-setosa
预测为: Iris-setosa 实际为: Iris-setosa
预测为: Iris-setosa 实际为: Iris-setosa
预测为: Iris-setosa 实际为: Iris-setosa
预测为: Iris-setosa 实际为: Iris-setosa
预测为: Iris-setosa 实际为: Iris-setosa
预测为: Iris-setosa 实际为: Iris-setosa
预测为: Iris-setosa 实际为: Iris-setosa
预测为: Iris-setosa 实际为: Iris-setosa
预测为: Iris-setosa 实际为: Iris-setosa
预测为: Iris-versicolor 实际为: Iris-versicolor
预测为: Iris-versicolor 实际为: Iris-versicolor
预测为: Iris-versicolor 实际为: Iris-versicolor
预测为: Iris-versicolor 实际为: Iris-versicolor
预测为: Iris-versicolor 实际为: Iris-versicolor
预测为: Iris-versicolor 实际为: Iris-versicolor
预测为: Iris-versicolor 实际为: Iris-versicolor
预测为: Iris-versicolor 实际为: Iris-versicolor
预测为: Iris-versicolor 实际为: Iris-versicolor
预测为: Iris-versicolor 实际为: Iris-versicolor
预测为: Iris-versicolor 实际为: Iris-versicolor
预测为: Iris-virginica 实际为: Iris-virginica
预测为: Iris-virginica 实际为: Iris-virginica
预测为: Iris-virginica 实际为: Iris-virginica
预测为: Iris-virginica 实际为: Iris-virginica
预测为: Iris-virginica 实际为: Iris-virginica
预测为: Iris-virginica 实际为: Iris-virginica
预测为: Iris-virginica 实际为: Iris-virginica
预测为: Iris-versicolor 实际为: Iris-virginica
预测为: Iris-virginica 实际为: Iris-virginica
预测为: Iris-virginica 实际为: Iris-virginica
预测为: Iris-virginica 实际为: Iris-virginica
预测为: Iris-virginica 实际为: Iris-virginica
预测为: Iris-virginica 实际为: Iris-virginica
预测为: Iris-virginica 实际为: Iris-virginica
精确度为: 97.61904761904762 %

Process finished with exit code 0
```

#### 支持向量机（上）

​	在机器学习中，支持向量机（英语：Support Vector Machine，常简称为SVM，又名支持向量网络）是在分类与回归分析中分析数据的**监督式学习模型**与相关的学习算法。给定一组训练实例，每个训练实例被标记为属于两个类别中的一个或另一个，SVM训练算法创建一个将新的实例分配给两个类别之一的模型，使其成为非概率二元线性分类器。SVM模型是将**实例表示为空间中的点**，这样映射就使得单独类别的实例被尽可能宽的明显的间隔分开。然后，<u>将新的实例映射到同一空间，并基于它们落在间隔的哪一侧来预测所属类别。</u>

​	对于支持向量机来说，**数据点被视为p 维向量**，而我们想知道是否可以**用(p-1) 维超平面来分开这些点**。这就是所谓的线性分类器。可能有许多超平面可以把数据分类。最佳超平面的一个**合理选择是以最大间隔把两个类分开的超平面**。因此，我们要选择能够让到每边最近的数据点的距离最大化的超平面。

这部分主要是[线性SVM](https://en.wikipedia.org/wiki/Support_vector_machine#Linear_SVM)，即线性可分的情况（用一条线可以分开）。

------

显然，算法复杂度仅仅由支持向量的个数决定，而不是由数据的维度决定。另外，SVM训练出来的模型，完全依赖于支持向量，其余的点删除也不影响。如果支持向量比较少，则更容易被泛化。



```python
import numpy as np
import pylab as pl #matplotlib的一个子模块，被单独拿了出来

from sklearn import svm

np.random.seed(0) #固定随机化种子

#生成数据
X=np.r_[np.random.randn(20,2)-[2,2],np.random.randn(20,2)+[2,2]]
#上面这句话的意思是先根据标准正态分布生成20*2的矩阵，然后对每一行减去[2,2]；然后把两个矩阵进行连接。实际上这样一减，点分别左右平移，这样方便分类。
Y=[0]*20+[1]*20

#拟合模型
classifier=svm.SVC(kernel='linear')
classifier.fit(X,Y)#进行模型拟合

#超平面
#w0x+w1y+w3=0转化为y=-w0/w1-w3/w1
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
pl.plot(xx,yy,'k-')#实线
pl.plot(xx,yy_down,'k--')
pl.plot(xx,yy_up,'k--')#虚线

pl.scatter(classifier.support_vectors_[:,0],classifier.support_vectors_[:,1],s=80,facecolors='none')
pl.scatter(X[:,0],X[:,1],c=Y,cmap=pl.cm.Paired)

pl.axis('tight')#为了让所有数据都能在一张图展示
pl.show()
```

如上代码运行结果如下：

![snipaste_20170806_092544](http://on7mhq4kh.bkt.clouddn.com//%E6%9A%91%E6%9C%9F%E5%AE%9E%E4%B9%A02017/snipaste_20170806_092544.png)



相关知识补充：

```python
numpy.r_
```

其是一个类的实例，实现了`__getitem__`方法，可用于连接矩阵。[详细文档](https://docs.scipy.org/doc/numpy/reference/generated/numpy.r_.html?highlight=r_#numpy.r_)

```python
numpy.random.randn(d0, d1, ..., dn)
```

根据标准正态分布（期望0，方差1）产生随机数，返回矩阵，参数指定维度。不传递参数时默认返回一个浮点数。

[SVC类](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)

由于该部分是线性SVM，所以，构造分类器时只需要将`kernel`关键字置为`linear`即可。

其有以下属性：

> support_ : array-like, shape = [n_SV]
> Indices of support vectors.
>
> support_vectors_ : array-like, shape = [n_SV, n_features]
> Support vectors.
>
> n_support_ : array-like, dtype=int32, shape = [n_class]
> Number of support vectors for each class.
>
> coef_ : array, shape = [n_class-1, n_features]
> Weights assigned to the features (coefficients in the primal problem). This is only available in the case of a linear kernel.
>
> intercept_ : array, shape = [n_class * (n_class-1) / 2]
> Constants in decision function.

[散点图](https://matplotlib.org/api/pyplot_api.html?highlight=scatter#matplotlib.pyplot.scatter)：

```python
matplotlib.pyplot.scatter(x, y, s=None, c=None, marker=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, edgecolors=None, hold=None, data=None, **kwargs)
```

x,y均是序列，代表点的坐标。s是点的大小，c上面程序用的是Y，其可以是一个序列，这样点会根据Y中值映射到不同颜色（根据cmap参数）。

#### 支持向量机（下）

下面讨论线性不可分的情况。为此，有人提出将原有限维空间映射到维数高得多的空间中，在该空间中进行分离可能会更容易。但是高维度计算时向量点积可能比较复杂，所以引入核函数，用来取代计算非线性映射函数的内积从而快速得到点积。

------

**SVM可以扩展到多个类的情况**

对于每个类，有一个当前类和其它类的二类分类器（one vs rest）

------

[人脸识别](http://scikit-learn.org/stable/auto_examples/applications/face_recognition.html#sphx-glr-auto-examples-applications-face-recognition-py)的示例：

1. 相关库函数用法总结

   - 数据集获取

     ```python
     lfw_people=fetch_lfw_people(min_faces_per_person=70,resize=0.4)
     ```

     min_faces_per_person : int, optional, default None
     数据集只会保留至少拥有`min_faces_person`张照片的人。

     resize : float, optional, default 0.5
     用于调整每张图片大小的比例。

     其返回一个字典类型的对象，有以下属性：

     > dataset : dict-like object with the following attributes:
     > dataset.data : numpy array of shape (13233, 2914)  二维数组，第一维算是序号，每一行存储图片数据（62*47==2914）
     > Each row corresponds to a ravelled face image of original size 62 x 47 pixels. Changing the slice_ or resize parameters will change the shape of the output.
     >
     > dataset.images : numpy array of shape (13233, 62, 47)  三维数组，第一维算是序号，后两维对应一张图片的长和高
     > Each row is a face image corresponding to one of the 5749 people in the dataset. Changing the slice_ or resize parameters will change the shape of the output.
     >
     > dataset.target : numpy array of shape (13233,)  一维数组，下标是序号，对应的值是相应的label
     > Labels associated to each face image. Those labels range from 0-5748 and correspond to the person IDs.

   - 数据集分割成测试集和训练集

     ```python
     sklearn.model_selection.train_test_split(*arrays, **options)
     ```

     该函数把数组或矩阵随机分成测试子集和训练子集。

     第一个参数是传入的序列，其有一个关键字参数下面会用到：

     > test_size : float, int, or None (default is None)
     > If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples. If None, the value is automatically set to the complement of the train size. If train size is also None, test size is set to 0.25.

     返回就是划分后的四个子集。

   - 降维——Principal component analysis主要组件分析 (PCA)

     ```python
     class sklearn.decomposition.PCA(n_components=None, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=None)
     ```

     该算法**主要利用奇异值分解将数据投影到更低维的空间**。

     主要参数详解：

     > n_components : int, float, None or string
     > Number of components to keep. if n_components is not set all components are kept: n_components == min(n_samples, n_features)
     >
     > ​
     >
     > svd_solver : string {‘auto’, ‘full’, ‘arpack’, ‘randomized’}
     >
     > randomized :run randomized SVD by the method of Halko et al.
     >
     > ​
     >
     > whiten : bool, optional (default False)
     > When True (False by default) the components_ vectors are multiplied by the square root of n_samples and then divided by the singular values to ensure uncorrelated outputs with unit component-wise variances.
     > Whitening will remove some information from the transformed signal (the relative variance scales of the components) but can sometime improve the predictive accuracy of the downstream estimators by making their data respect some hard-wired assumptions.

     主要属性如下：

     > components_ : array, [n_components, n_features]
     > Principal axes in feature space, representing the directions of maximum variance in the data.

     其有几个方法下面会用到：

     ```python
     fit(X, y=None)
     ```

     > 用X拟合
     >
     > Parameters:	
     > X: array-like, shape (n_samples, n_features) :
     > Training data, where n_samples in the number of samples and n_features is the number of features.
     > Returns:	
     > self : object
     > Returns the instance itself.

     ```python
     transform(X, y=None)
     ```

     对X进行降维：

     > Parameters:	
     > X : array-like, shape (n_samples, n_features)
     > New data, where n_samples is the number of samples and n_features is the number of features.
     > Returns:	
     > X_new : array-like, shape (n_samples, n_components)

   - 分类器求解[文档](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)

     ```python
     class sklearn.model_selection.GridSearchCV(estimator, param_grid, scoring=None, fit_params=None, n_jobs=1, iid=True, refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs', error_score='raise',return_train_score=True)
     ```

     其是一个穷举式算法，

     主要参数：

     > estimator : estimator object.
     > This is assumed to implement the scikit-learn estimator interface. Either estimator needs to provide a score function, or scoring must be passed.
     >
     > param_grid : dict or list of dictionaries   
     > Dictionary with parameters names (string) as keys and lists of parameter settings to try as values, or a list of such dictionaries, in which case the grids spanned by each dictionary in the list are explored. This enables searching over any sequence of parameter settings.

     其属性`best_estimator_` 为Estimator that was chosen by the search

     方法：

     ```python
     fit(X, y=None, groups=None)
     ```

     > X : array-like, shape = [n_samples, n_features]
     > Training vector, where n_samples is the number of samples and n_features is the number of features.
     >
     > y : array-like, shape = [n_samples] or [n_samples, n_output], optional
     > Target relative to X for classification or regression; None for unsupervised learning.
     >
     > groups : array-like, with shape (n_samples,), optional
     > Group labels for the samples used while splitting the dataset into train/test set.

     ```python
     predict(*args, **kwargs)
     ```

     > Call predict on the estimator with the best found parameters.
     >
     > Parameters:	
     > X : indexable, length n_samples

   - 分析报告[文档](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)

     ```python
     sklearn.metrics.classification_report(y_true, y_pred, labels=None, target_names=None, sample_weight=None, digits=2)
     ```

     > Parameters:	
     > y_true : 1d array-like, or label indicator array / sparse matrix
     > Ground truth (correct) target values.
     > y_pred : 1d array-like, or label indicator array / sparse matrix
     > Estimated targets as returned by a classifier.
     > labels : array, shape = [n_labels]
     > Optional list of label indices to include in the report.
     > target_names : list of strings
     > Optional display names matching the labels (same order).
     > sample_weight : array-like of shape = [n_samples], optional
     > Sample weights.
     > digits : int
     > Number of digits for formatting output floating point values
     >
     > ​
     >
     > Returns:	
     > report : string
     > Text summary of the precision, recall, F1 score for each class.

   - 混淆矩阵（confusion matrix）[文档](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix)

     矩阵的每一列代表一个类的实例预测，而每一行表示一个实际的类的实例。

     ```python
     sklearn.metrics.confusion_matrix(y_true, y_pred, labels=None, sample_weight=None)
     ```

     > Parameters:	
     > y_true : array, shape = [n_samples]
     > Ground truth (correct) target values.
     > y_pred : array, shape = [n_samples]
     > Estimated targets as returned by a classifier.
     > labels : array, shape = [n_classes], optional
     > List of labels to index the matrix. This may be used to reorder or select a subset of labels. If none is given, those that appear at least once in y_true or y_pred are used in sorted order.
     >
     > ​
     >
     > Returns:	
     > C : array, shape = [n_classes, n_classes] 代表混淆矩阵

     比如下面的例子

     ```python
     >>> from sklearn.metrics import confusion_matrix
     >>> y_true = [2, 0, 2, 2, 0, 1]
     >>> y_pred = [0, 0, 2, 2, 0, 2]
     >>> confusion_matrix(y_true, y_pred)
     array([[2, 0, 0],
            [0, 0, 1],
            [1, 0, 2]])
     ```

     ```
       0 1 2	
     0 2 0 0	
     1 0 0 1
     2 1 0 2
     ```

     把上述矩阵补全，这个矩阵中(0,0)的意思为实际为0预测为0的情况有几种，看上例显然有两种！

     ```python
     >>> y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
     >>> y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
     >>> confusion_matrix(y_true, y_pred, labels=["ant", "bird", "cat"])
     array([[2, 0, 0],
            [0, 0, 1],
            [1, 0, 2]])
     ```

     这里补全矩阵的行列就不再是上面的0 1 2了，而是那三个字符串！



该实验的数据集我已经下载完毕，在我本机路径`C:\Users\19777\scikit_learn_data\lfw_home`下。执行以下代码，运行结果如下

```python
from time import time
import matplotlib.pyplot as plt
import logging

from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import fetch_lfw_people

#为了实时监控运行情况
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')

#获取数据
lfw_people=fetch_lfw_people(min_faces_per_person=70,resize=0.4)

n_samples,h,w=lfw_people.images.shape

X=lfw_people.data
n_features=X.shape[1] #每个实例的属性个数，这里相当于每张图片的像素点个数

Y=lfw_people.target #每个实例对应的label
target_names=lfw_people.target_names #类别的序列（7,），即对应每个类的人的名字
n_classes=target_names.shape[0] #维度(7,)

print('n_samples:',n_samples)
print('n_features:',n_features)
print('n_classes:',n_classes)
print('图片的h:',h)
print('图片的w:',w)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25)#对数据进行拆分，test_size=0.25表示测试集占25%

n_components=150
print('从 %d 个脸中获取 %d 个特征脸(分割后训练集中包含 %d 个脸)'% (X_train.shape[0],n_components,X_train.shape[0]))
to=time()
pca=PCA(svd_solver='randomized',n_components=n_components,whiten=True).fit(X_train)
print('在%0.3fs内完成'%(time()-to))

#print(pca.components_.shape) #(150, 1850)

eigenfaces=pca.components_.reshape((n_components,h,w)) #(150, 50, 37)


print("Projecting the input data on the eigenfaces orthonormal basis")
to=time()
X_train_pca=pca.transform(X_train) #(n_samples, n_components) 可以看到第二维降低了很多
X_test_pca=pca.transform(X_test)
print('在%0.3fs内完成'%(time()-to))


print('根据训练集拟合模型')
to=time()
param_grid={'C':[1e3,5e3,1e4,5e4,1e5],
            'gamma':[0.0001,0.0005,0.001,0.005,0.01,0.1]}
classifier=GridSearchCV(estimator=SVC(kernel='rbf',class_weight='balanced'),param_grid=param_grid)
classifier=classifier.fit(X_train_pca,Y_train)
print('在%0.3fs内完成'%(time()-to))
print(classifier.best_estimator_) #最优的分类器


print('对测试集进行预测')
to=time()
Y_predict=classifier.predict(X_test_pca)
print('在%0.3fs内完成'%(time()-to))

print(classification_report(Y_test,Y_predict,target_names=target_names)) #统计正确情况，以表格的形式
print(confusion_matrix(Y_test,Y_predict,labels=range(n_classes))) #借助混淆矩阵来统计


def plot_gallery(images,titles,h,w,n_row=3,n_col=4):
    '''
    :param images: 图像
    :param titles: 标题
    :param h: 高
    :param w: 宽
    :param n_row:行      *默认只打印3*4个照片*
    :param n_col: 列
    :return: None
    '''
    plt.figure(figsize=(1.8*n_col,2.4*n_row))
    plt.subplots_adjust(bottom=0,left=.01,right=.99,top=.90,hspace=.35)
    for i in range(n_row*n_col):
        plt.subplot(n_row,n_col,i+1)
        plt.imshow(images[i].reshape((h,w)),cmap=plt.cm.gray)
        plt.title(titles[i],size=12)
        plt.xticks(())
        plt.yticks(())

def title(Y_predict,Y_test,target_names,i):
    predict_name=target_names[Y_predict[i]].rsplit(' ',1)[-1]
    real_name=target_names[Y_test[i]].rsplit(' ',1)[-1]
    return 'predict:%s real:%s' % (predict_name,real_name)


prediction_titles=[title(Y_predict,Y_test,target_names,i) for i in range(Y_predict.shape[0])]

plot_gallery(X_test,prediction_titles,h,w)

eigenface_titles=['eigenface %d' % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces,eigenface_titles,h,w)

plt.show()


```

![snipaste_20170806_092940](http://on7mhq4kh.bkt.clouddn.com//%E6%9A%91%E6%9C%9F%E5%AE%9E%E4%B9%A02017/snipaste_20170806_092940.png)

上述最后的就是混淆矩阵了，我已经在上面提到了，对角线越大表明预测越准确。

下图是输出了部分结果，左边是预测值和真实值，右边是提取的特征脸。

![snipaste_20170806_093004](http://on7mhq4kh.bkt.clouddn.com//%E6%9A%91%E6%9C%9F%E5%AE%9E%E4%B9%A02017/snipaste_20170806_093004.png)

#### 神经网络算法

##### 理论基础

1. 神经元

   ![](https://upload.wikimedia.org/wikipedia/commons/9/97/Ncell.png)

   - a1~an为输入向量的各个分量
   - w1~wn为神经元各个突触的权值
   - b为偏置
   - f为传递函数 activation function or transfer function，通常为非线性函数(sigmod函数)。
   - t为神经元输出

   所以有![](https://wikimedia.org/api/rest_v1/media/math/render/svg/ee01fe927d862cbc18097ac30a320331e98f4173)

   其中，W'是W的转置。

   可见，一个神经元的功能是求得输入向量与权向量的内积后，经一个非线性传递函数得到一个标量结果。

2. 常见结构

   一种常见的多层结构的前馈网络（Multilayer Feedforward Network）由三部分组成：
   输入层（Input layer），众多神经元（Neuron）接受大量非线形输入消息。输入的消息称为输入向量。
   输出层（Output layer），消息在神经元链接中传输、分析、权衡，形成输出结果。输出的消息称为输出向量。（通常个数等于类别个数）
   隐藏层（Hidden layer），简称“隐层”，是输入层和输出层之间众多神经元和链接组成的各个层面。隐层可以有多层，习惯上会用一层。隐层的节点（神经元）数目不定，但数目越多神经网络的非线性越显著，从而神经网络的强健性（robustness）（控制系统在一定结构、大小等的参数摄动下，维持某些性能的特性。）更显著。习惯上会选输入节点1.2至1.5倍的节点。

3. 交叉验证之K-fold cross-validation

   K次交叉验证，初始采样分割成K个子样本，一个单独的子样本被保留作为验证模型的数据，其他K-1个样本用来训练。交叉验证重复K次，每个子样本验证一次，平均K次的结果或者使用其它结合方式，最终得到一个单一估测。这个方法的优势在于，同时重复运用随机产生的子样本进行训练和验证，每次的结果验证一次，10次交叉验证是最常用的。

4. 反向传播算法——Backpropagation

   其本质上计算损失函数对权重w的偏导数或梯度。这里的损失函数（cost function or error function ）是一个函数，负责把一个或多个变量映射到实数域，代表与某事件有关的成本。在本算法中，主要负责计算输出和期待输出之间的差异。

   **这个算法一开始对权值和偏置进行随机初始化，然后通过不断计算输出和期望输出之间的差异，根据这个差异来向前更新权值和偏置，最后训练完成后结束。**

   其具体公式见[wiki](https://en.wikipedia.org/wiki/Backpropagation#Assumptions_about_the_loss_function)

5. 手动实现神经网络算法

   - 随机数生成

     ```python
     numpy.random.random(size=None)
     ```

     Return random floats in the half-open interval [0.0, 1.0)

     > Parameters:	
     >
     > size : int or tuple of ints, optional
     >
     > Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. Default is None, in which case a single value is returned.
     >
     > ​
     >
     > Returns:	
     >
     > out : float or ndarray of floats
     >
     > Array of random floats of shape size (unless size=None, in which case a single float is returned). 

        它可以生成[a,b)区间内的随机数，利用`(b - a) * random() + a`

     ```python
      numpy.random.randint(low, high=None, size=None, dtype='l')
     ```

     Return random integers from low (inclusive) to high (exclusive).如果high未赋值，范围就是[0,low)

     > Parameters:	
     > low : int
     > Lowest (signed) integer to be drawn from the distribution (unless high=None, in which case this parameter is one above the highest such integer).
     > high : int, optional
     > If provided, one above the largest (signed) integer to be drawn from the distribution (see above for behavior if high=None).
     > size : int or tuple of ints, optional
     > Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. Default is None, in which case a single value is returned.
     > dtype : dtype, optional
     > Desired dtype of the result. All dtypes are determined by their name, i.e., ‘int64’, ‘int’, etc, so byteorder is not available and a specific precision may have different C types depending on the platform. The default value is ‘np.int’.
     > ​
     >
     > Returns:	
     > out : int or ndarray of ints
     > size-shaped array of random integers from the appropriate distribution, or a single such random int if size not provided.

   - 维度保证[文档](https://docs.scipy.org/doc/numpy/reference/generated/numpy.atleast_2d.html?highlight=atleast_2d#numpy.atleast_2d)

     ```python
     numpy.atleast_2d(*arys)
     ```

     > View inputs as arrays with at least two dimensions.
     >
     > Parameters:	
     >
     > arys1, arys2, ... : array_like
     >
     > One or more array-like sequences. Non-array inputs are converted to arrays. Arrays that already have two or more dimensions are preserved.
     >
     > ​
     >
     > Returns:	
     >
     > res, res2, ... : ndarray
     >
     > An array, or list of arrays, each with a.ndim >= 2. Copies are avoided where possible, and views with two or more dimensions are returned

##### 代码实现

以下是手工实现了简单的神经网络。

```python
import numpy as np


def tanh(X):
    return np.tanh(X)

def tanh_derivation(X):
    return 1.0-np.tanh(X)**2

def logistic(X):
    return 1/(1+np.exp(-X))

def logistic_derivation(X):
    return logistic(X)*(1-logistic(X))


class NeuralNetwork:
    def __init__(self,layers,activation='tanh'):
        '''
        :param layers: list，每个代表每一层的神经元个数
        :param activation: "tanh" or "logistic"
        '''
        if activation=='logistic':
            self.activation=logistic
            self.activation_derivation=logistic_derivation
        elif activation=='tanh':
            self.activation=tanh
            self.activation_derivation=tanh_derivation
        self.weights=[] #每一个元素都是二维矩阵，且其中的值都在[-1/4,1/4)之间
        for i in range(1,len(layers)-1):
            self.weights.append((2 * np.random.random((layers[i - 1] +1, layers[i]+1)) - 1) * 0.25) #前
            self.weights.append((2 * np.random.random((layers[i]+1, layers[i + 1])) - 1) * 0.25) #后


    def fit(self,X,y,learn_rate=0.2,epochs=10000):
        '''
        :param X:训练集
        :param y: 输出class label
        :param learn_rate: 学习速率
        :param epochs: 循环更新次数
        :return:
        '''
        X=np.atleast_2d(X)#类型转化
        tmp=np.ones((X.shape[0],X.shape[1]+1))
        tmp[:,:-1]=X
        X=tmp #以上操作相当于给X增加了一列
        y=np.array(y)
        for k in range(epochs):
            # 每次随机抽取一行进行网络更新
            i=np.random.randint(X.shape[0]) #随机选择第i行
            a=[X[i]]
            for j in range(len(self.weights)): #从前向后传播，a记录每一层的输出
                a.append(self.activation(np.dot(a[j],self.weights[j])))
            error=y[i]-a[-1]
            deltas=[error*self.activation_derivation(a[-1])] #误差

            for j in range(len(a)-2,0,-1): #反向计算
                deltas.append(deltas[-1].dot(self.weights[j].T)*self.activation_derivation(a[j]))
            deltas.reverse()
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learn_rate * layer.T.dot(delta)


    def predict(self, x):
        x = np.array(x)
        temp = np.ones(x.shape[0] + 1)
        temp[0:-1] = x
        a = temp
        # a=x
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a


```

以下是测试代码：

```python
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

```

1. 测试神经网络

   - 引入测试数据集 [文档](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)

     ```python
     sklearn.datasets.load_digits(n_class=10, return_X_y=False)
     ```

     其有10个类，总共1797个用例，每个用例是一个8*8的图片。

   - 标记分类 [文档](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html)

     ```python
     class sklearn.preprocessing.LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)
     ```

     简而言之，该方法把label分成两类，即进行二值化。

     其有一个方法:

     ```python
     fit_transform(X, y=None, **fit_params)
     ```

     > Parameters:	
     > X : numpy array of shape [n_samples, n_features]
     > Training set.
     > y : numpy array of shape [n_samples]
     > Target values.
     >
     > ​
     >
     > Returns:	
     > X_new : numpy array of shape [n_samples, n_features_new]
     > Transformed array.

   - 获取某个维度的极值的索引

     ```python
     numpy.argmax(a, axis=None, out=None)
     ```

     如果`axis`不给定，则把a视作扁平化数组即比较全部元素，返回最大值索引。

     如果指定`axis`，则只对对应的轴进行操作。

     关于`axis`困扰了很久，总结如下：

     **axis=0代表跨行（down)，而axis=1代表跨列（across)！也就是说轴用来为超过一维的数组定义的属性，二维数据拥有两个轴：第0轴沿着行的垂直往下，第1轴沿着列的方向水平延伸。**

   - 测试代码如下

   - 运行结果

     ![](http://on7mhq4kh.bkt.clouddn.com/2017-07-19_110229.png)

     可以看到，第一个是混淆矩阵，其意义已经在前面说明。第二个就是图表化分析结果。第一列精度的意思是（都以第一行为例），预测为0的用用例中，真实值也为0的比例为1。第二列recall的意思相反，是真实为0的用例中，预测为0的比例。

     可见，precison可以由上述混淆矩阵中，**对角线元素/该列累计和**得到。而recall可以由**对角线元素/该行累计和**得到。

### 监督学习——回归

首先回顾一下，回归和分类的区别。

回归(regression) Y变量为**连续数值型**(continuous numerical variable)，如：房价，人数，降雨量

分类(Classification): Y变量为**类别型**(categorical variable)，如：颜色类别，电脑品牌，有无信誉

#### 简单线性回归(Simple Linear Regression)

很多做决定过过程通常是根据两个或者多个变量之间的关系，回归分析(regression analysis)用来建立方程模拟两个或者多个变量之间如何关联

1. 简单线性回归介绍

   - 简单线性回归包含一个自变量(x)和一个因变量(y) 
   - 以上两个变量的关系用一条直线来模拟
   - 如果包含两个以上的自变量，则称作<u>多元回归分析(multiple regression)</u>

2. 最小二乘法

   计算简单线性回归方程时，只需要用到很熟悉的最小二乘法即可。$\hat y=b1*x+b0$

   ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/69853af4b84a7f3597a642a56b9ca9ab5a1c63d0)

   求出b1之后根据$b0=\bar y-\overline x*b1$即可求出b0

#### 多元线性回归

与简单线性回归相对比，有多个自变量。

$\hat y=b_{1}*x_{1}+b_{2}*x_{2}+...+b_{0}$

*需要注意，如果如果自变量中有分类型变量(categorical data) ,比如车型信息，A、B、C，只需要新加入三个x分类，把这三种车型映射到对应的x分量。这样就变成了多了3个x分量的问题。*

1. 机器学习库介绍

   ```python
   class sklearn.linear_model.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
   ```

   其是最小二乘法的封装。[详细文档](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression.predict)

   这里用到了两个属性，`coef_`表示回归方程中系数的序列，`intercept_`是截距。

下面代码用到的数据集是

![snipaste_20170806_094158](http://on7mhq4kh.bkt.clouddn.com//%E6%9A%91%E6%9C%9F%E5%AE%9E%E4%B9%A02017/snipaste_20170806_094158.png)



```python
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


```

![snipaste_20170806_094252](http://on7mhq4kh.bkt.clouddn.com//%E6%9A%91%E6%9C%9F%E5%AE%9E%E4%B9%A02017/snipaste_20170806_094252.png)



#### 非线性回归

##### 梯度下降法

1. 梯度下降法基本概念

   - 步长（Learning rate）：步长决定了在梯度下降迭代的过程中，每一步沿梯度负方向前进的长度。用上面下山的例子，步长就是在当前这一步所在位置沿着最陡峭最易下山的位置走的那一步的长度。

   - 特征（feature）：指的是样本中输入部分，比如样本（x0,y0）,（x1,y1）,则样本特征为x，样本输出为y。

   - 假设函数（hypothesis function）：在监督学习中，为了拟合输入样本，而使用的假设函数，记为hθ(x)。比如对于样本（xi,yi）(i=1,2,...n),可以采用拟合函数如下： hθ(x) = θ0+θ1x。

   - 损失函数（loss function）：为了评估模型拟合的好坏，通常用损失函数来度量拟合的程度。损失函数极小化，意味着拟合程度最好，对应的模型参数即为最优参数。在线性回归中，损失函数通常为样本输出和假设函数的差取平方。比如对于样本（xi,yi）(i=1,2,...n),采用线性回归，损失函数为：

     $J(\theta_0, \theta_1) = \sum\limits_{i=1}^{m}(h_\theta(x_i) - y_i)^2$

     其中$x_i$表示样本特征x的第i个元素，$y_i$表示样本输出y的第i个元素，$h_\theta(x_i)$为假设函数。

2. 批量梯度下降法（Batch Gradient Descent）

   $\theta_i = \theta_i - \alpha\sum\limits_{j=0}^{m}(h_\theta(x_0^{j}, x_1^{j}, ...x_n^{j}) - y_j)x_i^{j}$

   由于我们有m个样本，这里求梯度的时候就用了所有m个样本的梯度数据。

3. 随机梯度下降法（Stochastic Gradient Descent）

   其实和批量梯度下降法原理类似，区别在与求梯度时没有用所有的m个样本的数据，而是仅仅选取一个样本j来求梯度。

$$
\theta_i = \theta_i - \alpha (h_\theta(x_0^{j}, x_1^{j}, ...x_n^{j}) - y_j)x_i^{j}
$$

4. 二者之间对比总结

   批量梯度下降：最小化所有训练样本的损失函数，使得最终求解的是全局的最优解，即求解的参数是使得风险函数最小，但是对于大规模样本问题效率低下。

   随机梯度下降：最小化每条样本的损失函数，虽然不是每次迭代得到的损失函数都向着全局最优方向， 但是大的整体的方向是向全局最优解的，最终的结果往往是在全局最优解附近，适用于大规模训练样本情况。

   **两者的关系可以这样理解：随机梯度下降方法以损失很小的一部分精确度和增加一定数量的迭代次数为代价，换取了总体的优化效率的提升。增加的迭代次数远远小于样本的数量。**

5. 梯度下降法的向量描述

$$
J(θ)=\frac {1}{2}(X_θ−Y)^T(X_θ−Y)
$$

$$
\frac {\partial J(θ)}{\partial \theta}=X^T(X_θ−Y)
$$

$$
θ=θ−αX^T(X_θ−Y)
$$



##### 逻辑回归

1. 基本原理

   Logistic Regression和Linear Regression的原理是相似的，可以简单的描述为这样的过程：

   - 找一个合适的预测函数（Andrew Ng的公开课中称为hypothesis），一般表示为h函数，该函数就是我们需要找的分类函数，它用来预测输入数据的判断结果。这个过程时非常关键的，需要对数据有一定的了解或分析，知道或者猜测预测函数的“大概”形式，比如是线性函数还是非线性函数。
   - 构造一个Cost函数（损失函数），该函数表示预测的输出（h）与训练数据类别（y）之间的偏差，可以是二者之间的差（h-y）或者是其他的形式。综合考虑所有训练数据的“损失”，将Cost求和或者求平均，记为J(θ)函数，表示所有训练数据预测值与实际类别的偏差。
   - 显然，J(θ)函数的值越小表示预测函数越准确（即h函数越准确），所以这一步需要做的是找到J(θ)函数的最小值。找函数的最小值有不同的方法，Logistic Regression实现时有的是梯度下降法（Gradient Descent）。

##### 回归中的相关性

1. 协方差（covariance）

   ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/7331bb9b6e36128d1d9cb735b11b65427929105d)

   ​

   协方差是一个反映两个随机变量相关程度的指标，如果一个变量跟随着另一个变量同时变大或者变小，那么这两个变量的协方差就是正值，反之相反。

   而 **Pearson correlation coefficient**是衡量两个变量的线性相关程度，值取[-1,1]。+1表示完全正相关，0表示不是线性相关，-1表示完全负相关。

   其定义为协方差除以它们标准差之积。

   ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/f76ccfa7c2ed7f5b085115086107bbe25d329cec)

   ​

2. 决定系数

   决定系数（英语：coefficient of determination，记为R2或r2）在统计学中用于度量因变量的变异中可由自变量解释部分所占的比例，以此来判断统计模型的解释力。

   比如，R平方为0.8，则表示回归关系可以解释因变量80%的变异。换句话说，如果我们能控制自变量不变，则因变量的变异程度会减少80%。

   对于简单线性回归而言，**决定系数为样本相关系数的平方**。当加入其他回归自变量后，决定系数相应地变为多重相关系数的平方。

   ​

   假设一数据集包括y1,...,yn共n个观察值，相对应的模型预测值分别为f1,...,fn。定义残差$e_i = y_i − f_i$，平均观察值为：$\bar y=\frac 1 n\sum _{i=1}^ny_i$

   于是得到总平方和（total sum of squares）$SS_{tot}=\sum _{i=1} ^n(y_i-\bar y)^2$，所以回归平方和（regression sum of squares）为$SS_{reg}=\sum _{i=1}^n(f_i-\bar y)^2$。

   另外，残差平方和（sum of squares of residuals）为$SS_{res}=\sum _{i=1} ^n(y_i-f_i)^2=\sum _{i=1}^ne_i^2$

   所以，决定系数为：

   ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/fed29779d54adeccdec58f0894870c680f3d6b5b)

   由于$R^2$会随着自变量个数增加而虚假的增加，所以修正的$R^2$为：![](https://wikimedia.org/api/rest_v1/media/math/render/svg/f6e88c86fcecf4cfb5418760909bbe2d499bd1aa)

   调整后的$R^2$值可以为负，其始终小于等于$R^2$。其中，n是样本数目，p是自变量个数。

##### 代码实现

1. 用到的函数库总结

   ```python
   numpy.polyfit(x, y, deg, rcond=None, full=False, w=None, cov=False)
   ```

   最小二乘法多项式拟合。拟合形式为`p(x) = p[0] * x**deg + ... + p[deg]`，返回最后p的向量形式。

   这里只用到了`deg`代表最高次数。其余见[文档](https://docs.scipy.org/doc/numpy/reference/generated/numpy.polyfit.html?highlight=numpy%20polyfit#numpy.polyfit)

   ```python
   class numpy.poly1d(c_or_r, r=False, variable=None)
   ```

   一个很方便的类，它对一维多项式进行了封装，第一个参数代表了多项式系数。`poly1d([1, 2, 3])`代表$x^2+2x+3$，如果`r=True`，则代表$(x-1)(x-2)(x-3)$。[文档](https://docs.scipy.org/doc/numpy/reference/generated/numpy.poly1d.html?highlight=poly1d#numpy.poly1d)

测试相关性的代码

```python
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
```

梯度下降代码：

```python
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
```

![snipaste_20170806_095008](http://on7mhq4kh.bkt.clouddn.com//%E6%9A%91%E6%9C%9F%E5%AE%9E%E4%B9%A02017/snipaste_20170806_095008.png)

**如上结果第一行是用梯度下降算法求出的结果，第二行是我直接利用矩阵求导，解出了关于$\theta$的表达式！**

证明过程比较繁琐，我就直接把我证明的表达式写下了了：
$$
\theta=(X^TX)^{-1}X^Ty
$$
所以，直接就可以求出来了，这也反面证明了，梯度下降法和真实结果很接近了！当然，这里关系比较简单，所以我可以解出来，如果太复杂就没办法了！

### 非监督学习

非监督学习和监督学习主要区别在于没有类别标记(class label)。

#### K-means算法

##### 基本概念

1. 基本功能

   算法接受参数 k ；然后将事先输入的n个数据对象划分为 k个聚类以便使得所获得的聚类满足：同一聚类中的对象相似度较高；而不同聚类中的对象相似度较小。

2. 算法流程

   - 适当选择c个类的初始中心；
   - 在第k次迭代中，对任意一个样本，求其到c各中心的距离，将该样本归到距离最短的中心所在的类；
   - 利用均值等方法更新该类的中心值；
   - 对于所有的c个聚类中心，如果利用以上两步的迭代法更新后，值保持不变（或者迭代次数达到了预设的阈值），则迭代结束，否则继续迭代。

3. 算法优缺点

   - 优点

     速度快，简单

   - 缺点

     最终结果跟初始点选择相关，容易陷入局部最优，另外需知道k值。

##### 代码实现

```python
import numpy as np
import traceback
import random



def K_means(X,k,max_count):
    '''
    :param X: 训练数据
    :param k: 类别个数
    :param max_count: 最大迭代次数
    :return:
    '''
    row,col=X.shape
    dataSet=np.zeros((row,col+1))
    dataSet[:,:-1]=X #添加最后一列记录类别，用1~k表示
    # Initialize centroids randomly
    # centroids=dataSet[np.random.randint(low=X.shape[0],size=k),:] #选了k个点,这个方法并不好，因为可能会产生两次随机到了同一个点！
    centroids=dataSet[random.sample(range(k),k),:] #该随机函数从第一个参数序列中选择k个元素，保证不会重复选择同一个！
    # centroids = dataSet[0:2,:]
    centroids[:,-1]=np.arange(1,k+1) #最后一行，赋值1~k记录类别

    # Initialize book keeping vars.
    iterations = 1
    oldCentroids = None



    # Run the main k-means algorithm
    while not Is_stop(oldCentroids,centroids,iterations,max_count):
        print('第%d次迭代'%iterations,'*'*50)
        print('中心点:')
        print(centroids)

        oldCentroids=np.copy(centroids) #深拷贝
        iterations+=1

        Updata_label(dataSet,centroids)
        centroids=getCentroids(dataSet,k)

    return  dataSet


def Is_stop(oldCentroids, centroids, iterations, max_count):
    if iterations>max_count:
        return True
    return np.array_equal(oldCentroids,centroids)


def Updata_label(dataSet,centroids):
    '''
    :param dataSet: 数据集
    :param centroids: 中心点数据集
    :return:
    '''
    for line in dataSet:
        line[-1]=getLabelFromClosestCentroid(line,centroids)



def getLabelFromClosestCentroid(dataSetRow, centroids):
    '''
    :param dataSetRow: 数据集中某一行
    :param centroids: 中心点的集合
    :return: 该行对应点的更新后的Label
    '''
    label=centroids[0,-1] #一个数字标记
    minDist = np.linalg.norm(dataSetRow[:-1]- centroids[0, :-1])
    for i in range(1,centroids.shape[0]):
        dis=np.linalg.norm(dataSetRow[:-1]-centroids[i,:-1])
        if dis<minDist:
            minDist=dis
            label=centroids[i,-1]
    return label

def getCentroids(dataSet, k):
    result = np.zeros((k, dataSet.shape[1])) #新建中心点数据集，一开始全为0
    for i in range(1, k + 1):
        oneCluster = dataSet[dataSet[:, -1] == i, :-1] #掩码方式选择行，挺高的！
        result[i - 1, :-1] = np.mean(oneCluster, axis=0) #跨行求平均值，结果是一维向量，每个值表示这一列的平均值。
        result[i - 1, -1] = i
    return result

x1 = np.array([1, 1])
x2 = np.array([2, 1])
x3 = np.array([4, 3])
x4 = np.array([5, 4])
testX = np.vstack((x1, x2, x3, x4))

result = K_means(testX, 2, 10)
print('最终结果',result)
```

#### 层次聚类

##### 基本概念

1. 算法流程

   假设有N个待聚类的样本，对于层次聚类来说，步骤：

   1. （初始化）把每个样本归为一类，计算每两个类之间的距离，也就是样本与样本之间的相似度；
   2. 寻找各个类之间最近的两个类，把他们归为一类（这样类的总数就少了一个）；
   3. 重新计算新生成的这个类与各个旧类之间的相似度；
   4. 重复2和3直到所有样本点都归为一类，结束（或者设置一个阈值，当类之间距离大于这个阈值时终止）

2. 求相似度

   整个聚类过程其实是**建立了一棵树**（两两合并，类似霍夫曼树的构造），在建立的过程中，可以通过在第二步上设置一个阈值，当最近的两个类的距离大于这个阈值，则认为迭代可以终止。另外关键的一步就是第三步，如何判断两个类之间的相似度有不少种方法。这里介绍一下三种：

   - SingleLinkage：又叫做 nearest-neighbor ，就是取两个类中距离最近的两个样本的距离作为这两个集合的距离，也就是说，最近两个样本之间的距离越小，这两个类之间的相似度就越大。容易造成一种叫做 Chaining 的效果，两个 cluster 明明从“大局”上离得比较远，但是由于其中个别的点距离比较近就被合并了，并且这样合并之后 Chaining 效应会进一步扩大，最后会得到比较松散的 cluster 。
   - CompleteLinkage：这个则完全是 Single Linkage 的反面极端，取两个集合中距离最远的两个点的距离作为两个集合的距离。其效果也是刚好相反的，限制非常大，两个 cluster 即使已经很接近了，但是只要有不配合的点存在，就顽固到底，老死不相合并，也是不太好的办法。这两种相似度的定义方法的共同问题就是指考虑了某个有特点的数据，而没有考虑类内数据的整体特点。
   - Average-linkage：这种方法就是把两个集合中的点两两的距离全部放在一起求一个平均值，相对也能得到合适一点的结果。
   - average-linkage的一个变种就是取两两距离的中值，与取均值相比更加能够解除个别偏离样本对结果的干扰。

##### 代码

```python
from numpy.ma import sqrt, array


class cluster_node:
    def __init__(self,vec,left=None,right=None,distance=0.0,id=None,count=1):
        '''
        :param vec: 数据集中一行
        :param left: 左孩子
        :param right: 右孩子
        :param distance: 结点距离
        :param id:
        :param count: 结点个数
        '''
        self.left=left
        self.right=right
        self.vec=vec
        self.id=id
        self.distance=distance
        self.count=count #only used for weighted average

#两个求距离的方法
def L2dist(v1, v2):
    return sqrt(sum((v1 - v2) ** 2))

def L1dist(v1, v2):
    return sum(abs(v1 - v2))


def hcluster(features, distance=L2dist):
    '''
    :param features: 数据集矩阵，每一行代表一个点
    :param distance: 指定求距离的方法
    :return: 根node
    '''
    # cluster the rows of the "features" matrix
    distances = {} #键是点的tuple，值是其距离
    currentclustid = -1

    # clusters are initially just the individual rows
    clust = [cluster_node(array(features[i]), id=i) for i in range(len(features))]

    while len(clust) > 1: #总类别大于1
        lowestpair = (0, 1) #最近的两个点下标的tuple
        closest = distance(clust[0].vec, clust[1].vec) #保存最近的两个点的距离

        # loop through every pair looking for the smallest distance
        for i in range(len(clust)):
            for j in range(i + 1, len(clust)):
                # distances is the cache of distance calculations
                if (clust[i].id, clust[j].id) not in distances:
                    distances[(clust[i].id, clust[j].id)] = distance(clust[i].vec, clust[j].vec)

                d = distances[(clust[i].id, clust[j].id)]

                if d < closest:
                    closest = d
                    lowestpair = (i, j)

        # calculate the average of the two clusters
        mergevec = [(clust[lowestpair[0]].vec[i] + clust[lowestpair[1]].vec[i]) / 2.0 \
                    for i in range(len(clust[0].vec))]

        # create the new cluster
        newcluster = cluster_node(array(mergevec), left=clust[lowestpair[0]],
                                  right=clust[lowestpair[1]],
                                  distance=closest, id=currentclustid)

        # cluster ids that weren't in the original set are negative
        currentclustid -= 1
        del clust[lowestpair[1]]
        del clust[lowestpair[0]]
        clust.append(newcluster)

    return clust[0]

def extract_clusters(clust,dist):
    '''
    :param clust: 根节点
    :param dist: 距离的阈值
    :return: 所有距离小于dist的子树
    '''
    # extract list of sub-tree clusters from hcluster tree with distance<dist
    clusters = {}
    if clust.distance<dist:
        # we have found a cluster subtree
        return [clust]
    else:
        # check the right and left branches
        cl = []
        cr = []
        if clust.left!=None:
            cl = extract_clusters(clust.left,dist=dist)
        if clust.right!=None:
            cr = extract_clusters(clust.right,dist=dist)
        return cl+cr

def get_cluster_elements(clust):
    '''
    :param clust: 根节点
    :return: id的列表
    '''
    # return ids for elements in a cluster sub-tree
    if clust.id>=0:
        # positive id means that this is a leaf
        return [clust.id]
    else:
        # check the right and left branches
        cl = []
        cr = []
        if clust.left!=None:
            cl = get_cluster_elements(clust.left)
        if clust.right!=None:
            cr = get_cluster_elements(clust.right)
        return cl+cr


def printclust(clust, labels=None, n=0):
    # indent to make a hierarchy layout
    for i in range(n): print(' ',end='')
    if clust.id < 0:
        # negative id means that this is branch
        print('-')
    else:
        # positive id means that this is an endpoint
        if labels == None:
            print(clust.id)
        else:
            print(labels[clust.id])

    # now print the right and left branches
    if clust.left != None:
        printclust(clust.left, labels=labels, n=n + 1)
    if clust.right != None:
        printclust(clust.right, labels=labels, n=n + 1)


def getheight(clust):
    # Is this an endpoint? Then the height is just 1
    if clust.left == None and clust.right == None:
        return 1

    # Otherwise the height is the same of the heights of
    # each branch
    return getheight(clust.left) + getheight(clust.right)


def getdepth(clust):
    # The distance of an endpoint is 0.0
    if clust.left == None and clust.right == None:
        return 0

    # The distance of a branch is the greater of its two sides
    # plus its own distance
    return max(getdepth(clust.left), getdepth(clust.right)) + clust.distance

```

## 进阶篇之神经网络总结

暑期实习之神经网络总结，包括反向传播、梯度下降以及卷积神经网络等。目的是识别图片中的手写数字。

说明：本文是学习神经网络的笔记，[英文原文](http://neuralnetworksanddeeplearning.com/)，本文内容均由我从原书中提炼而来，并对其中每个结论进行了数学证明（过于高深的问题未解决，只限于高等数学范畴）。另外，**原版代码基于python2.x，我对其进行了部分修改，保证在theano0.9和python3.6下运行无误（均是当前最新版本）**，并对部分加了注释（只是我一开始不理解的地方）。

作者：马源@prime		

实验代码[git]()		

### 环境搭建

我的基本环境如下：

Anacoda4.4 (python3.6)

Theano 0.9

Cuda 8.0（后面卷积神经网络cpu计算太慢，GPU对浮点数有优化，加速杠杠的！核弹厂比较是核弹厂。非核弹厂的卡就不行了）

VS2010（主要是cuda需要c++环境，安装时也只需要安装c++即可！）

我的安装顺序基本就是这样。硬件上是笔记本win10 i7 4710hq+GTX860M+16G RAM，跑起来还阔以（认真脸）。

1. Anacoda

   不多说了，一个python的发行版，科学计算无人不知无人不晓。内建conda的管理系统可以管理包，也可以创建环境。好用到爆炸。

2. Theano

   直接pip install theano即可安装！这里贴出0.9的[文档](http://deeplearning.net/software/theano/install_windows.html)，内含丰富的安装教程。虽然现在标记<3.6，我的3.6运行没问题！不放心的可以用conda创建一个低版本环境使用。

   然后用`conda install theano pygpu`安装必要的包，第一个就是c++环境，不装的话运行程序会用python的实现而不是c++的加速版，python实现你懂的。第二个是和GPU相关的。

3. cuda8.0

   直接英伟达官网下载即可。

4. vs2010

   只需要安装c++即可。后面运行的时候可能会提示`<inttypes.h>`头文件找不到，这时候去参考[博客](http://blog.csdn.net/acheld/article/details/50989438)

经过以上过程之后，还要在用户根目录(就是打开命令行的显示的目录)下创建配置文件，我的配置文件如下：

```
[lib]
cnmem = 0.8 #不设置会提示CNMeM is disable
[blas]
ldflags = 
[gcc]
cxxflags = -ID:\Anaconda\MinGW\include  
[nvcc]
flags=-LD:\Anaconda\libs
compiler_bindir=D:\VS2010\VC\bin
[global]
device=gpu
floatX=float32
```

然后还会提示`CuDNN not available`，这个其实不管也没事，但是强迫症得治。下载cudnn5.1然后将下载来的文件解压，解压出cuda文件夹，里面包含3个文件夹。将设三个文件夹替换掉系统里面的对应文件，进行覆盖替换即可。我的是自定义安装的，就是development文件夹里面。

由此，卷积网络大大加速，哈皮的时刻~~

------

补充一下theano的介绍，其是一个科学计算的库，可以让python拥有与手写的c几乎相同的效率！同时，它可利用GPU进行加速，从而胜过C+CPU！theano把计算机代数系统( computer algebra system (CAS))与优化编译器相结合，因此，对于大多数数学计算操作，可以生成最优的c代码。

### 神经网络概述

#### 感知器

感知器（英语：Perceptron）是Frank Rosenblatt在1957年就职于Cornell航空实验室（Cornell Aeronautical Laboratory）时所发明的一种人工神经网络。它可以被视为一种最简单形式的**前馈神经网络**，是一种**二元线性分类器**。

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/00a41aa6575886baef4193d943fa609b49534272)

感知器使用特征向量来表示的前馈神经网络，把矩阵上的输入x（实数值向量）映射到输出值 f(x)上（一个二元的值）。

![w](https://wikimedia.org/api/rest_v1/media/math/render/svg/88b1e0c8e1be5ebe69d18a8010676fa42d7961e6)是实数的表式权重的向量，![w\cdot x](https://wikimedia.org/api/rest_v1/media/math/render/svg/69b9832ae727dd93d743ed1daf1f7940ebc16f43)是点积。![b](https://wikimedia.org/api/rest_v1/media/math/render/svg/f11423fbb2e967f986e36804a8ae4271734917c3)是偏置，一个不依赖于任何输入值的常数。偏置可以认为是激励函数的偏移量，或者给神经元一个基础活跃等级。

其基本结构如下图

![](https://upload.wikimedia.org/wikipedia/commons/9/97/Ncell.png)

#### S型神经元

其基本结构和上述感知器类似，只不过，输出不再是0和1，而是$\sigma (\vec w \vec x +b)$。此处的$\sigma$被称为S型函数(sigmoid function)。其表达式为：
$$
\sigma (z)=\frac {1} {1+e^{-z}}
$$
该函数图像如下：

![](https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/480px-Logistic-curve.svg.png)

其又被称作**逻辑函数**。

其实，S型神经元可以理解为一个“平滑”的感知器，其平滑意为着权重和偏置的微小改变，会从神经元输出产生一个微小的变化。

#### 神经网络的基本架构

网络中最左边的输入层，其中的神经元被称作**输入神经元**。最右边的即输出层包含**输出神经元。**中间层被称作隐藏层。

本文讨论的都是以上一层的输出作为下一层的输入，这种网络被称作**前馈神经网络**，即意味着网络中是没有回路的——信息总是向前传播。当然，反馈回路也是可行的，这种网络叫做**递归神经网络**。但是后者的学习算法目前不够强大，虽然更接近大脑的工作方式，毕竟其思想就是具有休眠前会在一段有限时间内保持激活状态的神经元，可以刺激其它神经元，随后被激活并同样保持一段有限的时间。随着时间的推移，得到一个级联的神经元激活系统。

#### 使用梯度下降算法进行学习

定义一个代价函数如下：
$$
\begin{eqnarray} C(w,b) \equiv \frac{1}{2n} \sum_x \| y(x) - a\|^2 \end{eqnarray}
$$
这里w是权重，b是偏置，n是训练输入数据的个数，$\vec a$是表示当输入为$\vec x$时输出的向量，而y(x)表示真实的结果。

显而易见，我们训练的目的就是**最小化代价函数**。

首先，直观的，梯度下降可以理解为下山，肯定最陡峭的路下降速度最快。在多元函数微分学中，梯度方向是函数增加最快的方向，而其反方向就是函数减小最快的方向！
$$
\nabla C=(\frac {\partial C}{\partial w},\frac {\partial C}{\partial b})^T
$$
由此，得到了更新公式如下：
$$
w_k:=w_k-\eta \frac {\partial C}{\partial w_k}
$$

$$
b_l:=b_l-\eta \frac{\partial C}{\partial b_l}
$$

但是，这里引出了一个问题，注意代价函数C的形式：
$$
C=\frac {1}{n} \sum _xC_x
$$
也就是说，它是遍及每个训练样本代价$C_x=\frac {\|y(x)-a\|^2}{2}$的平均值。所以为了求解C的梯度，需要为每个训练输入x单独计算$\nabla C_x$。然后求平均值$\nabla C=\frac {1}{n}\sum_x \nabla C_x$。这样最明显的一个问题就是如果训练输入数量过大，就会使得学习非常缓慢！

------

由此，引入随机梯度下降算法，其思想就是**通过随机选取小量训练输入样本来计算$\nabla C_x$，进而估算$\nabla C$。**

更准确的说，随机梯度下降通过**随机选取**小量的m个训练输入来工作。将这些训练输入标记为$X_1,X_2,X_3...X_m$，并把它们称作一个小批量数据(mini-batch)。

所以有以下估计：
$$
\frac {\sum_{j=1}^m\nabla C_{x_j}}{m} \approx \frac {\sum_x \nabla C_x}{n}=\nabla C
$$
由此，更新的公式改为：
$$
w_k:=w_k-\frac {\eta}{m}\sum_j \frac {\partial C_{X_j}}{\partial w_k}
$$

$$
b_l:=b_l-\frac {\eta}{m}\sum_j \frac {\partial C_{X_j}}{\partial b_l}
$$

这里的求和是针对每个小批量数据的所有训练样本$X_j$进行的。然后再去挑选另一个随机的小批量数据，知道用完了所有的训练输入，这称作一次迭代期(epoch)。

特别说明一下，$\eta$这里是学习率的意思，显然越大表示每一步越大。

### 反向传播算法如何工作

首先，先定义一些规则，用$w^l_{jk}$表示从$l-1$层的第k个神经元到第$l$层的第$j$个神经元的权重。

同理，用$b^l_j$表示在第$l$层第j个神经元的偏置，$a^l_j$表示第$l$层第j个神经元的激活值($\sigma$也被称作激活函数)。

所以，有以下表达式：
$$
a^l_j=\sigma (\sum_k w^l_{jk}a^{l-1}_{k}+b^l_j)
$$
其中求和是在第$l-1$层的所有k个神经元上进行的。

于是，为每一层定义一个矩阵$W^l$,其第j行k列的元素就是$w^l_{jk}$。为每一层定义向量$B^l$，每个元素就是$b^l_j$。激活向量$A^l$同理。

这样，上述表达式的向量表示如下：
$$
a^l=\sigma (w^la^{l-1}+b^l)
$$
其中，上面的都是向量或矩阵。$\sigma$的参数是一个k*1的向量，记为$z^l$，其表达式：
$$
z^l_j=\sum ^k_{j=1}w^l_{jk}a^{l-1}_k+b^l_j
$$
即第$l$层第j个神经元的激活函数的带权输入。

------

此时，二次代价函数写作
$$
\begin{eqnarray} C(w,b) \equiv \frac{1}{2n} \sum_x \| y(x) - a^L\|^2 \end{eqnarray}
$$
其中，n是训练样本的总数，求和遍历了每个训练样本x；$y(x)$是对应的目标输出；L表示网络的层数；$a^L$是当输入为$\vec x$时网络输出的激活值向量。

显然，对于单个训练样本，有：
$$
C_x=\frac {1}{2}\|y-a^L\|^2
$$

### Hadamard乘积$\odot$

没什么好说的，就是按元素乘即可。区别于矩阵乘法O(∩_∩)O哈！

### 反向传播的四个方程

首先，先上这四个方程。待会我再证明它们！

定义$\delta ^l_j=\frac {\partial C}{\partial z^l_j}$表示输出层的误差。

四个方程如下：
$$
输出层误差\\\delta^l_j=\frac {\partial C}{\partial a^l_j}\sigma \prime(z^l_j) \tag{1}
$$

$$
使用下一层的误差来计算当前层的误差\\
\delta^l=((w^{l+1})^T\delta^{l+1})\odot\sigma \prime(z^l) \tag{2}
$$

$$
代价函数关于网络中任意偏置的改变率\\\frac {\partial C}{\partial b^l_j}=\delta ^l_j \tag{3}
$$

$$
代价函数关于任何一个权重的改变率\\\frac {\partial C}{w^l_{jk}}=a^{j-1}_k\delta ^l_j \tag{4}
$$

特别说明一下，(2)式给的是向量形式，特别的，(1)的向量形式为:
$$
\delta ^l=(a^l-y)\odot\sigma \prime(z^l)
$$
这四个方程就是反向传播的核心了。

### 证明上面的四个方程

原则就是一个，求导的链式法则！

1. 证明(1)式
   $$
   \begin{align*}
   &\because \delta^l_j=\frac {\partial C}{\partial z^l_j} \\
   &又\because a^l_j=\sigma(z^l_j)\\
   &\therefore \delta^l_j=\frac {\partial C}{a^l_j}\frac {\partial a^l_j}{z^l_j}=\frac {\partial C}{a^l_j}\sigma \prime(z^l_j)\\
   \end{align*}
   $$

2. 证明(2)式
   $$
   \begin {align*}
   &\because \delta ^l_j=\frac {\partial C}{\partial z^l_j}=\sum _k \frac {\partial C}{\partial k^{l+1}}\frac {z^{l+1}_k}{z^l_j}=\sum_k \frac {\partial z^{l+1}_k}{\partial z^l_j}\delta ^{l+1}_k\\&其中k是对下一层的神经元求和\\
   \\
   &又 \because z^{l+1}_k=\sum_jw^{l+1}_{kj}a^l_j+b^{l+1}_k=\sum _jw^{l+1}_{kj}\sigma(z^l_j)+b^{l+1}_k\\&其中j是对l层的神经元求和\\
   \\
   &\therefore \frac {\partial z^{l+1}_k}{z^l_j}=w^{l+1}_{kj}\sigma \prime(z^l_j)\\&求导的无关项都被消掉了\\
   \\
   &\therefore \delta^l_j=\sum_kw^{l+1}_{kj}\sigma \prime(z^l_j) \\&此处的w是矩阵
   \end{align*}
   $$










1. 证明(3)式
   $$
   \begin {align}\frac {\partial C}{b^l_j}&=\frac {\partial C}{\partial z^l_j} \frac {z^l_j}{b^l_j}\\&=\delta^l_j*1\\&=\delta^l_j
   \end{align}
   $$

2. 证明(4)式
   $$
   \because z^l_j=\sum^k_{j=1}w^l_{jk}a^{l-1}_k+b^l_j\\
   \begin{align*}
   \therefore \frac{\partial C}{\partial w^l_{jk}}&=\frac {\partial C}{\partial z^l_j}\frac {\partial z^l_j}{w^l_{jk}}\\&=\delta^l_j\frac {\partial z^l_j}{w^l_{jk}}\\&=a^{l-1}_k\delta^l_j
   \end{align*}
   $$










### 反向传播

该算法叙述如下：

- 输入$\vec x$:为输入层设置对应的激活值$a^1$
- 前向传播：为每个$l=2,3,4\cdots L$计算相应的$z^l=w^la^{l-1}+b^l$和$a^l=\sigma (z^l)$
- 输出层误差$\delta^L$:计算向量$\delta^L=\nabla_aC \odot\sigma\prime(z^L) $
- 反向传播误差：对每个$l=L-1,L-2\cdots2$，计算$\delta^l=((w^{l+1})^T\delta^{l+1})\odot\sigma \prime(z^l)$
- 输出：代价函数的梯度由$\frac {\partial C}{\partial w^l_{jk}}=a^{l-1}_k\delta^l_j和\frac {\partial C}{\partial b^l_j}=\delta^l_j$

当给定一个大小为m的小批量数据，对其中的每个样本，计算出$\delta^{x,l}$，对每个$l=L-1,L-2,\cdots,2$，根据$w^l:=w^l-\frac {\eta}{m}\sum_x\delta^{x,l}(a^{x,l-1})^T$和$b^l:=b^l-\frac {\eta}{m}\sum_x\delta^{x,l}$更新权重和偏置。

### 改进神经网络之交叉熵

#### 交叉熵代价函数

通常人在已经知道犯错误的情况下会加速学习修正错误，但是上面的神经网络在明显出错的时候学习速率反而不高！这就引发了问题。

也就是说，偏导数$\frac {\partial C}{\partial w},\frac {\partial C}{\partial b}$过小。

由此，我们先考虑一个神经元，其结构图如下：

![](https://upload.wikimedia.org/wikipedia/commons/9/97/Ncell.png)

代价函数$C=\frac {(y-a)^2}{2}$。$z=wx+b$
$$
\therefore \frac {\partial C}{\partial w}=(a-y)\sigma \prime(z)x \qquad \frac {\partial C}{\partial b}=(a-y)\sigma \prime(z)
$$
看一下sigmod函数图像，

![](https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/480px-Logistic-curve.svg.png)

显然，当z接近0或者1的时候，导数值明显比较小。

所以，引入了交叉熵代价函数代替二次代价函数。
$$
C=-\frac {1}{n}\sum_x[y\ln a+(1-y)\ln (1-a)]
$$
n是训练样本总数，x是某个具体的样本。

粗看这个函数，显然满足基本代价函数的要求：

1. C非负
2. 当$y=0且a\approx0$时，$C\approx0$
3. 当$y=1且a\approx1时，C\approx0$

代入$a=\sigma(z)$到上述表达式中，根据求导的链式法则，有
$$
\frac {\partial C} {\partial {w_j}}=\frac 1 n\sum_x\frac {\sigma \prime(z)x_j}{\sigma(z)(1-\sigma(z))}(\sigma(z)-y)\\
又\because \sigma(z)=\frac 1{1+e^{-z}},\sigma \prime(z)=\sigma(z)(1-\sigma(z))\\
\therefore \frac {\partial C}{\partial w_j}=\frac 1n\sum_xx_j(\sigma(z)-y)\\同理有\frac {\partial C}{\partial b_j}=\frac 1 n\sum_x(\sigma(z)-y)
$$
显然学习速率受误差控制，误差越小学习速率约小，否则学习越快！
推广到神经网络，$\vec y=y_1,y_2,\cdots,y_n$是输出神经元上的目标值，而$a^L_1,a^L_2,\cdots,a^L_n$是实际的输出。所以有：
$$
C=-\frac 1 n\sum_x\sum_j[y_j\ln a^L_j+(1-y_j)\ln (1-a^L_j)]
$$
这里的j是对输出层所有的神经元求和。

#### 交叉熵代价函数的推导

考虑之前的$\frac {\partial C}{\partial w}=(a-y)\sigma \prime(z)x$，自然想，为何不能选择一个不包含$\sigma \prime(z)$的代价函数呢？

由此，有了以下假定：
$$
\frac {\partial C}{\partial w_j}=x_j(a-y) \\
\frac {\partial C}{\partial b}=(a-y)
$$
那么，剩下的任务就是找到一个C满足上述表达式了。
$$
\begin{align}
\because \frac {\partial C}{\partial b}&=\frac {\partial C}{\partial a}\frac {\partial a}{\partial b}\\&=\frac {\partial C}{\partial a}\sigma \prime(z)\\
\end{align}
\\
又\because \sigma \prime(z)=\sigma(z)(1-\sigma(z))\\
\therefore \frac {\partial C}{\partial b}=\frac {\partial C}{\partial a}a(1-a)
$$
对比上述第二个表达式，显然有：
$$
\frac {(a-y)}{a(1-a)}=\frac {\partial C}{\partial a}
$$
不定积分之：
$$
C=-[y\ln a+(1-y)\ln (1-a)]+常数
$$
此时只是一个单独的样本$\vec x$对代价函数的贡献，对所有样本取平均后:
$$
C=-\frac 1n\sum_x[y\ln a+(1-y)\ln (1-a)]+常数
$$
此处的常数是上述的常数平均后的。

#### 交叉熵的含义

粗略的说，交叉熵是“不确定性”的一种度量，其衡量我们学习到y的正确值平均起来的不确定性。

### 改进神经网络之柔性最大值

引入如下定义：
$$
a^l_j=\frac {e^{z^l_j}}{\sum_ke^{z^l_k}}\qquad k是该层神经元的个数
$$
一个显而易见的 特点是，当某个$z_j$增大，其余的z就会减小，且$\sum_ja^l_j=1$。

所以，可以看做一个概率分布，即$a^l_j$解释为网络估计正确数字分类为j的概率。

由此，$\sigma$变了，代价函数也更改为：
$$
C=-\ln a^L_y
$$
其不会出现学习缓慢的情况，因为其两个偏导数如下：
$$
\frac {\partial C}{\partial b^L_j}=a^L_j-y_j
$$

$$
\frac {\partial C}{\partial w^L_{jk}}=a^{L-1}_k(a^L_j-y_j)
$$

PS：柔性最大值和对数似然函数**更适合于那些需要将输出激活值解释成概率的场景**。

### 改进神经网络之规范化

#### 过度拟合

过度拟合直白说就是网络单纯记忆训练集合，而没有对数字本质进行理解泛化到测试数据集上。检测过度拟合最明显的方法是跟踪测试数据集上的准确度随训练变化的情况。一般来说，最好的防止过度拟合的手段就是增加训练数据集规模。

#### L2规范化

又名权重衰减(weight decay)，其想法是增加一个额外的项到代价函数上，这个项就叫规范化项。

对交叉熵函数规范化如下，对原来的二次代价函数同样可以规范化。
$$
\begin{eqnarray} C = \frac{1}{2n} \sum_x \|y-a^L\|^2 +
  \frac{\lambda}{2n} \sum_w w^2.
\tag{1}\end{eqnarray}
$$
其中，$\lambda$就是规范化参数，n是训练集合的大小。
$$
\begin{eqnarray}  C = C_0 + \frac{\lambda}{2n}
\sum_w w^2,
\tag{2}\end{eqnarray}
$$
规范化可以看做寻找最小化代价函数和小权重的折中，这两部分的相对重要程度就取决于$\lambda$了，其越小越倾向于最小化原始代价函数；其越大越倾向于小的权重。

#### 如何将梯度下降应用于L2规范化的网络

对上述(2)式求导得，
$$
\frac{\partial C}{\partial w}  =  \frac{\partial C_0}{\partial w} + 
\frac{\lambda}{n} w \tag{3}
$$

$$
\frac{\partial C}{\partial b}  =  \frac{\partial C_0}{\partial b} \tag{4}
$$

所以偏置的学习规则不变：
$$
\begin{eqnarray}
b & \rightarrow & b -\eta \frac{\partial C_0}{\partial b}.
\end{eqnarray}
$$
而权重的学习规则就是加上一个项：
$$
\begin{eqnarray} 
  w & \rightarrow & w-\eta \frac{\partial C_0}{\partial
    w}-\frac{\eta \lambda}{n} w \\ 
  & = & \left(1-\frac{\eta \lambda}{n}\right) w -\eta \frac{\partial
    C_0}{\partial w}. 
\end{eqnarray}
$$
看上式，这也是权重衰减名字由来，因为相比以前的学习规则，权重更小了。

同理，梯度下降的学习规则如下：
$$
\begin{eqnarray} 
  w \rightarrow \left(1-\frac{\eta \lambda}{n}\right) w -\frac{\eta}{m}
  \sum_x \frac{\partial C_x}{\partial w}, 
\end{eqnarray}
$$

$$
\begin{eqnarray}
  b \rightarrow b - \frac{\eta}{m} \sum_x \frac{\partial C_x}{\partial b},
\end{eqnarray}
$$

PS：规范化没有限制偏置，因为网络对偏置不是很敏感。通常不对偏置进行规范化。

#### 其它规范化的手段

##### L1规范化

$$
\begin{eqnarray}  C = C_0 + \frac{\lambda}{n} \sum_w |w|.
\end{eqnarray}
$$

直观和L1规范化类似，都是惩罚大的权重，倾向于小权重。但是，下手轻重不同！
$$
\begin{eqnarray}  \frac{\partial C}{\partial
    w} = \frac{\partial C_0}{\partial w} + \frac{\lambda}{n} \, {\rm
    sgn}(w),
\end{eqnarray}
$$

$$
\begin{eqnarray}  w \rightarrow w' =
  w-\frac{\eta \lambda}{n} \mbox{sgn}(w) - \eta \frac{\partial
    C_0}{\partial w},
\end{eqnarray}
$$

##### 弃权(Dropout)

就是每次训练时随机删除一部分隐藏神经元。当我们弃权掉不同的神经元集合时，有点像在训练不同的网络。所以这个过程就如同大量不同网络效果平均那样。

##### 人为扩充训练集

通过一些算法，扩大训练集。

### 改进神经网络之权重初始化

之前的方式是根据标准正态分布来随机初始化权重，这种情况下会使得$z$的图像很宽，这样$\sigma(z)$的取值就容易达到0或1，看到之前的S函数图像可知，此时学习缓慢。

特别说明，**之前的交叉熵函数只是针对输出层，对隐藏层神经元的饱和是没有用的！**

所以，这里使用期望为0，标准差为$\frac 1{\sqrt n}$的正态分布来初始化权重，这样$z$的图像就倾向于集中在中间。至于偏置依旧影响不大，沿用上面的初始化方式即可。

### 深层神经网络很难训练

当训练深层神经网络时，会发现后面的神经元的学习速度快于前一层，或者相反，呈现一种波动性。



考虑如下超简单的神经网络，

![](http://neuralnetworksanddeeplearning.com/images/tikz37.png)

所以，
$$
z_j = w_{j} a_{j-1}+b_j
$$

$$
a_j=\sigma(z_j)
$$

由此，根据链式法则，
$$
\begin{eqnarray}
\frac{\partial C}{\partial b_1} = \sigma'(z_1) \, w_2 \sigma'(z_2) \,
 w_3 \sigma'(z_3) \, w_4 \sigma'(z_4) \, \frac{\partial C}{\partial a_4}.
\end{eqnarray}
$$
对于S函数，其导数在0点取最大值0.25，若使用标准方法初始化网络权重，那么一般$w_j\sigma \prime(z_j)<1$，所以越往后乘的项数越多，前面的导数越小。由此，就是“消失的梯度”。反之，如果权值给的很大，那么就相反了。

### 卷积神经网络

卷积神经网络(Convolutional Neural Network, CNN)是一种前馈神经网络，它的人工神经元可以响应一部分覆盖范围内的周围单元，对于大型图像处理有出色表现。传统的神经网络都是采用全连接的方式，即输入层到隐藏层的神经元都是全部连接的，这样做将导致参数量巨大，使得网络训练耗时甚至难以训练。

![](http://i.imgur.com/PHbta3D.jpg)



**局部感受野**(local receptive fields)：如上图，左图是全连接，隐藏层每个神经元都和输入层有连接，这参数多的难以训练！右图就是局部连接，这里局部感受野就是和每个隐藏层神经元相连的那部分区域。

**共享权重和偏置**：对于隐藏层的每个神经元，其权重矩阵和偏置是共享的，也就是说每个神经元的权重矩阵和偏置都是相同的。实际上，这意味着只提取了图像的一种特征。共享权重和偏置常被称作卷积核或滤波器。如果要多提取出一些特征，可以增加多个卷积核，不同的卷积核能够得到图像的不同映射下的特征，称之为**Feature Map即特征映射**。

**卷积层**：上面介绍共享权重和偏置的时候，用了隐藏层这个词。其实准确说应该是卷积层。

**混合层**：混合层通常紧接着卷积层，其要做的就是简化卷积层的信息。一种常见的是最大值混合(max-pooling)，它从卷积层的输入区域中选择一个最大的激活值输出。如下图

![](http://neuralnetworksanddeeplearning.com/images/tikz47.png)





可以把最大值混合理解为一种网络询问，是否有一个给定的特征在图像的某块区域中出现，然后丢掉具体的位置信息。这样进一步减少后面的层的参数。

另一种常用的技术是**L2混合(L2 pooling)**。这里取区域中激活值的平方和的平方根。其是一种凝聚从卷积层输出信息的方式。

------

关于卷积，可以直观解释如下：

先上数学公式，二维离散卷积
$$
f[x,y] * g[x,y] = \sum_{n_1=-\infty}^\infty \sum_{n_2=-\infty}^\infty f[n_1, n_2] \cdot g[x-n_1, y-n_2]
$$

$$
\text{这是一个 3x3 的均值滤波核，也就是卷积核：} \begin{bmatrix}     1/9 & 1/9 & 1/9 \\     1/9 & 1/9 & 1/9 \\     1/9 & 1/9 & 1/9 \\ \end{bmatrix} \\ \text{这是被卷积图像，这里简化为一个二维 5x5 矩阵：} \begin{bmatrix}     3 & 3 & 3 & 3 & 3 \\     4 & 4 & 4 & 4 & 4 \\     5 & 5 & 5 & 5 & 5 \\     6 & 6 & 6 & 6 & 6 \\     7 & 7 & 7 & 7 & 7 \\ \end{bmatrix} \\
$$

当卷积核运动到图像右下角处（卷积中心和图像对应图像第 4 行第 4 列）时，它和图像卷积的结果如下图所示：

![](http://mengqi92.github.io/2015/10/06/convolution/2d-convolution.png)



可以看出，二维卷积在图像中的效果就是：对图像的每个像素的邻域（邻域大小就是核的大小）加权求和得到该像素点的输出值。滤波器核在这里是作为一个“权重表”来使用的。

#### 针对mnist问题

在引入卷积神经网络之后，我们的网络架构是输入层+卷积层+混合层+全连接层+输出层。

对于此处的全连接层可以理解为在一种更抽象的层次上学习，从整个图像中整合信息。

另外，还引入了线性修正单元，即不再使用S激活函数，而是使用$f(z)=max(0,z)$。此时正确率达到了最高。

------

以上的大背景都是mnist问题。

### 受限玻尔兹曼机（Restricted Boltzmann Machine，RBM）

![](https://deeplearning4j.org/img/multiple_inputs_RBM.png)

如上图是一个简单的RBM网络，**层内无连接，层间全连接，显然RBM对应的图是一个二分图**（层的内部不存在通信－这就是受限玻尔兹曼机被称为*受限*的原因），其第一个输入层又名可见层，第二个是隐藏层。

如果这两个层属于一个深度神经网络，那么第一隐藏层的输出会成为第二隐藏层的输入，随后再通过任意数量的隐藏层，直至到达最终的分类层。

![](https://deeplearning4j.org/img/multiple_hidden_layers_RBM.png)

#### 重构

我们将重点关注受限玻尔兹曼机如何在无监督情况下学习重构数据（无监督指测试数据集没有作为实际基准的标签），在可见层和第一隐藏层之间进行多次正向和反向传递，而无需加大网络的深度。

在重构阶段，第一隐藏层的激活值成为反向传递中的输入。这些输入值与同样的权重相乘，每两个相连的节点之间各有一个权重，就像正向传递中输入x的加权运算一样。这些乘积的和再与每个可见层的偏置相加，所得结果就是重构值，亦即原始输入的近似值。这一过程可以用下图来表示：

![](https://deeplearning4j.org/img/reconstruction_RBM.png)

由于RBM权重初始值是随机决定的，重构值与原始输入之间的差别通常很大。可以将r值与输入值之差视为**重构误差**，此误差值随后经由反向传播来**修**

**RBM的权重**，如此不断反复，**直至误差达到最小**。

由此可见，RBM在正向传递中使用输入值来预测节点的激活值，亦即**输入为加权的x时输出a的概率**：`p(a|x; w)`。

但在反向传递时，激活值成为输入，而输出的是对于原始数据的重构值，或者说猜测值。此时RBM则是在尝试估计激活值为a时输入为x的概率，激活

的加权系数与正向传递中的权重相同。 第二个阶段可以表示为`p(x|a; w)`。

上述两种预测值相结合，可以得到输入 *x* 和激活值 *a* 的**联合概率分布**，即`p(x, a)`。

------

重构与回归、分类运算不同。回归运算根据许多输入值估测一个连续值，分类运算是猜测应当为一个特定的输入样例添加哪种具体的标签。

而重构则是在猜测原始输入的概率分布，亦即同时预测许多不同的点的值。这被称为[生成学习](http://cs229.stanford.edu/notes/cs229-notes2.pdf)，必须和分类器所进行的判别学习区分开来，后者是将输

值映射至标签，用直线将数据点划分为不同的组。

------

RBM用Kullback Leibler散度来衡量预测的概率分布与输入值的基准分布之间的距离。

KL散度衡量两条曲线下方不重叠（即离散）的面积，而RBM的优化算法会*尝试将这些离散部分的面积最小化*，使共用权重在与第一隐藏层的激活值相乘后，可以得到与原始输入高度近似的结果。下图左半边是一组原始输入的概率分布曲线*p*，与之并列的是重构值的概率分布曲线*q*；右半边的图则显示了两条曲线之间的差异。

![](https://deeplearning4j.org/img/KL_divergence_RBM.png)

RBM根据权重产生的误差反复调整权重，以此学习估计原始数据的近似值。可以说权重会慢慢开始反映出输入的结构，而这种结构被编码为第一个隐藏层的激活值。整个学习过程看上去像是两条概率分布曲线在逐步重合。

![](https://deeplearning4j.org/img/KLD_update_RBM.png)

最后一点：你会发现RBM有两个偏置值。隐藏的偏置值帮助RBM在正向传递中生成激活值（因为偏置设定了下限，所以无论数据有多稀疏，至少有一部分节点会被激活），而可见层的偏置则帮助RBM通过反向传递学习重构数据。

### 代码和相关测试

#### 基础神经网络部分

基础网络部分是卷积网络之前的代码。

见代码部分的的network.py、network2.py、network3.py

其中network.py实现了最基本神经网络，包括随机梯度下降和反向传播，随机初始化权重和偏置，以及用sigmoid作了代价函数。

而network2.py在上述基础上对权重初始化进行了改进，引入了规范化，以及交叉熵代价函数（当然之前的都有所保留，通过关键字参数可以选择）。

my_test.py是各种测试代码的集合，也包含下面的卷积神经网络的测试。

说明：输入层都是$28*28=784$个神经元，因为其每次输入一个灰度图片，每个图片尺寸$28*28=784$，每个神经元输入是一个0~1之间的值，0.0代表白色，1.0代表黑色；输出0-9表示识别的结果。

以下代码对network.py进行了测试：

```python
import Deeplearning.network as network
import Deeplearning.mnist_loader as mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)


net = network.Network([784, 30, 10])#输入784个神经元，输出10个神经元，隐藏一层30个神经元
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)#随机梯度下降，进行30次迭代，mini_batch是10，学习速率3
```

以上测试的运行结果如下：

```python
D:\Anaconda\python.exe E:/pycharm/ML/Deeplearning/my_test.py
Epoch 0: 9075 / 10000
Epoch 1: 9256 / 10000
Epoch 2: 9359 / 10000
Epoch 3: 9322 / 10000
Epoch 4: 9398 / 10000
Epoch 5: 9398 / 10000
Epoch 6: 9422 / 10000
Epoch 7: 9422 / 10000
Epoch 8: 9417 / 10000
Epoch 9: 9463 / 10000
Epoch 10: 9448 / 10000
Epoch 11: 9458 / 10000
Epoch 12: 9467 / 10000
Epoch 13: 9482 / 10000
Epoch 14: 9482 / 10000
Epoch 15: 9479 / 10000
Epoch 16: 9484 / 10000
Epoch 17: 9486 / 10000
Epoch 18: 9493 / 10000
Epoch 19: 9496 / 10000
Epoch 20: 9501 / 10000
Epoch 21: 9468 / 10000
Epoch 22: 9494 / 10000
Epoch 23: 9494 / 10000
Epoch 24: 9478 / 10000
Epoch 25: 9495 / 10000
Epoch 26: 9474 / 10000
Epoch 27: 9483 / 10000
Epoch 28: 9478 / 10000
Epoch 29: 9485 / 10000
```

以下代码对network2.py进行测试，其网络选择和之前一样，不过用交叉熵作为代价函数。学习速率改为0.1，引入规范化参数$\lambda =5.0$。

````python
import Deeplearning.network2 as network2

net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
net.large_weight_initializer()
net.SGD(training_data, 30, 10, 0.1, lmbda = 5.0,evaluation_data=validation_data,
    monitor_evaluation_accuracy=True)
````

运行结果如下：

```python
D:\Anaconda\python.exe E:/pycharm/ML/Deeplearning/my_test.py
Epoch 0 training complete
Accuracy on evaluation data: 8654 / 10000
Epoch 1 training complete
Accuracy on evaluation data: 8982 / 10000
Epoch 2 training complete
Accuracy on evaluation data: 9107 / 10000
Epoch 3 training complete
Accuracy on evaluation data: 9190 / 10000
Epoch 4 training complete
Accuracy on evaluation data: 9275 / 10000
Epoch 5 training complete
Accuracy on evaluation data: 9322 / 10000
Epoch 6 training complete
Accuracy on evaluation data: 9357 / 10000
Epoch 7 training complete
Accuracy on evaluation data: 9380 / 10000
Epoch 8 training complete
Accuracy on evaluation data: 9421 / 10000
Epoch 9 training complete
Accuracy on evaluation data: 9467 / 10000
Epoch 10 training complete
Accuracy on evaluation data: 9477 / 10000
Epoch 11 training complete
Accuracy on evaluation data: 9500 / 10000
Epoch 12 training complete
Accuracy on evaluation data: 9511 / 10000
Epoch 13 training complete
Accuracy on evaluation data: 9517 / 10000
Epoch 14 training complete
Accuracy on evaluation data: 9518 / 10000
Epoch 15 training complete
Accuracy on evaluation data: 9554 / 10000
Epoch 16 training complete
Accuracy on evaluation data: 9572 / 10000
Epoch 17 training complete
Accuracy on evaluation data: 9549 / 10000
Epoch 18 training complete
Accuracy on evaluation data: 9567 / 10000
Epoch 19 training complete
Accuracy on evaluation data: 9567 / 10000
Epoch 20 training complete
Accuracy on evaluation data: 9576 / 10000
Epoch 21 training complete
Accuracy on evaluation data: 9589 / 10000
Epoch 22 training complete
Accuracy on evaluation data: 9572 / 10000
Epoch 23 training complete
Accuracy on evaluation data: 9570 / 10000
Epoch 24 training complete
Accuracy on evaluation data: 9596 / 10000
Epoch 25 training complete
Accuracy on evaluation data: 9610 / 10000
Epoch 26 training complete
Accuracy on evaluation data: 9607 / 10000
Epoch 27 training complete
Accuracy on evaluation data: 9609 / 10000
Epoch 28 training complete
Accuracy on evaluation data: 9627 / 10000
Epoch 29 training complete
Accuracy on evaluation data: 9607 / 10000
```

#### 卷积神经网络的测试

以下代码利用了一层卷积层，$5*5$的局部感受野，建立了20个特征映射，运用了最大值混合（$2*2$）；一层全连接层；以及柔性最大值。迭代60次，学习速率0.1

````python
net = Network([
    ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                  filter_shape=(20, 1, 5, 5),
                  poolsize=(2, 2)),
    FullyConnectedLayer(n_in=20*12*12, n_out=100),
    SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)
````

运行结果如下：

```python
D:\Anaconda\python.exe E:/pycharm/ML/Deeplearning/my_test.py
WARNING (theano.sandbox.cuda): The cuda backend is deprecated and will be removed in the next release (v0.10).  Please switch to the gpuarray backend. You can get more information about how to switch at this URL:
 https://github.com/Theano/Theano/wiki/Converting-to-the-new-gpu-back-end%28gpuarray%29

Using gpu device 0: GeForce GTX 860M (CNMeM is enabled with initial size: 80.0% of memory, cuDNN 5110)
Trying to run under a GPU.  If this is not desired, then modify network3.py
to set the GPU flag to False.
E:\pycharm\ML\Deeplearning\network3.py:233: UserWarning: DEPRECATION: the 'ds' parameter is not going to exist anymore as it is going to be replaced by the parameter 'ws'.
  input=conv_out, ds=self.poolsize, ignore_border=True)
Training mini-batch number 0
Training mini-batch number 1000
Training mini-batch number 2000
Training mini-batch number 3000
Training mini-batch number 4000
Epoch 0: validation accuracy 94.26%
This is the best validation accuracy to date.
The corresponding test accuracy is 93.46%
Training mini-batch number 5000
Training mini-batch number 6000
Training mini-batch number 7000
Training mini-batch number 8000
Training mini-batch number 9000
Epoch 1: validation accuracy 96.45%
This is the best validation accuracy to date.
The corresponding test accuracy is 96.05%
Training mini-batch number 10000
Training mini-batch number 11000
Training mini-batch number 12000
Training mini-batch number 13000
Training mini-batch number 14000
Epoch 2: validation accuracy 97.25%
This is the best validation accuracy to date.
The corresponding test accuracy is 96.93%
Training mini-batch number 15000
Training mini-batch number 16000
Training mini-batch number 17000
Training mini-batch number 18000
Training mini-batch number 19000
Epoch 3: validation accuracy 97.66%
This is the best validation accuracy to date.
The corresponding test accuracy is 97.48%
Training mini-batch number 20000
Training mini-batch number 21000
Training mini-batch number 22000
Training mini-batch number 23000
Training mini-batch number 24000
Epoch 4: validation accuracy 97.99%
This is the best validation accuracy to date.
The corresponding test accuracy is 97.79%
Training mini-batch number 25000
Training mini-batch number 26000
Training mini-batch number 27000
Training mini-batch number 28000
Training mini-batch number 29000
Epoch 5: validation accuracy 98.22%
This is the best validation accuracy to date.
The corresponding test accuracy is 97.97%
Training mini-batch number 30000
Training mini-batch number 31000
Training mini-batch number 32000
Training mini-batch number 33000
Training mini-batch number 34000
Epoch 6: validation accuracy 98.34%
This is the best validation accuracy to date.
The corresponding test accuracy is 98.07%
Training mini-batch number 35000
Training mini-batch number 36000
Training mini-batch number 37000
Training mini-batch number 38000
Training mini-batch number 39000
Epoch 7: validation accuracy 98.35%
This is the best validation accuracy to date.
The corresponding test accuracy is 98.24%
Training mini-batch number 40000
Training mini-batch number 41000
Training mini-batch number 42000
Training mini-batch number 43000
Training mini-batch number 44000
Epoch 8: validation accuracy 98.40%
This is the best validation accuracy to date.
The corresponding test accuracy is 98.37%
Training mini-batch number 45000
Training mini-batch number 46000
Training mini-batch number 47000
Training mini-batch number 48000
Training mini-batch number 49000
Epoch 9: validation accuracy 98.49%
This is the best validation accuracy to date.
The corresponding test accuracy is 98.44%
Training mini-batch number 50000
Training mini-batch number 51000
Training mini-batch number 52000
Training mini-batch number 53000
Training mini-batch number 54000
Epoch 10: validation accuracy 98.54%
This is the best validation accuracy to date.
The corresponding test accuracy is 98.48%
Training mini-batch number 55000
Training mini-batch number 56000
Training mini-batch number 57000
Training mini-batch number 58000
Training mini-batch number 59000
Epoch 11: validation accuracy 98.60%
This is the best validation accuracy to date.
The corresponding test accuracy is 98.49%
Training mini-batch number 60000
Training mini-batch number 61000
Training mini-batch number 62000
Training mini-batch number 63000
Training mini-batch number 64000
Epoch 12: validation accuracy 98.62%
This is the best validation accuracy to date.
The corresponding test accuracy is 98.58%
Training mini-batch number 65000
Training mini-batch number 66000
Training mini-batch number 67000
Training mini-batch number 68000
Training mini-batch number 69000
Epoch 13: validation accuracy 98.66%
This is the best validation accuracy to date.
The corresponding test accuracy is 98.59%
Training mini-batch number 70000
Training mini-batch number 71000
Training mini-batch number 72000
Training mini-batch number 73000
Training mini-batch number 74000
Epoch 14: validation accuracy 98.63%
Training mini-batch number 75000
Training mini-batch number 76000
Training mini-batch number 77000
Training mini-batch number 78000
Training mini-batch number 79000
Epoch 15: validation accuracy 98.64%
Training mini-batch number 80000
Training mini-batch number 81000
Training mini-batch number 82000
Training mini-batch number 83000
Training mini-batch number 84000
Epoch 16: validation accuracy 98.67%
This is the best validation accuracy to date.
The corresponding test accuracy is 98.63%
Training mini-batch number 85000
Training mini-batch number 86000
Training mini-batch number 87000
Training mini-batch number 88000
Training mini-batch number 89000
Epoch 17: validation accuracy 98.71%
This is the best validation accuracy to date.
The corresponding test accuracy is 98.62%
Training mini-batch number 90000
Training mini-batch number 91000
Training mini-batch number 92000
Training mini-batch number 93000
Training mini-batch number 94000
Epoch 18: validation accuracy 98.69%
Training mini-batch number 95000
Training mini-batch number 96000
Training mini-batch number 97000
Training mini-batch number 98000
Training mini-batch number 99000
Epoch 19: validation accuracy 98.71%
This is the best validation accuracy to date.
The corresponding test accuracy is 98.63%
Training mini-batch number 100000
Training mini-batch number 101000
Training mini-batch number 102000
Training mini-batch number 103000
Training mini-batch number 104000
Epoch 20: validation accuracy 98.72%
This is the best validation accuracy to date.
The corresponding test accuracy is 98.63%
Training mini-batch number 105000
Training mini-batch number 106000
Training mini-batch number 107000
Training mini-batch number 108000
Training mini-batch number 109000
Epoch 21: validation accuracy 98.73%
This is the best validation accuracy to date.
The corresponding test accuracy is 98.64%
Training mini-batch number 110000
Training mini-batch number 111000
Training mini-batch number 112000
Training mini-batch number 113000
Training mini-batch number 114000
Epoch 22: validation accuracy 98.72%
Training mini-batch number 115000
Training mini-batch number 116000
Training mini-batch number 117000
Training mini-batch number 118000
Training mini-batch number 119000
Epoch 23: validation accuracy 98.71%
Training mini-batch number 120000
Training mini-batch number 121000
Training mini-batch number 122000
Training mini-batch number 123000
Training mini-batch number 124000
Epoch 24: validation accuracy 98.73%
This is the best validation accuracy to date.
The corresponding test accuracy is 98.64%
Training mini-batch number 125000
Training mini-batch number 126000
Training mini-batch number 127000
Training mini-batch number 128000
Training mini-batch number 129000
Epoch 25: validation accuracy 98.75%
This is the best validation accuracy to date.
The corresponding test accuracy is 98.65%
Training mini-batch number 130000
Training mini-batch number 131000
Training mini-batch number 132000
Training mini-batch number 133000
Training mini-batch number 134000
Epoch 26: validation accuracy 98.77%
This is the best validation accuracy to date.
The corresponding test accuracy is 98.63%
Training mini-batch number 135000
Training mini-batch number 136000
Training mini-batch number 137000
Training mini-batch number 138000
Training mini-batch number 139000
Epoch 27: validation accuracy 98.77%
This is the best validation accuracy to date.
The corresponding test accuracy is 98.64%
Training mini-batch number 140000
Training mini-batch number 141000
Training mini-batch number 142000
Training mini-batch number 143000
Training mini-batch number 144000
Epoch 28: validation accuracy 98.77%
This is the best validation accuracy to date.
The corresponding test accuracy is 98.67%
Training mini-batch number 145000
Training mini-batch number 146000
Training mini-batch number 147000
Training mini-batch number 148000
Training mini-batch number 149000
Epoch 29: validation accuracy 98.78%
This is the best validation accuracy to date.
The corresponding test accuracy is 98.67%
Training mini-batch number 150000
Training mini-batch number 151000
Training mini-batch number 152000
Training mini-batch number 153000
Training mini-batch number 154000
Epoch 30: validation accuracy 98.78%
This is the best validation accuracy to date.
The corresponding test accuracy is 98.67%
Training mini-batch number 155000
Training mini-batch number 156000
Training mini-batch number 157000
Training mini-batch number 158000
Training mini-batch number 159000
Epoch 31: validation accuracy 98.79%
This is the best validation accuracy to date.
The corresponding test accuracy is 98.68%
Training mini-batch number 160000
Training mini-batch number 161000
Training mini-batch number 162000
Training mini-batch number 163000
Training mini-batch number 164000
Epoch 32: validation accuracy 98.78%
Training mini-batch number 165000
Training mini-batch number 166000
Training mini-batch number 167000
Training mini-batch number 168000
Training mini-batch number 169000
Epoch 33: validation accuracy 98.78%
Training mini-batch number 170000
Training mini-batch number 171000
Training mini-batch number 172000
Training mini-batch number 173000
Training mini-batch number 174000
Epoch 34: validation accuracy 98.78%
Training mini-batch number 175000
Training mini-batch number 176000
Training mini-batch number 177000
Training mini-batch number 178000
Training mini-batch number 179000
Epoch 35: validation accuracy 98.79%
This is the best validation accuracy to date.
The corresponding test accuracy is 98.69%
Training mini-batch number 180000
Training mini-batch number 181000
Training mini-batch number 182000
Training mini-batch number 183000
Training mini-batch number 184000
Epoch 36: validation accuracy 98.78%
Training mini-batch number 185000
Training mini-batch number 186000
Training mini-batch number 187000
Training mini-batch number 188000
Training mini-batch number 189000
Epoch 37: validation accuracy 98.78%
Training mini-batch number 190000
Training mini-batch number 191000
Training mini-batch number 192000
Training mini-batch number 193000
Training mini-batch number 194000
Epoch 38: validation accuracy 98.78%
Training mini-batch number 195000
Training mini-batch number 196000
Training mini-batch number 197000
Training mini-batch number 198000
Training mini-batch number 199000
Epoch 39: validation accuracy 98.78%
Training mini-batch number 200000
Training mini-batch number 201000
Training mini-batch number 202000
Training mini-batch number 203000
Training mini-batch number 204000
Epoch 40: validation accuracy 98.77%
Training mini-batch number 205000
Training mini-batch number 206000
Training mini-batch number 207000
Training mini-batch number 208000
Training mini-batch number 209000
Epoch 41: validation accuracy 98.77%
Training mini-batch number 210000
Training mini-batch number 211000
Training mini-batch number 212000
Training mini-batch number 213000
Training mini-batch number 214000
Epoch 42: validation accuracy 98.77%
Training mini-batch number 215000
Training mini-batch number 216000
Training mini-batch number 217000
Training mini-batch number 218000
Training mini-batch number 219000
Epoch 43: validation accuracy 98.77%
Training mini-batch number 220000
Training mini-batch number 221000
Training mini-batch number 222000
Training mini-batch number 223000
Training mini-batch number 224000
Epoch 44: validation accuracy 98.78%
Training mini-batch number 225000
Training mini-batch number 226000
Training mini-batch number 227000
Training mini-batch number 228000
Training mini-batch number 229000
Epoch 45: validation accuracy 98.78%
Training mini-batch number 230000
Training mini-batch number 231000
Training mini-batch number 232000
Training mini-batch number 233000
Training mini-batch number 234000
Epoch 46: validation accuracy 98.78%
Training mini-batch number 235000
Training mini-batch number 236000
Training mini-batch number 237000
Training mini-batch number 238000
Training mini-batch number 239000
Epoch 47: validation accuracy 98.78%
Training mini-batch number 240000
Training mini-batch number 241000
Training mini-batch number 242000
Training mini-batch number 243000
Training mini-batch number 244000
Epoch 48: validation accuracy 98.79%
Training mini-batch number 245000
Training mini-batch number 246000
Training mini-batch number 247000
Training mini-batch number 248000
Training mini-batch number 249000
Epoch 49: validation accuracy 98.79%
Training mini-batch number 250000
Training mini-batch number 251000
Training mini-batch number 252000
Training mini-batch number 253000
Training mini-batch number 254000
Epoch 50: validation accuracy 98.79%
Training mini-batch number 255000
Training mini-batch number 256000
Training mini-batch number 257000
Training mini-batch number 258000
Training mini-batch number 259000
Epoch 51: validation accuracy 98.80%
This is the best validation accuracy to date.
The corresponding test accuracy is 98.73%
Training mini-batch number 260000
Training mini-batch number 261000
Training mini-batch number 262000
Training mini-batch number 263000
Training mini-batch number 264000
Epoch 52: validation accuracy 98.79%
Training mini-batch number 265000
Training mini-batch number 266000
Training mini-batch number 267000
Training mini-batch number 268000
Training mini-batch number 269000
Epoch 53: validation accuracy 98.79%
Training mini-batch number 270000
Training mini-batch number 271000
Training mini-batch number 272000
Training mini-batch number 273000
Training mini-batch number 274000
Epoch 54: validation accuracy 98.79%
Training mini-batch number 275000
Training mini-batch number 276000
Training mini-batch number 277000
Training mini-batch number 278000
Training mini-batch number 279000
Epoch 55: validation accuracy 98.78%
Training mini-batch number 280000
Training mini-batch number 281000
Training mini-batch number 282000
Training mini-batch number 283000
Training mini-batch number 284000
Epoch 56: validation accuracy 98.78%
Training mini-batch number 285000
Training mini-batch number 286000
Training mini-batch number 287000
Training mini-batch number 288000
Training mini-batch number 289000
Epoch 57: validation accuracy 98.78%
Training mini-batch number 290000
Training mini-batch number 291000
Training mini-batch number 292000
Training mini-batch number 293000
Training mini-batch number 294000
Epoch 58: validation accuracy 98.78%
Training mini-batch number 295000
Training mini-batch number 296000
Training mini-batch number 297000
Training mini-batch number 298000
Training mini-batch number 299000
Epoch 59: validation accuracy 98.78%
Finished training network.
Best validation accuracy of 98.80% obtained at iteration 259999
Corresponding test accuracy of 98.73%

Process finished with exit code 0

```

以下代码在上述测试代码的基础上又加了一层卷积层，其余基本不变。

```python
net = Network([
    ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                  filter_shape=(20, 1, 5, 5),
                  poolsize=(2, 2)),
    ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                  filter_shape=(40, 20, 5, 5),
                  poolsize=(2, 2)),
    FullyConnectedLayer(n_in=40*4*4, n_out=100),
    SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)
```

测试结果如下：

```python
D:\Anaconda\python.exe E:/pycharm/ML/Deeplearning/my_test.py
WARNING (theano.sandbox.cuda): The cuda backend is deprecated and will be removed in the next release (v0.10).  Please switch to the gpuarray backend. You can get more information about how to switch at this URL:
 https://github.com/Theano/Theano/wiki/Converting-to-the-new-gpu-back-end%28gpuarray%29

Using gpu device 0: GeForce GTX 860M (CNMeM is enabled with initial size: 80.0% of memory, cuDNN 5110)
Trying to run under a GPU.  If this is not desired, then modify network3.py
to set the GPU flag to False.
E:\pycharm\ML\Deeplearning\network3.py:233: UserWarning: DEPRECATION: the 'ds' parameter is not going to exist anymore as it is going to be replaced by the parameter 'ws'.
  input=conv_out, ds=self.poolsize, ignore_border=True)
Training mini-batch number 0
Training mini-batch number 1000
Training mini-batch number 2000
Training mini-batch number 3000
Training mini-batch number 4000
Epoch 0: validation accuracy 89.79%
This is the best validation accuracy to date.
The corresponding test accuracy is 89.50%
Training mini-batch number 5000
Training mini-batch number 6000
Training mini-batch number 7000
Training mini-batch number 8000
Training mini-batch number 9000
Epoch 1: validation accuracy 96.44%
This is the best validation accuracy to date.
The corresponding test accuracy is 96.45%
Training mini-batch number 10000
Training mini-batch number 11000
Training mini-batch number 12000
Training mini-batch number 13000
Training mini-batch number 14000
Epoch 2: validation accuracy 97.32%
This is the best validation accuracy to date.
The corresponding test accuracy is 97.17%
Training mini-batch number 15000
Training mini-batch number 16000
Training mini-batch number 17000
Training mini-batch number 18000
Training mini-batch number 19000
Epoch 3: validation accuracy 97.73%
This is the best validation accuracy to date.
The corresponding test accuracy is 97.72%
Training mini-batch number 20000
Training mini-batch number 21000
Training mini-batch number 22000
Training mini-batch number 23000
Training mini-batch number 24000
Epoch 4: validation accuracy 97.95%
This is the best validation accuracy to date.
The corresponding test accuracy is 98.04%
Training mini-batch number 25000
Training mini-batch number 26000
Training mini-batch number 27000
Training mini-batch number 28000
Training mini-batch number 29000
Epoch 5: validation accuracy 98.14%
This is the best validation accuracy to date.
The corresponding test accuracy is 98.29%
Training mini-batch number 30000
Training mini-batch number 31000
Training mini-batch number 32000
Training mini-batch number 33000
Training mini-batch number 34000
Epoch 6: validation accuracy 98.31%
This is the best validation accuracy to date.
The corresponding test accuracy is 98.39%
Training mini-batch number 35000
Training mini-batch number 36000
Training mini-batch number 37000
Training mini-batch number 38000
Training mini-batch number 39000
Epoch 7: validation accuracy 98.37%
This is the best validation accuracy to date.
The corresponding test accuracy is 98.47%
Training mini-batch number 40000
Training mini-batch number 41000
Training mini-batch number 42000
Training mini-batch number 43000
Training mini-batch number 44000
Epoch 8: validation accuracy 98.47%
This is the best validation accuracy to date.
The corresponding test accuracy is 98.55%
Training mini-batch number 45000
Training mini-batch number 46000
Training mini-batch number 47000
Training mini-batch number 48000
Training mini-batch number 49000
Epoch 9: validation accuracy 98.52%
This is the best validation accuracy to date.
The corresponding test accuracy is 98.61%
Training mini-batch number 50000
Training mini-batch number 51000
Training mini-batch number 52000
Training mini-batch number 53000
Training mini-batch number 54000
Epoch 10: validation accuracy 98.65%
This is the best validation accuracy to date.
The corresponding test accuracy is 98.62%
Training mini-batch number 55000
Training mini-batch number 56000
Training mini-batch number 57000
Training mini-batch number 58000
Training mini-batch number 59000
Epoch 11: validation accuracy 98.67%
This is the best validation accuracy to date.
The corresponding test accuracy is 98.71%
Training mini-batch number 60000
Training mini-batch number 61000
Training mini-batch number 62000
Training mini-batch number 63000
Training mini-batch number 64000
Epoch 12: validation accuracy 98.73%
This is the best validation accuracy to date.
The corresponding test accuracy is 98.71%
Training mini-batch number 65000
Training mini-batch number 66000
Training mini-batch number 67000
Training mini-batch number 68000
Training mini-batch number 69000
Epoch 13: validation accuracy 98.75%
This is the best validation accuracy to date.
The corresponding test accuracy is 98.68%
Training mini-batch number 70000
Training mini-batch number 71000
Training mini-batch number 72000
Training mini-batch number 73000
Training mini-batch number 74000
Epoch 14: validation accuracy 98.78%
This is the best validation accuracy to date.
The corresponding test accuracy is 98.70%
Training mini-batch number 75000
Training mini-batch number 76000
Training mini-batch number 77000
Training mini-batch number 78000
Training mini-batch number 79000
Epoch 15: validation accuracy 98.84%
This is the best validation accuracy to date.
The corresponding test accuracy is 98.71%
Training mini-batch number 80000
Training mini-batch number 81000
Training mini-batch number 82000
Training mini-batch number 83000
Training mini-batch number 84000
Epoch 16: validation accuracy 98.85%
This is the best validation accuracy to date.
The corresponding test accuracy is 98.76%
Training mini-batch number 85000
Training mini-batch number 86000
Training mini-batch number 87000
Training mini-batch number 88000
Training mini-batch number 89000
Epoch 17: validation accuracy 98.87%
This is the best validation accuracy to date.
The corresponding test accuracy is 98.79%
Training mini-batch number 90000
Training mini-batch number 91000
Training mini-batch number 92000
Training mini-batch number 93000
Training mini-batch number 94000
Epoch 18: validation accuracy 98.89%
This is the best validation accuracy to date.
The corresponding test accuracy is 98.76%
Training mini-batch number 95000
Training mini-batch number 96000
Training mini-batch number 97000
Training mini-batch number 98000
Training mini-batch number 99000
Epoch 19: validation accuracy 98.91%
This is the best validation accuracy to date.
The corresponding test accuracy is 98.80%
Training mini-batch number 100000
Training mini-batch number 101000
Training mini-batch number 102000
Training mini-batch number 103000
Training mini-batch number 104000
Epoch 20: validation accuracy 98.92%
This is the best validation accuracy to date.
The corresponding test accuracy is 98.81%
Training mini-batch number 105000
Training mini-batch number 106000
Training mini-batch number 107000
Training mini-batch number 108000
Training mini-batch number 109000
Epoch 21: validation accuracy 98.95%
This is the best validation accuracy to date.
The corresponding test accuracy is 98.80%
Training mini-batch number 110000
Training mini-batch number 111000
Training mini-batch number 112000
Training mini-batch number 113000
Training mini-batch number 114000
Epoch 22: validation accuracy 98.94%
Training mini-batch number 115000
Training mini-batch number 116000
Training mini-batch number 117000
Training mini-batch number 118000
Training mini-batch number 119000
Epoch 23: validation accuracy 98.95%
This is the best validation accuracy to date.
The corresponding test accuracy is 98.84%
Training mini-batch number 120000
Training mini-batch number 121000
Training mini-batch number 122000
Training mini-batch number 123000
Training mini-batch number 124000
Epoch 24: validation accuracy 98.96%
This is the best validation accuracy to date.
The corresponding test accuracy is 98.85%
Training mini-batch number 125000
Training mini-batch number 126000
Training mini-batch number 127000
Training mini-batch number 128000
Training mini-batch number 129000
Epoch 25: validation accuracy 98.97%
This is the best validation accuracy to date.
The corresponding test accuracy is 98.84%
Training mini-batch number 130000
Training mini-batch number 131000
Training mini-batch number 132000
Training mini-batch number 133000
Training mini-batch number 134000
Epoch 26: validation accuracy 98.95%
Training mini-batch number 135000
Training mini-batch number 136000
Training mini-batch number 137000
Training mini-batch number 138000
Training mini-batch number 139000
Epoch 27: validation accuracy 98.95%
Training mini-batch number 140000
Training mini-batch number 141000
Training mini-batch number 142000
Training mini-batch number 143000
Training mini-batch number 144000
Epoch 28: validation accuracy 98.96%
Training mini-batch number 145000
Training mini-batch number 146000
Training mini-batch number 147000
Training mini-batch number 148000
Training mini-batch number 149000
Epoch 29: validation accuracy 98.94%
Training mini-batch number 150000
Training mini-batch number 151000
Training mini-batch number 152000
Training mini-batch number 153000
Training mini-batch number 154000
Epoch 30: validation accuracy 98.94%
Training mini-batch number 155000
Training mini-batch number 156000
Training mini-batch number 157000
Training mini-batch number 158000
Training mini-batch number 159000
Epoch 31: validation accuracy 98.93%
Training mini-batch number 160000
Training mini-batch number 161000
Training mini-batch number 162000
Training mini-batch number 163000
Training mini-batch number 164000
Epoch 32: validation accuracy 98.92%
Training mini-batch number 165000
Training mini-batch number 166000
Training mini-batch number 167000
Training mini-batch number 168000
Training mini-batch number 169000
Epoch 33: validation accuracy 98.93%
Training mini-batch number 170000
Training mini-batch number 171000
Training mini-batch number 172000
Training mini-batch number 173000
Training mini-batch number 174000
Epoch 34: validation accuracy 98.93%
Training mini-batch number 175000
Training mini-batch number 176000
Training mini-batch number 177000
Training mini-batch number 178000
Training mini-batch number 179000
Epoch 35: validation accuracy 98.94%
Training mini-batch number 180000
Training mini-batch number 181000
Training mini-batch number 182000
Training mini-batch number 183000
Training mini-batch number 184000
Epoch 36: validation accuracy 98.95%
Training mini-batch number 185000
Training mini-batch number 186000
Training mini-batch number 187000
Training mini-batch number 188000
Training mini-batch number 189000
Epoch 37: validation accuracy 98.96%
Training mini-batch number 190000
Training mini-batch number 191000
Training mini-batch number 192000
Training mini-batch number 193000
Training mini-batch number 194000
Epoch 38: validation accuracy 98.97%
This is the best validation accuracy to date.
The corresponding test accuracy is 99.01%
Training mini-batch number 195000
Training mini-batch number 196000
Training mini-batch number 197000
Training mini-batch number 198000
Training mini-batch number 199000
Epoch 39: validation accuracy 98.96%
Training mini-batch number 200000
Training mini-batch number 201000
Training mini-batch number 202000
Training mini-batch number 203000
Training mini-batch number 204000
Epoch 40: validation accuracy 98.97%
This is the best validation accuracy to date.
The corresponding test accuracy is 99.03%
Training mini-batch number 205000
Training mini-batch number 206000
Training mini-batch number 207000
Training mini-batch number 208000
Training mini-batch number 209000
Epoch 41: validation accuracy 98.96%
Training mini-batch number 210000
Training mini-batch number 211000
Training mini-batch number 212000
Training mini-batch number 213000
Training mini-batch number 214000
Epoch 42: validation accuracy 98.96%
Training mini-batch number 215000
Training mini-batch number 216000
Training mini-batch number 217000
Training mini-batch number 218000
Training mini-batch number 219000
Epoch 43: validation accuracy 98.98%
This is the best validation accuracy to date.
The corresponding test accuracy is 99.03%
Training mini-batch number 220000
Training mini-batch number 221000
Training mini-batch number 222000
Training mini-batch number 223000
Training mini-batch number 224000
Epoch 44: validation accuracy 98.97%
Training mini-batch number 225000
Training mini-batch number 226000
Training mini-batch number 227000
Training mini-batch number 228000
Training mini-batch number 229000
Epoch 45: validation accuracy 98.97%
Training mini-batch number 230000
Training mini-batch number 231000
Training mini-batch number 232000
Training mini-batch number 233000
Training mini-batch number 234000
Epoch 46: validation accuracy 98.97%
Training mini-batch number 235000
Training mini-batch number 236000
Training mini-batch number 237000
Training mini-batch number 238000
Training mini-batch number 239000
Epoch 47: validation accuracy 98.95%
Training mini-batch number 240000
Training mini-batch number 241000
Training mini-batch number 242000
Training mini-batch number 243000
Training mini-batch number 244000
Epoch 48: validation accuracy 98.95%
Training mini-batch number 245000
Training mini-batch number 246000
Training mini-batch number 247000
Training mini-batch number 248000
Training mini-batch number 249000
Epoch 49: validation accuracy 98.95%
Training mini-batch number 250000
Training mini-batch number 251000
Training mini-batch number 252000
Training mini-batch number 253000
Training mini-batch number 254000
Epoch 50: validation accuracy 98.96%
Training mini-batch number 255000
Training mini-batch number 256000
Training mini-batch number 257000
Training mini-batch number 258000
Training mini-batch number 259000
Epoch 51: validation accuracy 98.98%
This is the best validation accuracy to date.
The corresponding test accuracy is 99.04%
Training mini-batch number 260000
Training mini-batch number 261000
Training mini-batch number 262000
Training mini-batch number 263000
Training mini-batch number 264000
Epoch 52: validation accuracy 98.99%
This is the best validation accuracy to date.
The corresponding test accuracy is 99.04%
Training mini-batch number 265000
Training mini-batch number 266000
Training mini-batch number 267000
Training mini-batch number 268000
Training mini-batch number 269000
Epoch 53: validation accuracy 99.00%
This is the best validation accuracy to date.
The corresponding test accuracy is 99.05%
Training mini-batch number 270000
Training mini-batch number 271000
Training mini-batch number 272000
Training mini-batch number 273000
Training mini-batch number 274000
Epoch 54: validation accuracy 98.99%
Training mini-batch number 275000
Training mini-batch number 276000
Training mini-batch number 277000
Training mini-batch number 278000
Training mini-batch number 279000
Epoch 55: validation accuracy 98.99%
Training mini-batch number 280000
Training mini-batch number 281000
Training mini-batch number 282000
Training mini-batch number 283000
Training mini-batch number 284000
Epoch 56: validation accuracy 99.01%
This is the best validation accuracy to date.
The corresponding test accuracy is 99.05%
Training mini-batch number 285000
Training mini-batch number 286000
Training mini-batch number 287000
Training mini-batch number 288000
Training mini-batch number 289000
Epoch 57: validation accuracy 99.01%
This is the best validation accuracy to date.
The corresponding test accuracy is 99.05%
Training mini-batch number 290000
Training mini-batch number 291000
Training mini-batch number 292000
Training mini-batch number 293000
Training mini-batch number 294000
Epoch 58: validation accuracy 99.01%
This is the best validation accuracy to date.
The corresponding test accuracy is 99.04%
Training mini-batch number 295000
Training mini-batch number 296000
Training mini-batch number 297000
Training mini-batch number 298000
Training mini-batch number 299000
Epoch 59: validation accuracy 99.01%
This is the best validation accuracy to date.
The corresponding test accuracy is 99.03%
Finished training network.
Best validation accuracy of 99.01% obtained at iteration 299999
Corresponding test accuracy of 99.03%

Process finished with exit code 0

```

最后，下面的代码用到了线性修正单元和L2规范化，精确度达到了最高！

```python
from Deeplearning.network3 import ReLU
net = Network([
    ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                  filter_shape=(20, 1, 5, 5),
                  poolsize=(2, 2),
                  activation_fn=ReLU),
    ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                  filter_shape=(40, 20, 5, 5),
                  poolsize=(2, 2),
                  activation_fn=ReLU),
    FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=ReLU),
    SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
net.SGD(training_data, 60, mini_batch_size, 0.03, validation_data, test_data, lmbda=0.1)
```

运行结果：

```python
D:\Anaconda\python.exe E:/pycharm/ML/Deeplearning/my_test.py
WARNING (theano.sandbox.cuda): The cuda backend is deprecated and will be removed in the next release (v0.10).  Please switch to the gpuarray backend. You can get more information about how to switch at this URL:
 https://github.com/Theano/Theano/wiki/Converting-to-the-new-gpu-back-end%28gpuarray%29

Using gpu device 0: GeForce GTX 860M (CNMeM is enabled with initial size: 80.0% of memory, cuDNN 5110)
Trying to run under a GPU.  If this is not desired, then modify network3.py
to set the GPU flag to False.
E:\pycharm\ML\Deeplearning\network3.py:233: UserWarning: DEPRECATION: the 'ds' parameter is not going to exist anymore as it is going to be replaced by the parameter 'ws'.
  input=conv_out, ds=self.poolsize, ignore_border=True)


Training mini-batch number 0
Training mini-batch number 1000
Training mini-batch number 2000
Training mini-batch number 3000
Training mini-batch number 4000
Epoch 0: validation accuracy 97.65%
This is the best validation accuracy to date.
The corresponding test accuracy is 97.22%
Training mini-batch number 5000
Training mini-batch number 6000
Training mini-batch number 7000
Training mini-batch number 8000
Training mini-batch number 9000
Epoch 1: validation accuracy 98.19%
This is the best validation accuracy to date.
The corresponding test accuracy is 98.11%
Training mini-batch number 10000
Training mini-batch number 11000
Training mini-batch number 12000
Training mini-batch number 13000
Training mini-batch number 14000
Epoch 2: validation accuracy 98.35%
This is the best validation accuracy to date.
The corresponding test accuracy is 98.31%
Training mini-batch number 15000
Training mini-batch number 16000
Training mini-batch number 17000
Training mini-batch number 18000
Training mini-batch number 19000
Epoch 3: validation accuracy 98.46%
This is the best validation accuracy to date.
The corresponding test accuracy is 98.32%
Training mini-batch number 20000
Training mini-batch number 21000
Training mini-batch number 22000
Training mini-batch number 23000
Training mini-batch number 24000
Epoch 4: validation accuracy 98.53%
This is the best validation accuracy to date.
The corresponding test accuracy is 98.68%
Training mini-batch number 25000
Training mini-batch number 26000
Training mini-batch number 27000
Training mini-batch number 28000
Training mini-batch number 29000
Epoch 5: validation accuracy 98.53%
Training mini-batch number 30000
Training mini-batch number 31000
Training mini-batch number 32000
Training mini-batch number 33000
Training mini-batch number 34000
Epoch 6: validation accuracy 98.63%
This is the best validation accuracy to date.
The corresponding test accuracy is 98.48%
Training mini-batch number 35000
Training mini-batch number 36000
Training mini-batch number 37000
Training mini-batch number 38000
Training mini-batch number 39000
Epoch 7: validation accuracy 98.62%
Training mini-batch number 40000
Training mini-batch number 41000
Training mini-batch number 42000
Training mini-batch number 43000
Training mini-batch number 44000
Epoch 8: validation accuracy 98.72%
This is the best validation accuracy to date.
The corresponding test accuracy is 98.71%
Training mini-batch number 45000
Training mini-batch number 46000
Training mini-batch number 47000
Training mini-batch number 48000
Training mini-batch number 49000
Epoch 9: validation accuracy 98.52%
Training mini-batch number 50000
Training mini-batch number 51000
Training mini-batch number 52000
Training mini-batch number 53000
Training mini-batch number 54000
Epoch 10: validation accuracy 98.65%
Training mini-batch number 55000
Training mini-batch number 56000
Training mini-batch number 57000
Training mini-batch number 58000
Training mini-batch number 59000
Epoch 11: validation accuracy 98.71%
Training mini-batch number 60000
Training mini-batch number 61000
Training mini-batch number 62000
Training mini-batch number 63000
Training mini-batch number 64000
Epoch 12: validation accuracy 98.65%
Training mini-batch number 65000
Training mini-batch number 66000
Training mini-batch number 67000
Training mini-batch number 68000
Training mini-batch number 69000
Epoch 13: validation accuracy 98.72%
This is the best validation accuracy to date.
The corresponding test accuracy is 98.91%
Training mini-batch number 70000
Training mini-batch number 71000
Training mini-batch number 72000
Training mini-batch number 73000
Training mini-batch number 74000
Epoch 14: validation accuracy 98.78%
This is the best validation accuracy to date.
The corresponding test accuracy is 98.98%
Training mini-batch number 75000
Training mini-batch number 76000
Training mini-batch number 77000
Training mini-batch number 78000
Training mini-batch number 79000
Epoch 15: validation accuracy 98.71%
Training mini-batch number 80000
Training mini-batch number 81000
Training mini-batch number 82000
Training mini-batch number 83000
Training mini-batch number 84000
Epoch 16: validation accuracy 98.76%
Training mini-batch number 85000
Training mini-batch number 86000
Training mini-batch number 87000
Training mini-batch number 88000
Training mini-batch number 89000
Epoch 17: validation accuracy 98.88%
This is the best validation accuracy to date.
The corresponding test accuracy is 99.05%
Training mini-batch number 90000
Training mini-batch number 91000
Training mini-batch number 92000
Training mini-batch number 93000
Training mini-batch number 94000
Epoch 18: validation accuracy 98.89%
This is the best validation accuracy to date.
The corresponding test accuracy is 99.06%
Training mini-batch number 95000
Training mini-batch number 96000
Training mini-batch number 97000
Training mini-batch number 98000
Training mini-batch number 99000
Epoch 19: validation accuracy 98.75%
Training mini-batch number 100000
Training mini-batch number 101000
Training mini-batch number 102000
Training mini-batch number 103000
Training mini-batch number 104000
Epoch 20: validation accuracy 98.94%
This is the best validation accuracy to date.
The corresponding test accuracy is 99.01%
Training mini-batch number 105000
Training mini-batch number 106000
Training mini-batch number 107000
Training mini-batch number 108000
Training mini-batch number 109000
Epoch 21: validation accuracy 99.06%
This is the best validation accuracy to date.
The corresponding test accuracy is 99.08%
Training mini-batch number 110000
Training mini-batch number 111000
Training mini-batch number 112000
Training mini-batch number 113000
Training mini-batch number 114000
Epoch 22: validation accuracy 98.89%
Training mini-batch number 115000
Training mini-batch number 116000
Training mini-batch number 117000
Training mini-batch number 118000
Training mini-batch number 119000
Epoch 23: validation accuracy 99.02%
Training mini-batch number 120000
Training mini-batch number 121000
Training mini-batch number 122000
Training mini-batch number 123000
Training mini-batch number 124000
Epoch 24: validation accuracy 99.03%
Training mini-batch number 125000
Training mini-batch number 126000
Training mini-batch number 127000
Training mini-batch number 128000
Training mini-batch number 129000
Epoch 25: validation accuracy 99.07%
This is the best validation accuracy to date.
The corresponding test accuracy is 99.17%
Training mini-batch number 130000
Training mini-batch number 131000
Training mini-batch number 132000
Training mini-batch number 133000
Training mini-batch number 134000
Epoch 26: validation accuracy 99.08%
This is the best validation accuracy to date.
The corresponding test accuracy is 99.17%
Training mini-batch number 135000
Training mini-batch number 136000
Training mini-batch number 137000
Training mini-batch number 138000
Training mini-batch number 139000
Epoch 27: validation accuracy 99.09%
This is the best validation accuracy to date.
The corresponding test accuracy is 99.16%
Training mini-batch number 140000
Training mini-batch number 141000
Training mini-batch number 142000
Training mini-batch number 143000
Training mini-batch number 144000
Epoch 28: validation accuracy 99.10%
This is the best validation accuracy to date.
The corresponding test accuracy is 99.15%
Training mini-batch number 145000
Training mini-batch number 146000
Training mini-batch number 147000
Training mini-batch number 148000
Training mini-batch number 149000
Epoch 29: validation accuracy 99.08%
Training mini-batch number 150000
Training mini-batch number 151000
Training mini-batch number 152000
Training mini-batch number 153000
Training mini-batch number 154000
Epoch 30: validation accuracy 99.09%
Training mini-batch number 155000
Training mini-batch number 156000
Training mini-batch number 157000
Training mini-batch number 158000
Training mini-batch number 159000
Epoch 31: validation accuracy 99.10%
This is the best validation accuracy to date.
The corresponding test accuracy is 99.16%
Training mini-batch number 160000
Training mini-batch number 161000
Training mini-batch number 162000
Training mini-batch number 163000
Training mini-batch number 164000
Epoch 32: validation accuracy 99.10%
This is the best validation accuracy to date.
The corresponding test accuracy is 99.16%
Training mini-batch number 165000
Training mini-batch number 166000
Training mini-batch number 167000
Training mini-batch number 168000
Training mini-batch number 169000
Epoch 33: validation accuracy 99.12%
This is the best validation accuracy to date.
The corresponding test accuracy is 99.16%
Training mini-batch number 170000
Training mini-batch number 171000
Training mini-batch number 172000
Training mini-batch number 173000
Training mini-batch number 174000
Epoch 34: validation accuracy 99.12%
This is the best validation accuracy to date.
The corresponding test accuracy is 99.16%
Training mini-batch number 175000
Training mini-batch number 176000
Training mini-batch number 177000
Training mini-batch number 178000
Training mini-batch number 179000
Epoch 35: validation accuracy 99.12%
This is the best validation accuracy to date.
The corresponding test accuracy is 99.17%
Training mini-batch number 180000
Training mini-batch number 181000
Training mini-batch number 182000
Training mini-batch number 183000
Training mini-batch number 184000
Epoch 36: validation accuracy 99.11%
Training mini-batch number 185000
Training mini-batch number 186000
Training mini-batch number 187000
Training mini-batch number 188000
Training mini-batch number 189000
Epoch 37: validation accuracy 99.11%
Training mini-batch number 190000
Training mini-batch number 191000
Training mini-batch number 192000
Training mini-batch number 193000
Training mini-batch number 194000
Epoch 38: validation accuracy 99.11%
Training mini-batch number 195000
Training mini-batch number 196000
Training mini-batch number 197000
Training mini-batch number 198000
Training mini-batch number 199000
Epoch 39: validation accuracy 99.10%
Training mini-batch number 200000
Training mini-batch number 201000
Training mini-batch number 202000
Training mini-batch number 203000
Training mini-batch number 204000
Epoch 40: validation accuracy 99.10%
Training mini-batch number 205000
Training mini-batch number 206000
Training mini-batch number 207000
Training mini-batch number 208000
Training mini-batch number 209000
Epoch 41: validation accuracy 99.10%
Training mini-batch number 210000
Training mini-batch number 211000
Training mini-batch number 212000
Training mini-batch number 213000
Training mini-batch number 214000
Epoch 42: validation accuracy 99.10%
Training mini-batch number 215000
Training mini-batch number 216000
Training mini-batch number 217000
Training mini-batch number 218000
Training mini-batch number 219000
Epoch 43: validation accuracy 99.10%
Training mini-batch number 220000
Training mini-batch number 221000
Training mini-batch number 222000
Training mini-batch number 223000
Training mini-batch number 224000
Epoch 44: validation accuracy 99.10%
Training mini-batch number 225000
Training mini-batch number 226000
Training mini-batch number 227000
Training mini-batch number 228000
Training mini-batch number 229000
Epoch 45: validation accuracy 99.10%
Training mini-batch number 230000
Training mini-batch number 231000
Training mini-batch number 232000
Training mini-batch number 233000
Training mini-batch number 234000
Epoch 46: validation accuracy 99.09%
Training mini-batch number 235000
Training mini-batch number 236000
Training mini-batch number 237000
Training mini-batch number 238000
Training mini-batch number 239000
Epoch 47: validation accuracy 99.10%
Training mini-batch number 240000
Training mini-batch number 241000
Training mini-batch number 242000
Training mini-batch number 243000
Training mini-batch number 244000
Epoch 48: validation accuracy 99.09%
Training mini-batch number 245000
Training mini-batch number 246000
Training mini-batch number 247000
Training mini-batch number 248000
Training mini-batch number 249000
Epoch 49: validation accuracy 99.08%
Training mini-batch number 250000
Training mini-batch number 251000
Training mini-batch number 252000
Training mini-batch number 253000
Training mini-batch number 254000
Epoch 50: validation accuracy 99.08%
Training mini-batch number 255000
Training mini-batch number 256000
Training mini-batch number 257000
Training mini-batch number 258000
Training mini-batch number 259000
Epoch 51: validation accuracy 99.08%
Training mini-batch number 260000
Training mini-batch number 261000
Training mini-batch number 262000
Training mini-batch number 263000
Training mini-batch number 264000
Epoch 52: validation accuracy 99.08%
Training mini-batch number 265000
Training mini-batch number 266000
Training mini-batch number 267000
Training mini-batch number 268000
Training mini-batch number 269000
Epoch 53: validation accuracy 99.08%
Training mini-batch number 270000
Training mini-batch number 271000
Training mini-batch number 272000
Training mini-batch number 273000
Training mini-batch number 274000
Epoch 54: validation accuracy 99.08%
Training mini-batch number 275000
Training mini-batch number 276000
Training mini-batch number 277000
Training mini-batch number 278000
Training mini-batch number 279000
Epoch 55: validation accuracy 99.08%
Training mini-batch number 280000
Training mini-batch number 281000
Training mini-batch number 282000
Training mini-batch number 283000
Training mini-batch number 284000
Epoch 56: validation accuracy 99.08%
Training mini-batch number 285000
Training mini-batch number 286000
Training mini-batch number 287000
Training mini-batch number 288000
Training mini-batch number 289000
Epoch 57: validation accuracy 99.08%
Training mini-batch number 290000
Training mini-batch number 291000
Training mini-batch number 292000
Training mini-batch number 293000
Training mini-batch number 294000
Epoch 58: validation accuracy 99.08%
Training mini-batch number 295000
Training mini-batch number 296000
Training mini-batch number 297000
Training mini-batch number 298000
Training mini-batch number 299000
Epoch 59: validation accuracy 99.08%
Finished training network.
Best validation accuracy of 99.12% obtained at iteration 179999
Corresponding test accuracy of 99.17%

Process finished with exit code 0

```







### 鸣谢

本文还参考了以下博客和资料：

[卷积神经网络(CNN)学习笔记1：基础入门](http://www.jeyzhang.com/cnn-learning-notes-1.html)

[我对卷积的理解](http://mengqi92.github.io/2015/10/06/convolution/)

[玻尔兹曼机](https://deeplearning4j.org/cn/restrictedboltzmannmachine)

[RBM学习笔记](https://zhuanlan.zhihu.com/p/22794772)





