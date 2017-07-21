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

