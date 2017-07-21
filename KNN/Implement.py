import csv
import random
from math import sqrt
from operator import itemgetter

def loadDataset(filename,split,trainingSet,testSet):
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
    return sorted(Class,key=itemgetter(1),reverse=True)[0]

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



