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

