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

