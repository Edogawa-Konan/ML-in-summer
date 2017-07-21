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


