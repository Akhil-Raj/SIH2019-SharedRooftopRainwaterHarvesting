
# coding: utf-8

# In[55]:


from sklearn.cluster import KMeans
#from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

#read = [[1, 2], [2, 3], [5, 6], [7,8]]

def req_k() :
    diff = 0
    for x in range(1,len(data)-1):
        m1
        = (distortions[x]-distortions[x-1])
        m2 = (distortions[x+1]-distortions[x])
        m = abs(m2-m1)
        if m>diff:
            diff = m
            k = x+1
    return(k)

def find_k(data):
    X = np.array(data)
    distortions = []
    K = range(1,len(data)+1)
    for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(data)
        kmeanModel.fit(data)
        distortions.append(sum(np.min(cdist(data, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
    print(k, distortions)
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()
    k = req_k()
    return (k)


if __name__ == '__main__':
    k = find_k(read)
    print(k)

