
# coding: utf-8

# In[75]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from scipy.spatial.distance import cdist
import seaborn as sns
import matplotlib.pyplot as plt


# In[17]:


data = pd.read_csv("roof.csv")
print(data.describe())
print("\n")
data.fillna(data.mean(), inplace=True) #remove_missing_values


# In[76]:


# X = np.array([[1, 2], [1, 4], [1, 0],
#...               [10, 2], [10, 4], [10, 0]])
kmeans = KMeans(n_clusters= 4, random_state=0).fit(data)
kmeans.labels_
kmeans.cluster_centers_


# In[79]:


distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(data)
    kmeanModel.fit(data)
    distortions.append(sum(np.min(cdist(data, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / data.shape[0])
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
    


# In[81]:



# generate random floating point values
from numpy.random import seed
from numpy.random import rand
# seed random number generator
seed(1)
# generate random numbers between 0-1
values_x = rand(100)
values_y = rand(100)
#print(values_x*20, values_y*40)
plt.plot(values_x, values_y, 'x')
plt.axis('equal')
plt.show()
combined = np.vstack((values_x, values_y)).T
kmeans = KMeans(n_clusters= 2, random_state=0).fit(combined )
kmeans.labels_
#kmeans.cluster_centers_


# In[80]:


distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(combined)
    kmeanModel.fit(data)
    distortions.append(sum(np.min(cdist(combined, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / data.shape[0])
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
    

