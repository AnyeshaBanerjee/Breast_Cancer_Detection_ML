#!/usr/bin/env python
# coding: utf-8

# In[1]:


import mglearn
import numpy as np


# In[2]:


X, y = mglearn.datasets.make_forge()


# In[3]:


X.shape


# In[4]:


X[:,0]


# In[5]:


import matplotlib.pyplot as plt


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


plt.scatter(X[:, 0], X[:, 1],c=y, s=60, cmap=mglearn.cm2)


# In[8]:


X, y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X, y, 'o')
plt.plot(X, -3 * np.ones(len(X)), 'o')
plt.ylim(-3.1, 3.1)


# <h1>Breast Cancer DataSet</h1>

# In[9]:


from sklearn.datasets import load_breast_cancer


# In[10]:


cancer = load_breast_cancer()


# In[11]:


cancer.keys()


# In[12]:


cancer['data'][:1]


# In[13]:


cancer['feature_names']


# In[14]:


print(cancer.target_names)
np.bincount(cancer.target)


# In[15]:


cancer['target'].shape


# In[18]:


plt.figure(figsize=(16,6))
mglearn.plots.plot_knn_classification(n_neighbors=1)
plt.title("forge_one_neighbor");


# In[21]:


plt.figure(figsize=(8,6))
mglearn.plots.plot_knn_classification(n_neighbors=3)
plt.title("forge_one_neighbor");


# In[22]:


from sklearn.model_selection import train_test_split
X, y = mglearn.datasets.make_forge()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# In[23]:


from sklearn.neighbors import KNeighborsClassifier


# In[24]:


modelmg = KNeighborsClassifier(n_neighbors=3)


# In[26]:


X_train.shape


# In[27]:


modelmg.fit(X_train,y_train)


# In[28]:


modelmg.predict(X_test)


# In[29]:


modelmg.score(X_test,y_test)


# In[32]:


fig, axes = plt.subplots(1, 3, figsize=(20, 6))
i = 0
for n_neighbors, ax in zip([1, 3, 9], axes):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
    ax.scatter(X[:, 0], X[:, 1], c=y, s=60, cmap=mglearn.cm2)
    ax.set_title("%d neighbor(s)" % n_neighbors)
    #plt.savefig('img%d.jpg'%(i))
    i += 1


# In[33]:


#create dataframe


# In[34]:


import pandas as pd


# In[35]:


df = pd.DataFrame(data=cancer['data'],columns=cancer['feature_names'])


# In[36]:


df['target'] = cancer['target']


# In[37]:


cancer.keys()


# In[38]:


df['target'].value_counts()


# In[39]:


X_train, X_test, y_train, y_test = train_test_split(df.drop('target',axis=1),df['target'], test_size=0.3, random_state=101)


# In[40]:


train_acc = []
test_acc = []

for i in range(1,10):
    modelCr = KNeighborsClassifier(n_neighbors=i)
    modelCr.fit(X_train,y_train)
    train_acc.append(modelCr.score(X_train,y_train))
    test_acc.append(modelCr.score(X_test,y_test))


# In[41]:


train_acc


# In[42]:


test_acc


# In[43]:


plt.figure(figsize=(8,6))
plt.ylim(0.88,1)
plt.xlim(1,9)
plt.plot(range(1,10),train_acc,'ro-',markersize=5,label='Training Acc')
plt.plot(range(1,10),test_acc,'o-',markersize=5,label='Testing Acc')

plt.legend()


# <h1>KNN Algo Done for Classification</h1>
