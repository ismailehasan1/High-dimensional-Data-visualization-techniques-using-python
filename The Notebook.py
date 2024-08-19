#!/usr/bin/env python
# coding: utf-8

# # WELCOME TO THE NOTEBOOK
# ---
# In this notebook we are going to learn about high-dimensionl data and how to analyze it.<br>
# ##### High-dimensional data analysis tasks:
#     - Corellation Analysis 
#     - Outlier detection 
#     - Cluster Anlysis 
# 
# #### Project Agenda:
#     - Importing the Dataset 
#     - Normalization 
#     - K-means Clustering 
#     - Scatterplot Matrix 
#     - PCP
#     - Data Reduction
# 
# 

# Importing the module 

# In[1]:


import pandas as pd 

# clustering 
from sklearn.cluster import KMeans

# preprocessing 
from sklearn import preprocessing

# visualization
import plotly.express as px
import matplotlib.pyplot as plt

# Projections 
from sklearn.decomposition import PCA

print("Modules are imported.")


# Task 1: Loading the dataset

# In[13]:


data = pd.read_csv("Dataset.csv")
data.head()


# Let's check the data types 

# In[14]:


data.dtypes


# let's check a summary of our data

# In[15]:


data.describe()


# We want all the values between 0 and 1 

# In[24]:


min_max_scaler = preprocessing.MinMaxScaler()
x = data.drop("Country or region", axis = 1).values
x_scaled = min_max_scaler.fit_transform(x)

data[data.columns[1:]] = x_scaled
data.describe()


# In[37]:





# ### Now let's apply clustering 

# In[38]:



clustering_data = data.drop("Country or region", axis = 1)
kmeans = KMeans(n_clusters=3, max_iter = 300).fit(clustering_data)

data["class"] = kmeans.labels_.astype(str)

data["class"].value_counts()


# ### Scatter plot matrix: 
# 

# Let's define a function to show the Scatter plot Matrix

# In[39]:


data


# In[45]:


dims = data.drop(["Country or region","class"], axis = 1).columns
fig = px.scatter_matrix(data,dimensions=dims, color = data["class"], hover_name = data["Country or region"])

fig.update_traces(marker = dict(size = 4), diagonal_visible = False, showupperhalf = False )
fig.update_layout(width = 800, height = 800, title = "Happiness index Scatterplot Matrix", font_size = 7)
fig.show()


# ### What are our findings ? 

# #### Task we can solve using Scatter plot matrix: 
#         - corellation analysis 
#         - cluster analysis 
#         - outlier detection 
# ---

# ### Parallel coordinate plot: 
# 

# In[48]:


fig = px.parallel_coordinates(data, 
                              
                              color = data["class"].astype(int),
                              color_continuous_scale = px.colors.diverging.Tealrose
                             )
fig.show()


# In[55]:


data_sampled = data.sample(n=100)
data_sampled


# #### Data reduction
#     - reduce the data size -> sampling 
#     - reduce the dimensions -> UMAP 

# Sampling

# In[56]:


dims = data_sampled.drop(["Country or region","class"], axis = 1).columns
fig = px.scatter_matrix(data_sampled,dimensions=dims, color = data_sampled["class"], hover_name = data_sampled["Country or region"])

fig.update_traces(marker = dict(size = 4), diagonal_visible = False, showupperhalf = False )
fig.update_layout(width = 800, height = 800, title = "Happiness index Scatterplot Matrix", font_size = 7)
fig.show()


# scatter plot matrix of the sampled data

# In[ ]:





# #### PCA projection 

# let's check the dataset again

# In[57]:


data.head()


# let's apply PCA projection

# In[58]:


pca = PCA(2)
projected_data = pca.fit_transform(data.drop(["Country or region","class"], axis = 1))


# let's take a look at the projected data

# In[59]:


projected_data


# let's check the scatter plot of the projected data

# In[66]:


fig = px.scatter(projected_data[:,0],
                 projected_data[:,1],
                 color = data["class"],
                labels = {"index" : "d1", "X" : "d2"},
                hover_name = data["Country or region"])
                              
        
fig.show()


# In[ ]:




