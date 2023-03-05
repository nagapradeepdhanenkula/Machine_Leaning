#!/usr/bin/env python
# coding: utf-8

# # Amyotrophic Lateral Sclerosis (ALS) Case-Study Cluster analysis

# In[109]:


#Importing the Libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


# In[110]:


#Reading the Dataset into Pandas


# In[111]:


training_df = pd.read_csv('ALS_TrainingData_2223.csv')
training_df.head(5)


# In[112]:


testing_df = pd.read_csv('ALS_TestingData_78.csv')
testing_df.head(5)


# In[113]:


# Deleting the Unnecessary Columns 


# In[114]:


del training_df['ID']
del training_df['SubjectID']


# In[115]:


training_df.head()


# In[116]:


del testing_df['ID']
del testing_df['SubjectID']
testing_df.head()


# In[117]:


# check No of rows and columns


# In[118]:


training_df.shape


# In[119]:


testing_df.shape


# In[120]:


#Check for null values


# In[121]:


training_df.isnull().values.any()


# In[122]:


testing_df.isnull().values.any()


# In[123]:


# Summarized Descriptive Statistics
training_df.describe()


# In[124]:


training_df.hist(figsize=(28,38), color='red')
plt.show()


# In[125]:


#Standard Scalar

from sklearn.preprocessing import StandardScaler

scaler_N = StandardScaler()

N = training_df['ALSFRS_slope']
P = pd.DataFrame(scaler.fit_transform(training_df.drop('ALSFRS_slope', axis=1)),columns = training_df.drop('ALSFRS_slope', axis=1).columns)
P


# In[126]:


#Elbow Method for finding the Number of Clusters


# In[127]:


#Elbow Method for finding the Number of Clusters
from sklearn.cluster import KMeans
sum_of_squared_distance = []
for i in range(1, 12):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(P)
    sum_of_squared_distance.append(kmeans.inertia_)
# Plotting Elbow Graph (Visualization)
plt.figure(figsize=(12,6))
sns.lineplot(sum_of_squared_distance,marker = 'o',color='red')
plt.title('The Elbow Method for clusters')
plt.xlabel('Number of clusters')
plt.ylabel('sum_of_squared_distance')
plt.show()


# In[128]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

range_n_clusters = [2, 3, 4, 5, 6]

for n_clusters in range_n_clusters:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(P) + (n_clusters + 1) * 10])

    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = clusterer.fit_predict(P)

    silhouette_avg = silhouette_score(P, cluster_labels)
    print( "For n_clusters =", n_clusters,"average of silhouette_score is :",silhouette_avg,)

    sample_silhouette_values = silhouette_samples(P, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),0,ith_cluster_silhouette_values,facecolor=color,edgecolor=color,alpha=0.7,)
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(P.iloc[:, 0], P.iloc[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k")

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0],centers[:, 1],marker="o",c="white",alpha=1,s=200,edgecolor="k",)

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle("Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"% n_clusters,fontsize=14,fontweight="bold",)

plt.show()


# In[ ]:




