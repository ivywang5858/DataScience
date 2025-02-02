# Created by ivywang at 2025-01-18
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
from sklearn.cluster import KMeans
import os

current_path = os.path.abspath(os.path.dirname(__file__))

# Unsupervised Learning - Cluster
# Cluster analysis is used to explore the data and identify patterns
# Market Segmentation
# Image Segmentation

# Classification: predicting an output category, given input data
# Clustering: grouping data points together based on similarities among them and difference from others

# Euclidean distance: distance between two data points
# Centroid: the mean position of a group of points (the center of the mass)

# K-means Clustering
# 1. Choose the number of clusters
# 2. Specify the cluster seeds (the starting centroid)
# 3. Assign each point to a centroid (based on their Euclidean distance from the seeds)
# 4. Adjust the centroids
# Repeat #3 and #4

# Clustering is about minimizing the distance between points in a clusters and maximizing the distance between clusters
# Distance between points in a cluster: 'within-cluster sum of squares', or WCSS
# If we minimize WCSS, we have reached the perfect clustering solution

# The elbow method: used to determine k

# The pros and cons of k means clustering
# Pros:
# 1. Simple to understand
# 2. Fast to cluster
# 3. Widely available
# 4. Easy to implement
# 5. Always yields a result (also a con, as it may be deceiving)

# Cons & Remedies:
# 1. We need to pick k; can be fixed with the elbow method
# 2. Sensitive to initialization (initialization matters); k-means++ which used by sklearn
# 3. Sensitive to outliers; remove outliers
# 4. Produces spherical solutions
# 5. Standardization


def K_Means_Clustering():
    # Load the data
    FILE_NAME = "Cluster_Countries_exercise.csv"
    raw_data = pd.read_csv(f"{current_path+'/Data/'}{FILE_NAME}")
    # print(raw_data.head())
    data = raw_data.copy()
    fig, axs = plt.subplots(2,1)
    axs[0].scatter(data['Longitude'], data['Latitude'])
    plt.xlim(-180,180)
    plt.ylim(-90,90)

    x = data.iloc[:,1:3]
    kmeans = KMeans(7)
    kmeans.fit(x)
    identified_clusters = kmeans.fit_predict(x)
    data_with_clusters = data.copy()
    data_with_clusters['Cluster'] = identified_clusters
    axs[1].scatter(data['Longitude'], data['Latitude'],c=data_with_clusters['Cluster'],cmap='rainbow')
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    plt.show()
    return

def K_Means_Categorical():
    # Load the data
    FILE_NAME = "Cluster_Categorical.csv"
    raw_data = pd.read_csv(f"{current_path + '/Data/'}{FILE_NAME}")
    data = raw_data.copy()
    print(data.head())
    data_mapped = data.copy()
    data_mapped['continent'] = data_mapped['continent'].map(
        {'North America': 0, 'Europe': 1, 'Asia': 2, 'Africa': 3, 'South America': 4, 'Oceania': 5,
         'Seven seas (open ocean)': 6, 'Antarctica': 7})
    x = data_mapped.iloc[:,3:4]
    kmeans = KMeans(4)
    kmeans.fit(x)
    identified_clusters = kmeans.fit_predict(x)
    print(identified_clusters)
    data_with_clusters = data_mapped.copy()
    data_with_clusters['Cluster'] = identified_clusters
    plt.scatter(data['Longitude'], data['Latitude'], c=data_with_clusters['Cluster'], cmap='rainbow')
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    plt.show()
    return

def Choose_Num_of_Cluster():
    # Load the data
    FILE_NAME = "Cluster_Countries_exercise.csv"
    raw_data = pd.read_csv(f"{current_path + '/Data/'}{FILE_NAME}")
    data = raw_data.copy()
    fig,axs = plt.subplots(2,1)
    axs[0].scatter(data['Longitude'], data['Latitude'])
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    x = data.iloc[:, 1:3]
    kmeans = KMeans(4)
    kmeans.fit(x)
    identified_clusters = kmeans.fit_predict(x)
    data_with_clusters = data.copy()
    data_with_clusters['Cluster'] = identified_clusters
    axs[1].scatter(data['Longitude'], data['Latitude'], c=data_with_clusters['Cluster'], cmap='rainbow')
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    plt.show()

    # WCSS - 'within-cluster sum of squares'
    kmeans.inertia_
    wcss = []
    cl_num = 11
    for i in range(1,cl_num):
        kmeans=KMeans(i)
        kmeans.fit(x)
        wcss_iter = kmeans.inertia_
        wcss.append(wcss_iter)

    # The Elbow Method
    number_clusters = range(1, cl_num)
    plt.plot(number_clusters, wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Within-cluster Sum of Squares')
    plt.show()

    # According to the elbow method, k would be 2 or 3
    kmeans = KMeans(2)
    kmeans.fit(x)
    identified_clusters = kmeans.fit_predict(x)
    data_with_clusters = data.copy()
    data_with_clusters['Cluster'] = identified_clusters
    fig,axs = plt.subplots(2,1)
    axs[0].scatter(data['Longitude'], data['Latitude'], c=data_with_clusters['Cluster'], cmap='rainbow')
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)

    kmeans = KMeans(3)
    kmeans.fit(x)
    identified_clusters = kmeans.fit_predict(x)
    data_with_clusters = data.copy()
    data_with_clusters['Cluster'] = identified_clusters
    axs[1].scatter(data['Longitude'], data['Latitude'], c=data_with_clusters['Cluster'], cmap='rainbow')
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    plt.show()
    return

def Cluster_Analysis():
    # Load the data
    FILE_NAME = "Cluster_iris_dataset.csv"
    data = pd.read_csv(f"{current_path + '/Data/'}{FILE_NAME}")
    # create a scatter plot based on two corresponding features (sepal_length and sepal_width; OR petal_length and petal_width)
    fig,axs = plt.subplots(3,1)
    axs[0].scatter(data['sepal_length'], data['sepal_width'])
    # name your axes
    plt.xlabel('Lenght of sepal')
    plt.ylabel('Width of sepal')
    # create a variable which will contain the data for the clustering
    x = data.copy()
    # create a k-means object with 2 clusters
    kmeans = KMeans(2)
    # fit the data
    kmeans.fit(x)
    # create a copy of data, so we can see the clusters next to the original data
    clusters = data.copy()
    # predict the cluster for each observation
    clusters['cluster_pred'] = kmeans.fit_predict(x)
    # create a scatter plot based on two corresponding features (sepal_length and sepal_width; OR petal_length and petal_width)
    axs[1].scatter(clusters['sepal_length'], clusters['sepal_width'], c=clusters['cluster_pred'], cmap='rainbow')
    # plt.show()

    # Standardize the variables
    # import some preprocessing module
    from sklearn import preprocessing

    # scale the data for better results
    x_scaled = preprocessing.scale(data)
    # create a k-means object with 2 clusters
    kmeans_scaled = KMeans(2)
    # fit the data
    kmeans_scaled.fit(x_scaled)
    # create a copy of data, so we can see the clusters next to the original data
    clusters_scaled = data.copy()
    # predict the cluster for each observation
    clusters_scaled['cluster_pred'] = kmeans_scaled.fit_predict(x_scaled)
    # create a scatter plot based on two corresponding features (sepal_length and sepal_width; OR petal_length and petal_width)
    axs[2].scatter(clusters_scaled['sepal_length'], clusters_scaled['sepal_width'], c=clusters_scaled['cluster_pred'],
                cmap='rainbow')
    plt.show()

    # WCSS
    wcss = []
    # 'cl_num' is a that keeps track the highest number of clusters we want to use the WCSS method for.
    # We have it set at 10 right now, but it is completely arbitrary.
    cl_num = 10
    for i in range(1, cl_num):
        kmeans = KMeans(i)
        kmeans.fit(x_scaled)
        wcss_iter = kmeans.inertia_
        wcss.append(wcss_iter)
    # Apply the elbow method
    number_clusters = range(1, cl_num)
    plt.plot(number_clusters, wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Within-cluster Sum of Squares')
    plt.show()
    # It seems like 2 or 3-cluster solutions are the best.
    return

# K_Means_Clustering()
# K_Means_Categorical()
# Choose_Num_of_Cluster()
Cluster_Analysis()




