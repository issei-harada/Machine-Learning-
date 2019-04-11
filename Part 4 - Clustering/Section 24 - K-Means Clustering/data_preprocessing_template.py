#Importing libraries
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd

dataset = pd.read_csv('Mall_Customers.csv')

X = dataset.iloc[:, [3,4]].values

from sklearn.cluster import KMeans
wcss = []

for i in range(1,11):
    cluster = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 20)
    cluster.fit(X)
    wcss.append(cluster.inertia_)

plt.plot(range(1,11),wcss)
plt.title('Elbow Method')

cluster = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 20)
y_kmean = cluster.fit_predict(X)

#X1 = Xaxis, X2 = yaxis, s=size
plt.scatter(X[y_kmean==0, 0], X[y_kmean==0, 1], s=100, c='red', label = 'Cluster0')
plt.scatter(X[y_kmean==1, 0], X[y_kmean==1, 1], s=100, c='blue', label = 'Cluster1')
plt.scatter(X[y_kmean==2, 0], X[y_kmean==2, 1], s=100, c='green', label = 'Cluster2')
plt.scatter(X[y_kmean==3, 0], X[y_kmean==3, 1], s=100, c='orange', label = 'Cluster3')
plt.scatter(X[y_kmean==4, 0], X[y_kmean==4, 1], s=100, c='purple', label = 'Cluster4')
plt.scatter(cluster.cluster_centers_[:, 0], cluster.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
