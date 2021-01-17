import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

df = pd.read_csv("p8.csv")

V1 = df["V1"].values
V2 = df["V2"].values
X = np.array([V1, V2]).T
print("X:", X)

plt.figure()
plt.scatter(V1, V2)
plt.title("Input data")

kmeans = KMeans(2, random_state=0)
kmeans_y = kmeans.fit_predict(X)
print("K-means labels:", kmeans_y)

centroids = kmeans.cluster_centers_
print("K-means centroids:", centroids)

plt.figure()
plt.scatter(V1, V2, c=kmeans_y)
plt.scatter(centroids[:, 0], centroids[:, 1], marker="*")
plt.title("Graph using Kmeans Algorithm")

gmm = GaussianMixture(2)
gmm_y = gmm.fit_predict(X)
print("GMM labels:", gmm_y)

plt.figure()
plt.scatter(V1, V2, c=gmm_y)
plt.title("Graph using EM Algorithm")

plt.show()
