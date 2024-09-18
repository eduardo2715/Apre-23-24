import matplotlib.pyplot as plt
from sklearn import metrics, datasets, tree
from sklearn.model_selection import train_test_split

# 1. load 

data = datasets.load_breast_cancer()
X,y= data.data, data.target
from sklearn import datasets, metrics, cluster, mixture

# parameterize clustering
kmeans_algo = cluster.KMeans(n_clusters=2,algorithm='lloyd',init='random',n_init=1)

# learn the model
kmeans_model = kmeans_algo.fit(X)

# return centroids
kmeans_model.cluster_centers_
labels = kmeans_model.labels_
print("means:\n",kmeans_model.cluster_centers_)
# compute silhouette
print("Silhouette:",metrics.silhouette_score(X, labels, metric='euclidean'))
from sklearn.metrics import davies_bouldin_score
print("Davies Bouldin:",davies_bouldin_score(X, labels))
from sklearn.mixture import GaussianMixture

# learn EM with multivariate Gaussian assumption
em_algo = GaussianMixture(n_components=2, covariance_type='full',n_init=1) 
em_model = em_algo.fit(X)

# describe EM solution
print("means:\n",em_model.means_,"\n\ncovariances:\n",em_model.covariances_)
prob=em_model.predict_proba(X)
prob[5]
labels_em= em_model.predict(X)
print("Silhouette:",metrics.silhouette_score(X, labels_em, metric='euclidean'))
print("Davies Bouldin:",davies_bouldin_score(X, labels_em))
from sklearn.decomposition import PCA

# learn the transformation (components as linear combination of features)
pca = PCA(n_components=2)
X_pca = pca.fit(X).transform(X)
print("Components:\n",pca.components_)
plt.scatter(X_pca[:,0], X_pca[:,1],c=y)
plt.show()
X_pca = pca.transform(X)


# learn the model
kmeans_model = kmeans_algo.fit(X_pca)

# return centroids
kmeans_model.cluster_centers_
labels = kmeans_model.labels_
print("means:\n",kmeans_model.cluster_centers_)
# compute silhouette
print("Silhouette:",metrics.silhouette_score(X_pca, labels, metric='euclidean'))
plt.scatter(X_pca[:,0], X_pca[:,1],c=labels)
plt.show()
em_model = em_algo.fit(X_pca)
labels_em= em_model.predict(X_pca)
# compute silhouette
print("Silhouette:",metrics.silhouette_score(X_pca, labels_em, metric='euclidean'))

plt.scatter(X_pca[:,0], X_pca[:,1],c=labels_em)
plt.show()
