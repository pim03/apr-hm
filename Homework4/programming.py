import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, metrics, cluster, mixture
from scipy.io.arff import loadarff
from sklearn.preprocessing import MinMaxScaler

# Reading the ARFF file
data = loadarff('column_diagnosis.arff')
df = pd.DataFrame(data[0])
df['class'] = df['class'].str.decode('utf-8')

# Separate input from output data 
features = df.drop('class', axis=1)
target = df['class']

# Normalize data using sklearn's minmax scaler
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

########## Exercise 1 ##########

# Apply k-means clustering with k=2,3,4,5
kmeans2 = cluster.KMeans(n_clusters=2, random_state=0).fit(features_scaled)
kmeans3 = cluster.KMeans(n_clusters=3, random_state=0).fit(features_scaled)
kmeans4 = cluster.KMeans(n_clusters=4, random_state=0).fit(features_scaled)
kmeans5 = cluster.KMeans(n_clusters=5, random_state=0).fit(features_scaled)

y_pred2 = kmeans2.labels_
y_pred3 = kmeans3.labels_
y_pred4 = kmeans4.labels_
y_pred5 = kmeans5.labels_

# Assess silhouette and purity scores for each clustering
silhouette2 = metrics.silhouette_score(features_scaled, y_pred2)
silhouette3 = metrics.silhouette_score(features_scaled, y_pred3)
silhouette4 = metrics.silhouette_score(features_scaled, y_pred4)
silhouette5 = metrics.silhouette_score(features_scaled, y_pred5)

def purity_score(y_true, y_pred):
    # compute contingency/confusion matrix
    confusion_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(confusion_matrix, axis=0)) / np.sum(confusion_matrix) 

purity2 = purity_score(target, y_pred2)
purity3 = purity_score(target, y_pred3)
purity4 = purity_score(target, y_pred4)
purity5 = purity_score(target, y_pred5)

#Plot Silhouette
silhouettes = [silhouette2, silhouette3, silhouette4, silhouette5]
plt.plot([2,3,4,5], silhouettes, 'o-')
plt.title('Silhouette scores for k-means clustering')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette score')
plt.savefig('ex1_silhouette.png')
plt.show()

#Plot Purity
purities = [purity2, purity3, purity4, purity5]
plt.plot([2,3,4,5], purities, 'o-')
plt.title('Purity scores for k-means clustering')
plt.xlabel('Number of clusters')
plt.ylabel('Purity score')
plt.savefig('ex1_purity.png')
plt.show()



