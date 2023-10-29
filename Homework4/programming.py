import math
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics, cluster
from scipy.io.arff import loadarff
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

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

# ########## Exercise 1 ##########

# # Apply k-means clustering with k=2,3,4,5
# kmeans2 = cluster.KMeans(n_clusters=2, random_state=0).fit(features_scaled)
# kmeans3 = cluster.KMeans(n_clusters=3, random_state=0).fit(features_scaled)
# kmeans4 = cluster.KMeans(n_clusters=4, random_state=0).fit(features_scaled)
# kmeans5 = cluster.KMeans(n_clusters=5, random_state=0).fit(features_scaled)

# y_pred2 = kmeans2.labels_
# y_pred3 = kmeans3.labels_
# y_pred4 = kmeans4.labels_
# y_pred5 = kmeans5.labels_

# # Assess silhouette and purity scores for each clustering
# silhouette2 = metrics.silhouette_score(features_scaled, y_pred2)
# silhouette3 = metrics.silhouette_score(features_scaled, y_pred3)
# silhouette4 = metrics.silhouette_score(features_scaled, y_pred4)
# silhouette5 = metrics.silhouette_score(features_scaled, y_pred5)

# def purity_score(y_true, y_pred):
#     # compute contingency/confusion matrix
#     confusion_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
#     return np.sum(np.amax(confusion_matrix, axis=0)) / np.sum(confusion_matrix) 

# purity2 = purity_score(target, y_pred2)
# purity3 = purity_score(target, y_pred3)
# purity4 = purity_score(target, y_pred4)
# purity5 = purity_score(target, y_pred5)

# #Plot Silhouette
# silhouettes = [silhouette2, silhouette3, silhouette4, silhouette5]
# plt.plot([2,3,4,5], silhouettes, 'o-')
# plt.title('Silhouette scores for k-means clustering')
# plt.xlabel('Number of clusters')
# plt.ylabel('Silhouette score')
# plt.savefig('ex1_silhouette.png')
# plt.show()

# #Plot Purity
# purities = [purity2, purity3, purity4, purity5]
# plt.plot([2,3,4,5], purities, 'o-')
# plt.title('Purity scores for k-means clustering')
# plt.xlabel('Number of clusters')
# plt.ylabel('Purity score')
# plt.savefig('ex1_purity.png')
# plt.show()

### Exercise 1 ###

def purity_score(y_true, y_pred):
    # compute contingency/confusion matrix
    confusion_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(confusion_matrix, axis=0)) / np.sum(confusion_matrix) 

silhouettes = []
purities = []
for k in range(2, 6):
    kmeans_algo = cluster.KMeans(n_clusters=k, random_state=0 , n_init= 'auto')
    kmeans_model = kmeans_algo.fit(features_scaled)
    target_pred = kmeans_model.labels_
    purity = purity_score(target, target_pred)
    silhouette = metrics.silhouette_score(features_scaled, target_pred)
    silhouettes.append(silhouette)
    purities.append(purity)
    
    print("Purity score for k = " , str(k) , " is " , purity)
    print("Silhouette score for k = " , str(k) , " is " , silhouette)
    
#Plot Silhouette
plt.plot([2,3,4,5], silhouettes, 'o-')
plt.title('Silhouette scores for k-means clustering')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette score')
plt.savefig('ex1_silhouette.png')
plt.show()

#Plot Purity
plt.plot([2,3,4,5], purities, 'o-')
plt.title('Purity scores for k-means clustering')
plt.xlabel('Number of clusters')
plt.ylabel('Purity score')
plt.savefig('ex1_purity.png')
plt.show()
    
### Exercise 2 ###

pca = PCA(n_components=2)
pca.fit(features_scaled)
X_pca = pca.transform(features_scaled)

print("Components (eigenvectors):\n",pca.components_)
print("Explained variance (eigenvalues) =",pca.explained_variance_)
print("Explained variance (ratio) =",pca.explained_variance_ratio_)

xvector = pca.components_[0] * max(X_pca[:,0])
yvector = pca.components_[1] * max(X_pca[:,1])

columns = features.columns
impt_features1 = {columns[i] : math.sqrt(xvector[i]**2) for i in range(len(columns))}
sorted_features1 = sorted(zip(impt_features1.values(),impt_features1.keys()),reverse=True)
print('Features sorted by importance for the first component: \n')
for i in range(len(sorted_features1)):
    print(f'{sorted_features1[i][1]} : {sorted_features1[i][0]: .5f}')

impt_features2 = {columns[i] : math.sqrt(yvector[i]**2) for i in range(len(columns))}
sorted_features2 = sorted(zip(impt_features2.values(),impt_features2.keys()),reverse=True)
print('\nFeatures sorted by importance for the second component: \n')
for i in range(len(sorted_features2)):
    print(f'{sorted_features2[i][1]} : {sorted_features2[i][0]: .5f}')

impt_features = {columns[i] : math.sqrt(xvector[i]**2 + yvector[i]**2) for i in range(len(columns))}
sorted_features = sorted(zip(impt_features.values(),impt_features.keys()),reverse=True)
print('\nFeatures sorted by importance: \n')
for i in range(len(sorted_features)):
    print(f'{sorted_features[i][1]} : {sorted_features[i][0]: .5f}')

### Exercise 3 ###

# i)
plt.figure(figsize=(12,7))
plt.scatter(X_pca[target=='Normal', 0], X_pca[target=='Normal', 1], alpha=0.6, label='Normal')
plt.scatter(X_pca[target=='Hernia', 0], X_pca[target=='Hernia', 1], alpha=0.6, label='Hernia')
plt.scatter(X_pca[target=='Spondylolisthesis', 0], X_pca[target=='Spondylolisthesis', 1], alpha=0.6, label='Spondylolisthesis')

plt.legend()
plt.title('Points with Ground diagnosis')
plt.savefig('ex3_ground_diagnosis.png')
plt.show()

# ii)

kmeans_algo = cluster.KMeans(n_clusters=3, random_state=0)
kmeans_model = kmeans_algo.fit(features_scaled)
target_pred = kmeans_model.labels_

plt.figure(figsize=(12, 7))
plt.scatter(X_pca[:,0], X_pca[:,1], c=target_pred, alpha=0.6)

plt.legend()
plt.title('Cluster for k = 3')
plt.savefig('ex3_cluster.png')
plt.show()

# iii)

cluster_mapping = pd.DataFrame({'Cluster': target_pred, 'Class': target})

# Calculate the mode class for each cluster
cluster_mode = cluster_mapping.groupby('Cluster')['Class'].agg(lambda x: x.mode().iat[0])

plt.figure(figsize=(12, 7))
for cluster in set(target_pred):
    data = X_pca[target_pred == cluster]
    plt.scatter(data[:, 0], data[:, 1], label=f'Cluster {cluster}', alpha=0.6)

plt.title('K-means Clustering with k = 3 and labelled with the most frequent class') 

# Create a legend using the calculated mode class for each cluster
legend_labels = [f'Cluster {cluster+1}: {mode_class}' for cluster, mode_class in cluster_mode.items()]
plt.legend(legend_labels)
plt.savefig('ex3_cluster_labelled.png')
# Show the plot
plt.show()


'''legenda = {cluster : mode_class for cluster, mode_class in cluster_mode.items()}
# identify incorrect predictions (pre-matching is necessary)
#codes = {'Normal': 2, 'Hernia': 1, 'Spondylolisthesis': 0}

y_ref = []
for i in range(len(target)):
    if target[i] == legenda
    

y_ref = target.map(codes)
changes = [i for i in range(len(y_ref)) if y_ref[i] != target_pred.map(codes)[i]]
print("incorrect observations:",changes)

# plot incorrect predictions 
plt.plot(features.iloc[:,0], features.iloc[:,1], 'w', markerfacecolor='b', marker='.', markersize=10)
plt.plot(features.iloc[changes,0], features.iloc[changes,1], 'w', markerfacecolor='r', marker='.', markersize=10)
plt.show()'''