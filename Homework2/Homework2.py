import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from scipy import stats
from scipy.io.arff import loadarff
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier


# Reading the ARFF file
data = loadarff('Homework2/column_diagnosis.arff')
df = pd.DataFrame(data[0])
df['class'] = df['class'].str.decode('utf-8')

# Separate input from output data 
features = df.drop('class', axis=1)
target = df['class']


### Exercise 1 ###

#a)

acc_folds_gauss = []
acc_folds_knn = []
folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

# Gaussian Naive Bayes
gaussNB = GaussianNB()

# KNN
knn_predictor = KNeighborsClassifier(n_neighbors=5)

for train_k, test_k in folds.split(features, target):
    X_train, X_test = features.iloc[train_k], features.iloc[test_k]
    y_train, y_test = target.iloc[train_k], target.iloc[test_k]
    
    gaussNB.fit(X_train, y_train)
    y_pred_gauss = gaussNB.predict(X_test)
    acc_folds_gauss.append(round(metrics.accuracy_score(y_test, y_pred_gauss),2))
    
    knn_predictor.fit(X_train, y_train)
    y_pred_knn = knn_predictor.predict(X_test)
    acc_folds_knn.append(round(metrics.accuracy_score(y_test, y_pred_knn),2))

print("Fold accuracies GaussianNB:", acc_folds_gauss)
print("Fold accuracies kNN:", acc_folds_knn)

plt.boxplot([acc_folds_gauss, acc_folds_knn], labels=['GaussianNB', 'kNN'])
plt.title('Accuracies for GaussianNB and kNN')
plt.xlabel('Classifiers')
plt.ylabel('Accuracy')
plt.savefig('ex1a_boxplot.png')
plt.show()

#b)
#Null Hypothesis: kNN is not statistically superior to GaussianNB
hypothesis = stats.ttest_rel(acc_folds_knn, acc_folds_gauss, alternative='greater')

if hypothesis[1] < 0.05:
    print("The null hypothesis is rejected and kNN is statistically superior to GaussianNB")
    
else:
    print("The null hypothesis is not rejected and there is no statistical superiority between kNN and GaussianNB")
    

############ Exercise 2 ############

print('--------------Exercise 2-----------------')


cum_conf_matrix1 = np.zeros((3,3))
cum_conf_matrix5 = np.zeros((3,3))

for train_k, test_k in folds.split(features, target):
    X_train, X_test = features.iloc[train_k], features.iloc[test_k]
    y_train, y_test = target.iloc[train_k], target.iloc[test_k]
    
    knn1 = KNeighborsClassifier(n_neighbors=1,weights='uniform',metric='euclidean')
    knn5 = KNeighborsClassifier(n_neighbors=5,weights='uniform',metric='euclidean')

    knn1.fit(X_train, y_train)
    knn5.fit(X_train, y_train)

    y_pred1 = knn1.predict(X_test)
    y_pred5 = knn5.predict(X_test)

    conf_matrix1 = confusion_matrix(y_test, y_pred1)
    conf_matrix5 = confusion_matrix(y_test, y_pred5)

    cum_conf_matrix1 += conf_matrix1
    cum_conf_matrix5 += conf_matrix5

conf_matrix_diff = cum_conf_matrix1 - cum_conf_matrix5

confusion1 = pd.DataFrame(conf_matrix_diff, index=knn1.classes_, columns=['Predicted Hernia', 'Predicted Normal', 'Predicted Spondylolisthesis'])

plt.figure(figsize=(10, 5))
heatmap = plt.imshow(conf_matrix_diff,cmap="coolwarm", interpolation='nearest')
plt.title('Differences between the two cumulative confusion matrices (k1 - k5)')
plt.xlabel('Predicted label')
plt.xticks([0, 1, 2], ['Hernia', 'Normal', 'Spondylolisthesis'])
plt.yticks([0, 1, 2], ['Hernia', 'Normal', 'Spondylolisthesis'])
plt.ylabel('True label')

cbar = plt.colorbar(heatmap)
cbar.set_label('Difference Magnitude', rotation=90)

for i in range(conf_matrix_diff.shape[0]):
    for j in range(conf_matrix_diff.shape[1]):
        plt.text(j, i, str(int(conf_matrix_diff[i, j])), ha='center', va='center', color='black')
plt.savefig('ex2_cummatrix.png')
plt.show()


############ Exercise 3 ############

print('--------------Exercise 3-----------------')

#Histograms for each feature:
features.hist(figsize=(10,10),density=True)
plt.savefig('ex3_1_hist.png')
plt.show()

#3. The dataset is not balanced, which can lead to a bias in the classifier.
df['class'].value_counts()
print(df['class'].value_counts())


#4. Check if variables are independent: correlation matrix
df = df.drop('class', axis=1)
df.corr(method='pearson')
sns.heatmap(df.corr(method='pearson'), annot=True)
plt.savefig('ex3_3_coormatrix.png')
plt.show()







    
    
    