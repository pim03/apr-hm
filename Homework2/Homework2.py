import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats
from scipy.io.arff import loadarff
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn import metrics, datasets, tree
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.feature_selection import f_classif
import graphviz
from sklearn.neighbors import KNeighborsClassifier


# Reading the ARFF file
data = loadarff('column_diagnosis.arff')
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

# iterate per fold
for train_k, test_k in folds.split(features, target):
    X_train, X_test = features.iloc[train_k], features.iloc[test_k]
    y_train, y_test = target.iloc[train_k], target.iloc[test_k]
    
    ## train and assess
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
plt.show()

#b)

hypothesis = stats.ttest_rel(acc_folds_knn, acc_folds_gauss, alternative='greater')

if hypothesis[1] < 0.05:
    print("The null hypothesis is rejected and kNN is statistically superior to GaussianNB")
    
else:
    print("The null hypothesis is not rejected and there is no statistical superiority between kNN and GaussianNB")
    
# ### Exercise 2 ###

# knn1 = KNeighborsClassifier(n_neighbors=1, weights='uniform', metric='euclidean')
# knn5 = KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='euclidean')

# cumulative_tp1 = 0
# cumulative_tn1 = 0
# cumulative_fp1 = 0
# cumulative_fn1 = 0

# cumulative_fn5 = 0
# cumulative_tp5 = 0
# cumulative_tn5 = 0
# cumulative_fp5 = 0

# # iterate per fold
# for train_k, test_k in folds.split(features, target):
#     X_train, X_test = features.iloc[train_k], features.iloc[test_k]
#     y_train, y_test = target.iloc[train_k], target.iloc[test_k]
    
#     knn1.fit(X_train, y_train)
#     knn1_pred = knn1.predict(X_test)
    
#     knn5.fit(X_train, y_train)
#     knn5_pred = knn5.predict(X_test)
    
#     cm1 = np.array(confusion_matrix(y_test, knn1_pred))
#     cumulative_tp1 += cm1[1, 1]
#     cumulative_fp1 += cm1[0, 1]
#     cumulative_fn1 += cm1[1, 0]
#     cumulative_tn1 += cm1[0, 0]
    
#     cm5 = np.array(confusion_matrix(y_test, knn5_pred))
#     cumulative_tp5 += cm5[1, 1]
#     cumulative_fp5 += cm5[0, 1]
#     cumulative_fn5 += cm5[1, 0]
#     cumulative_tn5 += cm5[0, 0]
    
#     confusion1 = pd.DataFrame(cm1, index=knn1.classes_, columns=['Predicted Hernia', 'Predicted Normal', 'Predicted Spondylolisthesis'])
#     confusion5 = pd.DataFrame(cm5, index=knn5.classes_, columns=['Predicted Hernia', 'Predicted Normal', 'Predicted Spondylolisthesis'])

# print("Confusion matrix for kNN with k=1:\n", confusion1, '\n')
# print("Confusion matrix for kNN with k=5:\n", confusion5)   

# plt.figure(figsize=(10, 5))
# plt.imshow(confusion1-confusion5, cmap='Blues', interpolation='nearest')
# plt.title('Differences between the two cumulative confusion matrices')
# plt.xlabel('Predicted label')
# plt.xticks([0, 1, 2], ['Hernia', 'Normal', 'Spondylolisthesis'])
# plt.yticks([0, 1, 2], ['Hernia', 'Normal', 'Spondylolisthesis'])
# plt.ylabel('True label') 
# plt.show()



############ Exercise 2 ############

# Load your dataset and split it into training and testing sets
# X_train, X_test, y_train, y_test = ...

# Train kNN classifiers with k=1 and k=5
knn1 = KNeighborsClassifier(n_neighbors=1,uniform=True,metric='euclidean')
knn5 = KNeighborsClassifier(n_neighbors=5,uniform=True,metric='euclidean')

knn1.fit(X_train, y_train)
knn5.fit(X_train, y_train)

# Make predictions
y_pred1 = knn1.predict(X_test)
y_pred5 = knn5.predict(X_test)

# Calculate confusion matrices
conf_matrix1 = confusion_matrix(y_test, y_pred1)
conf_matrix5 = confusion_matrix(y_test, y_pred5)

# Calculate cumulative confusion matrices
cumulative_conf_matrix1 = np.cumsum(conf_matrix1, axis=0)
cumulative_conf_matrix5 = np.cumsum(conf_matrix5, axis=0)

# Calculate the difference between cumulative confusion matrices
conf_matrix_diff = cumulative_conf_matrix1 - cumulative_conf_matrix5

# Visualize the differences using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_diff, annot=True, fmt="d", cmap="coolwarm", cbar=True)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Difference in Cumulative Confusion Matrices (k=1 - k=5)")
plt.show()



    




    
    
    