from scipy.io.arff import loadarff
import matplotlib.pyplot as plt
from sklearn import metrics, datasets, tree
from sklearn.model_selection import train_test_split
import pandas as pd, numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_selection import f_classif
import seaborn as sns

# Reading the ARFF file
data = loadarff('column_diagnosis.arff')
df = pd.DataFrame(data[0])
df['class'] = df['class'].str.decode('utf-8')
#print(df)

### Exercise 1 ###

features = df.drop('class', axis=1)
target = df['class']

fvalues, pvalues = f_classif(features, target)
columns = df.columns

for i in range(6):
    print('the f-value and the p-value of', df.columns[i], 'is:', fvalues[i], 'and', pvalues[i], 'respectively')

# the greater the f-value the better
# the lower the p-value the better

# degree_spondylolisthesis is the variable with higher discriminative power.
# pelvic_radius is the variable with lower discriminative power.

new_features = df.drop('class', axis=1)
new_features = df.drop('pelvic_incidence', axis=1)
new_features = df.drop('pelvic_tilt', axis=1)
new_features = df.drop('lumbar_lordosis_angle', axis=1)
new_features = df.drop('sacral_slope', axis=1)                     

hernia = df[df['class'] == 'Hernia']
spondylolisthesis = df[df['class'] == 'Spondylolisthesis']
normal = df[df['class'] == 'Normal']

#Graphic1
sns.kdeplot(hernia['degree_spondylolisthesis'], label= 'Hernia', fill = True)
sns.kdeplot(spondylolisthesis['degree_spondylolisthesis'], label= 'Spondylolisthesis', fill = True)
sns.kdeplot(normal['degree_spondylolisthesis'], label= 'Normal', fill = True)
plt.xlabel('degree_spondylolisthesis')
plt.ylabel('Densidade de Probabilidade')
plt.xlim(-20, 150)
plt.legend()
plt.title('Funções de Densidade de Probabilidade para degree_spondylolisthesis')
plt.show()

#Graphic2
sns.kdeplot(hernia['pelvic_radius'], label= 'Hernia', fill = True)
sns.kdeplot(spondylolisthesis['pelvic_radius'], label= 'Spondylolisthesis', fill = True)
sns.kdeplot(normal['pelvic_radius'], label= 'Normal', fill = True)
plt.xlabel('pelvic_radius')
plt.ylabel('Densidade de Probabilidade')
plt.xlim(50, 250)
plt.legend()
plt.title('Funções de Densidade de Probabilidade para pelvic_radius')
plt.show()

# #Final Graphic
# sns.pairplot(df, hue='class', height=2)
# plt.show()


### Exercise 2 ###
#variables = df.drop('class', axis= 1)
#target = df['class']

depth_limits = [1, 2, 3, 4, 5, 6, 8, 10]

variables_train, variables_test, target_train, target_test= train_test_split(variables, target, 
                                                                             train_size=0.7, stratify=target, random_state=0)

final_acc1, final_acc2 = np.array([]), np.array([])
std1, std2 = np.array([]), np.array([])

for depth in depth_limits:
    acc_folder1, acc_folder2 = np.array([]), np.array([])
    for i in range(len(depth_limits)):
        tree = DecisionTreeClassifier(criterion='gini', max_depth=depth)
        tree.fit(variables_train, target_train)
        
        y_pred1 = tree.predict(variables_test)
        y_pred2 = tree.predict(variables_train)
        
        acc_folder2 = np.append(acc_folder2, round(metrics.accuracy_score(target_train, y_pred2),2))
        acc_folder1 = np.append(acc_folder1, round(metrics.accuracy_score(target_test, y_pred1),2))
    
    final_acc1 = np.append(final_acc1, np.mean(np.array(acc_folder1)))
    final_acc2 = np.append(final_acc2, np.mean(np.array(acc_folder2)))
    std1 =np.append(std1, np.std(acc_folder1))
    std2 =np.append(std2, np.std(acc_folder2))
    
print("accuracy test list:", final_acc1)
print("accuracy train list:", final_acc2)

#print('std1',std1)
#Graphics
plt.figure(figsize=(12,7))
#tenho de por pontos ou não?
plt.errorbar(depth_limits, final_acc1, yerr=std1, label='Test Accuracy', color='#008080')
plt.errorbar(depth_limits, final_acc2, yerr=std2, label='Train Accuracy', color='red')
plt.title("Accuracy vs Depth")
plt.ylabel("Training Accuracy")
plt.xlabel("Depth")
plt.ylim(0.73, 1.03)
plt.legend() #por legenda a esquerda
plt.show()

# ### Exercise 3 ###
# x_train = df.drop('class', axis= 1)
# y_train = df['class']

# tree4 = DecisionTreeClassifier(criterion='gini', min_samples_leaf=20, random_state=0)
# tree4.fit(x_train, y_train)
# target_pred = tree4.predict(x_train)

# print('accuracy:', round(metrics.accuracy_score(y_train, target_pred), 2))

# class_names = ['Hernia', 'Spondylolisthesis', 'Normal']

# plot = tree.plot_tree(tree4, filled=True, feature_names=x_train.columns, class_names=tree4.classes_)
# plt.show()


