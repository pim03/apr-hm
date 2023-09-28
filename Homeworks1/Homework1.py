from scipy.io.arff import loadarff
import matplotlib.pyplot as plt
from sklearn import metrics, tree
from sklearn.model_selection import train_test_split
import pandas as pd, numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_classif
import seaborn as sns

# Reading the ARFF file
data = loadarff('column_diagnosis.arff')
df = pd.DataFrame(data[0])
df['class'] = df['class'].str.decode('utf-8')
#print(df.head())

### Exercise 1 ###

variables = df.drop('class', axis= 1)
target = df['class']

fvalues, pvalues = f_classif(variables, target)

fvalue_df = pd.DataFrame({'variable': variables.columns, 'fvalues': fvalues, 'pvalues': pvalues})
print(fvalue_df.head())

'''the greater the f-value the better
the lower the p-value the better

degree_spondylolisthesis is the variable with higher discriminative power.
pelvic_radius is the variable with lower discriminative power.'''

hernia = df[df['class'] == 'Hernia']
spondylolisthesis = df[df['class'] == 'Spondylolisthesis']
normal = df[df['class'] == 'Normal']

#Graphic1
plt.figure(figsize=(12,7))
sns.kdeplot(hernia['degree_spondylolisthesis'], label= 'Hernia', fill = True)
sns.kdeplot(spondylolisthesis['degree_spondylolisthesis'], label= 'Spondylolisthesis', fill = True)
sns.kdeplot(normal['degree_spondylolisthesis'], label= 'Normal', fill = True)
plt.xlabel('degree_spondylolisthesis')
plt.ylabel('Densidade de Probabilidade')
plt.legend()
plt.title('Funções de Densidade de Probabilidade para degree_spondylolisthesis')
plt.show()

#Graphic2
plt.figure(figsize=(12,7))
sns.kdeplot(hernia['pelvic_radius'], label= 'Hernia', fill = True)
sns.kdeplot(spondylolisthesis['pelvic_radius'], label= 'Spondylolisthesis', fill = True)
sns.kdeplot(normal['pelvic_radius'], label= 'Normal', fill = True)
plt.xlabel('pelvic_radius')
plt.ylabel('Densidade de Probabilidade')
plt.legend()
plt.title('Funções de Densidade de Probabilidade para pelvic_radius')
plt.show()

#Final Graphic
#sns.pairplot(df, hue='class', height=2)
#plt.show()


### Exercise 2 ###

depth_limits = [1, 2, 3, 4, 5, 6, 8, 10]

variables_train, variables_test, target_train, target_test= train_test_split(variables, target, 
                                                                             train_size=0.7, stratify=target, random_state=0)

final_acc1, final_acc2 = np.array([]), np.array([])
std1, std2 = np.array([]), np.array([])

for depth in depth_limits:
    acc_folder1, acc_folder2 = np.array([]), np.array([])
    for i in range(10):
        tree2 = DecisionTreeClassifier(criterion='gini', max_depth=depth)
        tree2.fit(variables_train, target_train)
        
        y_pred1 = tree2.predict(variables_test)
        y_pred2 = tree2.predict(variables_train)
        
        acc_folder2 = np.append(acc_folder2, round(metrics.accuracy_score(target_train, y_pred2),2))
        acc_folder1 = np.append(acc_folder1, round(metrics.accuracy_score(target_test, y_pred1),2))
    
    final_acc1 = np.append(final_acc1, np.mean(np.array(acc_folder1)))
    final_acc2 = np.append(final_acc2, np.mean(np.array(acc_folder2)))
    std1 =np.append(std1, np.std(acc_folder1))
    std2 =np.append(std2, np.std(acc_folder2))
    
print("accuracy test list:", final_acc1)
print("accuracy train list:", final_acc2)


#Graphics
plt.figure(figsize=(12,7))
plt.errorbar(depth_limits, final_acc1, yerr=std1, label='Test Accuracy', color='#008080')
plt.errorbar(depth_limits, final_acc2, yerr=std2, label='Train Accuracy', color='red')
plt.plot(depth_limits, final_acc1, 'o', color='#008080', markersize=3)
plt.plot(depth_limits, final_acc2, 'o', color='red', markersize=3)
plt.title("Accuracy vs Depth")
plt.ylabel("Accuracy")
plt.xlabel("Depth")
plt.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0)) #move the legend to the left
plt.show() 

### Exercise 3 ###

tree4 = DecisionTreeClassifier(criterion='gini', min_samples_leaf=20, random_state=0)
tree4.fit(variables, target)
target_pred = tree4.predict(variables)

print('accuracy:', round(metrics.accuracy_score(target, target_pred), 2))

tree.plot_tree(tree4, filled=True, feature_names=variables.columns, class_names=tree4.classes_)
plt.title("Tree with min_samples_leaf=20")
plt.show()

