import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn import metrics

# Reading the csv file
df = pd.read_csv('Homework3/winequality-red.csv')
# Separating the variables from the target
variables = df.drop("quality", axis= 1)
target = df['quality']

variables_train, variables_test, target_train, target_test= train_test_split(variables, target, 
                                                                         train_size=0.8, stratify=target, random_state=0)

y_pred = np.zeros(len(target_test))
number_of_iterations = [20, 50, 100, 200]
first_rmse = 0

for seed in range(1,11):
    predictor = MLPRegressor(hidden_layer_sizes=(10,10), activation='relu', early_stopping=True, validation_fraction=0.2, random_state=seed)
    predictor.fit(variables_train,target_train)
    y_pred += predictor.predict(variables_test)

y_pred = y_pred/10
first_rmse = np.sqrt(np.mean((target_test - y_pred)**2))

print('ypred', y_pred)
print('the average RMSE is', first_rmse)

rmse_final = np.array([])

for iteration in number_of_iterations:
    y_pred2 = np.zeros(len(target_test))
    for seed in range(1,11):
        predictor = MLPRegressor(hidden_layer_sizes=(10,10), activation='relu', max_iter=iteration, random_state=seed)
        predictor.fit(variables_train,target_train)
        y_pred2 += predictor.predict(variables_test)

    y_pred2 = y_pred2/10
    rmse_final = np.append(rmse_final, np.sqrt(np.mean((target_test - y_pred2)**2)))

print('the average RMSE is', rmse_final)

def const(x):
    return first_rmse

plt.plot(number_of_iterations, rmse_final, '-o', label='RMSE')
plt.hlines(first_rmse, xmin=min(number_of_iterations), xmax=max(number_of_iterations), colors='r', linestyles='dashed')
plt.xlabel('Number of iterations') 
plt.ylabel('RMSE')
plt.title('RMSE vs number of iterations')
plt.show()



