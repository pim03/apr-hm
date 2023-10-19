import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

# Reading the csv file
df = pd.read_csv('Homework3/winequality-red.csv')
# Separating the variables from the target
variables = df.drop("quality", axis= 1)
target = df['quality']

# Training Test Split
variables_train, variables_test, target_train, target_test= train_test_split(variables, target, 
                                                                         train_size=0.8, stratify=target, random_state=0)

######### Exercise 1 ##########

residues = pd.Series()

for i in range(1, 11):
    # Learn the MLP regressor 
    mlp = MLPRegressor(hidden_layer_sizes=(10,10), activation='relu', early_stopping=True, validation_fraction=0.2, random_state=i)
    #Predict output
    y_pred = mlp.fit(variables_train,target_train).predict(variables_test)
    #Calculate residues
    print('y_pred shape: ',y_pred.shape)
    print('target shape: ',target_test.shape)
    residue = target_test - y_pred
    

print(residues.shape)
# Plot all the residues
# plt.hist(residues, edgecolor='black',bins=20)
# plt.title('Histogram of the residues')
# plt.xlabel('Residues')
# plt.ylabel('Frequency')
# plt.savefig('ex1_histogram.png')
# plt.show()

# ########## Exercise 2 ##########

# # Round and bound the predictions

# for i in range(1, 11):
#     y_pred = np.zeros(len(target_test))
#     # Learn the MLP regressor 
#     mlp = MLPRegressor(hidden_layer_sizes=(10,10), activation='relu', early_stopping=True, validation_fraction=0.2, random_state=i)
#     #Predict output
#     y_pred = mlp.fit(variables_train,target_train).predict(variables_test)
#     #Calculate MAE (not rounded and bounded)
#     mae = np.mean(abs(target_test - y_pred))
#     #Round and bound the predictions
#     y_pred = np.round(y_pred)
#     y_pred = np.clip(y_pred, a_min=1, a_max=10)
#     #Calculate MAE (rounded and bounded)
#     mae_rounded = np.mean(abs(target_test - y_pred))

# mean_mae = np.mean(mae)
# mean_mae_rounded = np.mean(mae_rounded)

# # Print the results
# print('MAE (not rounded and bounded): ', mean_mae)
# print('MAE (rounded and bounded): ', mean_mae_rounded)

# ########## Exercise 3 ##########

# # Calculate the RMSE for the old MLP regressor
# sum_rmse_old = 0
# for i in range(1, 11):
#     #Learn the old MLP regressor
#     mlp = MLPRegressor(hidden_layer_sizes=(10,10), activation='relu', early_stopping=True, validation_fraction=0.2, random_state=i)
#     #Predict old output
#     y_pred_old = mlp.fit(variables_train,target_train).predict(variables_test)
#     #Calculate old RMSE
#     rmse = np.sqrt(np.mean((target_test - y_pred_old) ** 2))
#     sum_rmse_old += rmse

# average_rmse_old = np.mean(sum_rmse_old/10)

# # Calculate the RMSE for each number of iterations
# new_rmse_array = []
# iter_array = [20,50,100,200]
# for iter in iter_array:
#     y_pred_new = np.zeros(len(target_test))
#     sum_rmse_new = 0
#     for i in range(1, 11):
#         # Learn the new MLP regressor 
#         mlp = MLPRegressor(hidden_layer_sizes=(10,10), activation='relu', solver='adam', max_iter = iter, random_state=i)
#         #Predict new output
#         y_pred_new = mlp.fit(variables_train,target_train).predict(variables_test)
#         #Calculate new RMSE
#         rmse = np.sqrt(np.mean((target_test - y_pred_new) ** 2))
#         sum_rmse_new += rmse
#     #Append new RMSE
#     new_rmse_array += [sum_rmse_new/10]


# def const(x): return average_rmse_old

# # Plot the RMSE
# plt.plot(iter_array, new_rmse_array, '-o', label='RMSE')
# plt.hlines(average_rmse_old, xmin=min(iter_array), xmax=max(iter_array), colors='r', linestyles='dashed')
# plt.xlabel('Number of iterations') 
# plt.ylabel('RMSE')
# plt.title('RMSE vs number of iterations')
# plt.savefig('ex3_rmse.png')
# plt.show()
