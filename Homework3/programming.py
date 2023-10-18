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

y_pred = np.zeros(len(target_test))

# Average the mlp regressor
for i in range(1, 11):
    # Learn the MLP regressor 
    mlp = MLPRegressor(hidden_layer_sizes=(10,10), activation='relu', early_stopping=True, validation_fraction=0.2, random_state=i)
    #Predict output
    y_pred += mlp.fit(variables_train,target_train).predict(variables_test)

y_pred = y_pred/10
first_rmse = np.sqrt(np.mean((target_test - y_pred)**2))

######### Exercise 1 ##########

# Calculate the residues
residues = abs(target_test - y_pred)

# Plot the residues
plt.hist(residues, edgecolor='black' ,bins=20)
plt.title('Histogram of the residues')
plt.xlabel('Residues')
plt.ylabel('Frequency')
plt.savefig('ex1_histogram.png')
plt.show()

########## Exercise 2 ##########

# Round and bound the predictions
rounded_predictions = np.round(y_pred)
min_value = 1
max_value = 10 
rounded_and_bounded_predictions = np.clip(rounded_predictions, min_value, max_value)

# Calculate previous MAE
mae = np.mean(abs(target_test - y_pred))
#Calculate new MAE
mae_new = np.mean(abs(target_test - rounded_and_bounded_predictions))

print('The previous MAE is: ', mae)
print('The new MAE is: ', mae_new)

if mae_new < mae:
    print('The new MAE is lower than the previous one')
elif mae_new > mae:
    print('The new MAE is higher than the previous one')
else:
    print('The new MAE is equal to the previous one')

########## Exercise 3 ##########

# Calculate the RMSE for each number of iterations
rmse_final = []
iter_array = [20,50,100,200]
for iter in iter_array:
    y_pred = np.zeros(len(target_test))
    for i in range(1, 11):
        # Learn the MLP regressor 
        mlp = MLPRegressor(hidden_layer_sizes=(10,10), activation='relu', solver='adam', max_iter = iter, random_state=i)
        #Predict output
        y_pred += mlp.fit(variables_train,target_train).predict(variables_test)
    y_pred = y_pred/10
    rmse_final.append(np.sqrt(np.mean((y_pred-target_test)**2)))

def const(x):
    return first_rmse

# Plot the RMSE
plt.plot(iter_array, rmse_final, '-o', label='RMSE')
plt.hlines(first_rmse, xmin=min(iter_array), xmax=max(iter_array), colors='r', linestyles='dashed')
plt.xlabel('Number of iterations') 
plt.ylabel('RMSE')
plt.title('RMSE vs number of iterations')
plt.savefig('ex3_rmse.png')
plt.show()
