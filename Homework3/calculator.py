import numpy as np 

'''
def phi(x,c):
    return np.exp(-0.5*np.linalg.norm(x-c)**2)

x1 = np.array([0.7,-0.3])
x2 = np.array([0.4,0.5])
x3 = np.array([-0.2,0.8])
x4 = np.array([-0.4,0.3])

z = np.array([0.8,0.6,0.3,0.3])

c1 = np.array([0.0,0.0])
c2 = np.array([1,-1])
c3 = np.array([-1,1])

lamb = 0.1

transformed_x1 = np.array([phi(x1,c1),phi(x1,c2),phi(x1,c3)])
transformed_x2 = np.array([phi(x2,c1),phi(x2,c2),phi(x2,c3)])
transformed_x3 = np.array([phi(x3,c1),phi(x3,c2),phi(x3,c3)])
transformed_x4 = np.array([phi(x4,c1),phi(x4,c2),phi(x4,c3)])

print(transformed_x1)
print(transformed_x2)
print(transformed_x3)
print(transformed_x4)

line1 = np.array([1,transformed_x1[0], transformed_x1[1], transformed_x1[2]])
line2 = np.array([1,transformed_x2[0], transformed_x2[1], transformed_x2[2]])
line3 = np.array([1,transformed_x3[0], transformed_x3[1], transformed_x3[2]])
line4 = np.array([1,transformed_x4[0], transformed_x4[1], transformed_x4[2]])

phi_matrix = np.array([line1,line2,line3,line4])
print('\n Phi matrix: \n',phi_matrix)
phi_matrix_transposed = phi_matrix.transpose()
print('\n Phi Matrix transposed: \n',phi_matrix_transposed)
lambda_matrix = np.array([np.array([lamb,0,0,0]),np.array([0,lamb,0,0]),np.array([0,0,lamb,0]),np.array([0,0,0,lamb])])
inverse = np.linalg.inv(phi_matrix_transposed @ phi_matrix + lambda_matrix)

print('\n inverse*transposed: \n',inverse@phi_matrix_transposed)

w = np.dot(inverse@phi_matrix_transposed,z)

print('\n w: \n',w)

#Def my regression function
def ridge(x):
    return w[0]+w[1]*x[0]+w[2]*x[1]+w[3]*x[2]

#Calculate estimated values
z_hat = np.array([ridge(transformed_x1),ridge(transformed_x2),ridge(transformed_x3),ridge(transformed_x4)])
print('\n z1: ',z_hat[0])
print('\n z2: ',z_hat[1])
print('\n z3: ',z_hat[2])
print('\n z4: ',z_hat[3])

sum = 0
for i in range(len(z_hat)):
    sum += (z_hat[i]-z[i])**2

rmse = np.sqrt(sum/len(z_hat))
print('\n RMSE: ',rmse)
#Calculate the training rms error 
print('\n Training rms error: \n',np.linalg.norm(phi_matrix@w-z)/np.sqrt(4)) '''


### MLP ###

x1 = np.array([[1], [1], [1], [1]])
x2 = np.array([[1], [0], [0], [-1]])

w1 = np.array([[1, 1, 1, 1], [1, 1, 2, 1], [1, 1, 1, 1]])
print('w1: \n', w1)
b1 = np.array([[1], [1], [1]])
print('b1: \n', b1)
t1 = np.array([[0], [1], [0]])
print('t1: \n', t1)

w2 = np.array([[1, 4, 1], [1, 1, 1]])
print('w2: \n', w2)
b2 = np.array([[1], [1]])
print('b2: \n', b2)
t2 = np.array([[1], [0], [0]])
print('t2: \n', t2)

w3 = np.array([[1, 1], [3, 1], [1, 1]])
print('w3: \n', w3)
b3 = np.array([[1], [1], [1]])
print('b3: \n', b3)

def phi(x):
    return np.tanh(0.5 * x - 2)

def phi_prime(x):
    return 0.5 * (1 - np.tanh(0.5 * x - 2) ** 2)

# x1_p and x2_p being p the index of the layer

### first observation ###

z1_1 = np.dot(w1, x1) + b1
print('z1_1: \n', z1_1)
x1_1 = phi(z1_1)
print('x1_1: \n', x1_1)

z1_2 = np.dot(w2, x1_1) + b2
print('z1_2: \n', z1_2)
x1_2 = phi(z1_2)
print('x1_2: \n', x1_2)

z1_3 = np.dot(w3, x1_2) + b3
print('z1_3: \n', z1_3)
x1_3 = phi(z1_3) 
print('x1_3: \n', x1_3)

delta1_3 = (x1_3 - t1) * phi_prime(z1_3)
print('delta1_3: \n', delta1_3)

delta1_2 = np.dot(w3.transpose(), delta1_3) * phi_prime(z1_2)
print('delta1_2: \n', delta1_2)

delta1_1 = np.dot(w2.transpose(), delta1_2) * phi_prime(z1_1)
print('delta1_1: \n', delta1_1)

### second observation ###

z2_1 = np.dot(w1, x2) + b1
print('z2_1: \n', z2_1)
x2_1 = phi(z2_1)
print('x2_1: \n', x2_1)

z2_2 = np.dot(w2, x2_1) + b2
print('z2_2: \n', z2_2)
x2_2 = phi(z2_2)
print('x2_2: \n', x2_2)

z2_3 = np.dot(w3, x2_2) + b3
print('z2_3: \n', z2_3)
x2_3 = phi(z2_3)
print('x2_3: \n', x2_3)

delta2_3 = (x2_3 - t2) * phi_prime(z2_3)
print('delta2_3: \n', delta2_3)

delta2_2 = np.dot(w3.transpose(), delta2_3) * phi_prime(z2_2)
print('delta2_2: \n', delta2_2)

delta2_1 = np.dot(w2.transpose(), delta2_2) * phi_prime(z2_1)
print('delta2_1: \n', delta2_1)

### derivatives ###
dE1_dw1 = np.dot(delta1_1, x1.transpose())
print('dE1_dw1: \n', dE1_dw1)

dE2_dw1 = np.dot(delta2_1, x2.transpose())
print('dE2_dw1: \n', dE2_dw1)

dE1_dw2 = np.dot(delta1_2, x1_1.transpose())
print('dE1_dw2: \n', dE1_dw2)

dE2_dw2 = np.dot(delta2_2, x2_1.transpose())
print('dE2_dw2: \n', dE2_dw2)

dE1_dw3 = np.dot(delta1_3, x1_2.transpose())
print('dE1_dw3: \n', dE1_dw3)

dE2_dw3 = np.dot(delta2_3, x2_2.transpose())
print('dE2_dw3: \n', dE2_dw3)

### final weights ###

w1_new = w1 - 0.1 * (dE1_dw1 + dE2_dw1)
print('w1_new: \n', w1_new)

w2_new = w2 - 0.1 * (dE1_dw2 + dE2_dw2)
print('w2_new: \n', w2_new)

w3_new = w3 - 0.1 * (dE1_dw3 + dE2_dw3)
print('w3_new: \n', w3_new)

### final biases ###

b1_new = b1 - 0.1 * (delta1_1 + delta2_1)
print('b1_new: \n', b1_new)

b2_new = b2 - 0.1 * (delta1_2 + delta2_2)
print('b2_new: \n', b2_new)

b3_new = b3 - 0.1 * (delta1_3 + delta2_3)
print('b3_new: \n', b3_new)

