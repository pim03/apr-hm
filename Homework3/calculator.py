import numpy as np 

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
print('\n Training rms error: \n',np.linalg.norm(phi_matrix@w-z)/np.sqrt(4))
