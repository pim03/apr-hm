import numpy as np
from scipy.stats import multivariate_normal
import scipy.stats as stats

def covariance(x1,x2,mu1,mu2):
    sum = 0
    for i in range(len(x1)):
        sum = sum + (x1[i]-mu1)*(x2[i]-mu2)
    return sum/(len(x1))

def bi_dim_gaussian_prob(x, mu, inverse,det_sigma):
    return 1/(2*np.pi*np.sqrt(det_sigma))*np.exp(-0.5*np.dot(np.dot((x-mu).T,inverse),(x-mu)))

print('--------------class A-----------------')
#Training data
training_data_y1_A = [0.24,0.16,0.32]
training_data_y2_A = [0.36,0.48,0.72]

#Test data
test_data_A = [0.42,0.59]

#Calculate parameters
mu1 = np.mean(training_data_y1_A)
mu2 = np.mean(training_data_y2_A)
cov_y1_y2 = covariance(training_data_y1_A,training_data_y2_A,mu1,mu2)
cov_y1_y1 = covariance(training_data_y1_A,training_data_y1_A,mu1,mu2)
cov_y2_y2 = covariance(training_data_y2_A,training_data_y2_A,mu1,mu2)

print('Parameters:')
print('mu1:',mu1)
print('mu2:',mu2)
print('cov_y1_y2:',cov_y1_y2)
print('cov_y1_y1:',cov_y1_y1)
print('cov_y2_y2:',cov_y2_y2)

# Define the parameters of the 2D normal distribution
mean = np.array([mu1, mu2])  # Mean vector [mu1, mu2]
covariance_matrix = np.array([[cov_y1_y1, cov_y1_y2], [cov_y1_y2, cov_y2_y2]])  # Covariance matrix
determinant = np.linalg.det(covariance_matrix)  # Determinant of the covariance matrix
inverse_covariance_matrix = np.linalg.inv(covariance_matrix)  # Inverse of the covariance matrix
print('determinant:',determinant)
print('inverse:',inverse_covariance_matrix)

# Define the vector for which you want to calculate the probability
x1 = np.array(test_data_A)

# Create a multivariate normal distribution object
multivar_normal = multivariate_normal(mean=mean, cov=covariance_matrix)

# Calculate the probability density at the given vector 'x'
probability = multivar_normal.pdf(x1)

<<<<<<< HEAD
print("Probability xA:", probability)
=======
probability_mine = bi_dim_gaussian_prob(x1,mean,inverse_covariance_matrix,determinant)
>>>>>>> a00509bad4af7f5842fc9249e8f0d6ef1796ae65

print("Probability xA multivar:", probability)
#print("Probability xA mine:", probability_mine)
############################# class B #########################################

print('--------------class B-----------------')


training_data_y1_B=[0.54,0.66,0.76,0.41]
training_data_y2_B=[0.11,0.39,0.28,0.53]

#Test data
test_data_B = [0.38,0.52]

#Calculate parameters
#Calculate parameters
mu1 = np.mean(training_data_y1_B)
mu2 = np.mean(training_data_y2_B)
cov_y1_y2 = covariance(training_data_y1_B,training_data_y2_B,mu1,mu2)
cov_y1_y1 = covariance(training_data_y1_B,training_data_y1_B,mu1,mu2)
cov_y2_y2 = covariance(training_data_y2_B,training_data_y2_B,mu1,mu2)
print('Parameters:')    
print('mu1:',mu1)
print('mu2:',mu2)
print('cov_y1_y2:',cov_y1_y2)
print('cov_y1_y1:',cov_y1_y1)
print('cov_y2_y2:',cov_y2_y2)

# Define the parameters of the 2D normal distribution
mean = np.array([mu1, mu2])  # Mean vector [mu1, mu2]
covariance_matrix2 = np.array([[cov_y1_y1, cov_y1_y2], [cov_y1_y2, cov_y2_y2]])  # Covariance matrix
determinant = np.linalg.det(covariance_matrix2)  # Determinant of the covariance matrix
inverse_covariance_matrix2 = np.linalg.inv(covariance_matrix2)  # Inverse of the covariance matrix
print('determinant:',determinant)
print('inverse:',inverse_covariance_matrix2)

# Define the vector for which you want to calculate the probability
xB = np.array(test_data_B)

# Create a multivariate normal distribution object
multivar_normal = multivariate_normal(mean=mean, cov=covariance_matrix2)

# Calculate the probability density at the given vector 'x'
<<<<<<< HEAD
probability = multivar_normal.pdf(xB)

print("Probability xB:", probability)
=======
probability = multivar_normal.pdf(x1)
probability_mine = bi_dim_gaussian_prob(x1,mean,inverse_covariance_matrix,determinant)
print("Probability xB:", probability)
print("Probability xB mine:", probability_mine)

num = 1/(2*np.pi*np.sqrt(determinant))
print('num:',num)
>>>>>>> a00509bad4af7f5842fc9249e8f0d6ef1796ae65



