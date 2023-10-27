import numpy as np
from scipy.stats import multivariate_normal

u1 = np.array([1,1])
Sigma1 = np.array([[2,0.5],[0.5,2]])

u2 = np.array([0,0])
Sigma2 = np.array([[1.5,1],[1,1.5]])

p1 = 0.3 #probability of y1 in cluster 1
p2 = 0.7 #probability of y1 in cluster 2

pi1 = 0.5 #probability of cluster 1
pi2 = 0.5 #probability of cluster 2

x1 = np.array([1,0.6,0.1])
x2 = np.array([0,-0.4,0.8])
x3 = np.array([0,0.2,0.5])
x4 = np.array([1,0.4,-0.1])

mvn1 = multivariate_normal(u1,Sigma1) #probability density function of y2 and y3 in cluster 1
mvn2 = multivariate_normal(u2,Sigma2) #probability density function of y2 and y3 in cluster 2

#Calculate probability of each x:
p_x1 = pi1*mvn1.pdf(x1[1:])*p1 + pi2*mvn2.pdf(x1[1:])*p2
p_x2 = pi1*mvn1.pdf(x2[1:])*p1 + pi2*mvn2.pdf(x2[1:])*p2
p_x3 = pi1*mvn1.pdf(x3[1:])*p1 + pi2*mvn2.pdf(x3[1:])*p2
p_x4 = pi1*mvn1.pdf(x4[1:])*p1 + pi2*mvn2.pdf(x4[1:])*p2

print('p_x1: ',p_x1)
print('p_x2: ',p_x2)
print('p_x3: ',p_x3)
print('p_x4: ',p_x4)

#Calculate p(ck|xi) = p(xi|ck)*p(ck)/p(xi)
p_x1_c1 = pi1*mvn1.pdf(x1[1:])*p1/p_x1
p_x1_c2 = pi2*mvn2.pdf(x1[1:])*p2/p_x1
p_x2_c1 = pi1*mvn1.pdf(x2[1:])*p1/p_x2
p_x2_c2 = pi2*mvn2.pdf(x2[1:])*p2/p_x2
p_x3_c1 = pi1*mvn1.pdf(x3[1:])*p1/p_x3
p_x3_c2 = pi2*mvn2.pdf(x3[1:])*p2/p_x3
p_x4_c1 = pi1*mvn1.pdf(x4[1:])*p1/p_x4
p_x4_c2 = pi2*mvn2.pdf(x4[1:])*p2/p_x4

print('p_x1_c1: ',p_x1_c1)
print('p_x1_c2: ',p_x1_c2)
print('p_x2_c1: ',p_x2_c1)
print('p_x2_c2: ',p_x2_c2)
print('p_x3_c1: ',p_x3_c1)
print('p_x3_c2: ',p_x3_c2)
print('p_x4_c1: ',p_x4_c1)
print('p_x4_c2: ',p_x4_c2)

