import numpy as np
from scipy.stats import multivariate_normal

u1 = np.array([1,1])
Sigma1 = np.array([[2,0.5],[0.5,2]])

u2 = np.array([0,0])
Sigma2 = np.array([[1.5,1],[1,1.5]])

p1 = 0.3 #probability of y1=1 in cluster 1
p2 = 0.7 #probability of y1=1 in cluster 2

pi1 = 0.5 #probability of cluster 1
pi2 = 0.5 #probability of cluster 2

x1 = np.array([1,0.6,0.1])
x2 = np.array([0,-0.4,0.8])
x3 = np.array([0,0.2,0.5])
x4 = np.array([1,0.4,-0.1])

mvn1 = multivariate_normal(u1,Sigma1) #probability density function of y2 and y3 in cluster 1
mvn2 = multivariate_normal(u2,Sigma2) #probability density function of y2 and y3 in cluster 2

def probability_calculator(xi):
    if xi[0] ==1:
        return pi1*mvn1.pdf(xi[1:])*p1 + pi2*mvn2.pdf(xi[1:])*p2
    if xi[0] ==0:
        return pi1*mvn1.pdf(xi[1:])*(1-p1) + pi2*mvn2.pdf(xi[1:])*(1-p2)

def gamma_calculator_c1(xi):
    if xi[0] ==1:
        gamma_xi_c1 = pi1*mvn1.pdf(xi[1:])*p1/probability_calculator(xi)
        return gamma_xi_c1
    elif xi[0] ==0:
        gamma_xi_c1 = pi1*mvn1.pdf(xi[1:])*(1-p1)/probability_calculator(xi)
        return gamma_xi_c1
    else:
        raise ValueError('y1 must be 0 or 1')
    
def gamma_calculator_c2(xi):
    if xi[0] ==1:
        gamma_xi_c2 = pi2*mvn2.pdf(xi[1:])*p2/probability_calculator(xi)
        return gamma_xi_c2
    if xi[0] ==0:
        gamma_xi_c2 = pi2*mvn2.pdf(xi[1:])*(1-p2)/probability_calculator(xi)
        return gamma_xi_c2
    else:
        raise ValueError('y1 must be 0 or 1')

#Calculate probability of each x:
p_x1 = probability_calculator(x1)
p_x2 = probability_calculator(x2)
p_x3 = probability_calculator(x3)
p_x4 = probability_calculator(x4)

print('p_x1: ',p_x1)
print('')
print('p_x2: ',p_x2)
print('')
print('p_x3: ',p_x3)
print('')
print('p_x4: ',p_x4)
print('\n')

#Calculate p(ck|xi) = p(xi|ck)*p(ck)/p(xi)
gamma_x1_c1 = gamma_calculator_c1(x1)
gamma_x1_c2 = gamma_calculator_c2(x1)
gamma_x2_c1 = gamma_calculator_c1(x2)
gamma_x2_c2 = gamma_calculator_c2(x2)
gamma_x3_c1 = gamma_calculator_c1(x3)
gamma_x3_c2 = gamma_calculator_c2(x3)
gamma_x4_c1 = gamma_calculator_c1(x4)
gamma_x4_c2 = gamma_calculator_c2(x4)

print('gamma_x1_c1: ',gamma_x1_c1)
print('')
print('gamma_x1_c2: ',gamma_x1_c2)
print('')
print('gamma_x2_c1: ',gamma_x2_c1)
print('')
print('gamma_x2_c2: ',gamma_x2_c2)
print('')
print('gamma_x3_c1: ',gamma_x3_c1)
print('')
print('gamma_x3_c2: ',gamma_x3_c2)
print('')
print('gamma_x4_c1: ',gamma_x4_c1)
print('')
print('gamma_x4_c2: ',gamma_x4_c2)
print('\n')

#Update parameters 

#Update pi
pi1 = (gamma_x1_c1 + gamma_x2_c1 + gamma_x3_c1 + gamma_x4_c1)/4
pi2 = (gamma_x1_c2 + gamma_x2_c2 + gamma_x3_c2 + gamma_x4_c2)/4
print('pi1_new: ',pi1)
print('')
print('pi2_new: ',pi2)
print('\n')

#Update p (probability of y1 = 1)
p1 = (gamma_x1_c1 + gamma_x4_c1)/(gamma_x1_c1 + gamma_x2_c1 + gamma_x3_c1 + gamma_x4_c1)
p2 = (gamma_x1_c2 + gamma_x4_c2)/(gamma_x1_c2 + gamma_x2_c2 + gamma_x3_c2 + gamma_x4_c2)
print('p1_new: ',p1)
print('')
print('p2_new: ',p2)
print('\n')

#Update u
u1 = (gamma_x1_c1*x1[1:] + gamma_x2_c1*x2[1:] + gamma_x3_c1*x3[1:] + gamma_x4_c1*x4[1:])/np.sum([gamma_x1_c1, gamma_x2_c1, gamma_x3_c1, gamma_x4_c1])
u2 = (gamma_x1_c2*x1[1:] + gamma_x2_c2*x2[1:] + gamma_x3_c2*x3[1:] + gamma_x4_c2*x4[1:])/np.sum([gamma_x1_c2, gamma_x2_c2, gamma_x3_c2, gamma_x4_c2])
print('u1_new: ',u1)
print('')
print('u2_new: ',u2)
print('\n')

#Update Sigma
Sigma1 = (gamma_x1_c1*np.outer(x1[1:]-u1,x1[1:]-u1) + gamma_x2_c1*np.outer(x2[1:]-u1,x2[1:]-u1) + gamma_x3_c1*np.outer(x3[1:]-u1,x3[1:]-u1) + gamma_x4_c1*np.outer(x4[1:]-u1,x4[1:]-u1))/np.sum([gamma_x1_c1, gamma_x2_c1, gamma_x3_c1, gamma_x4_c1])
Sigma2 = (gamma_x1_c2*np.outer(x1[1:]-u2,x1[1:]-u2) + gamma_x2_c2*np.outer(x2[1:]-u2,x2[1:]-u2) + gamma_x3_c2*np.outer(x3[1:]-u2,x3[1:]-u2) + gamma_x4_c2*np.outer(x4[1:]-u1,x4[1:]-u2))/np.sum([gamma_x1_c2, gamma_x2_c2, gamma_x3_c2, gamma_x4_c2])

print('Sigma1_new: ',Sigma1)
print('')
print('Sigma2_new: ',Sigma2)
print('\n')

#Update mvn1 and mvn2
mvn1 = multivariate_normal(u1,Sigma1) #probability density function of y2 and y3 in cluster 1
mvn2 = multivariate_normal(u2,Sigma2) #probability density function of y2 and y3 in cluster 2

def probability_calculator(xi):
    if x1[0] ==1:
        return pi1*mvn1.pdf(xi[1:])*p1 + pi2*mvn2.pdf(xi[1:])*p2
    if x1[0] ==0:
        return pi1*mvn1.pdf(xi[1:])*(1-p1) + pi2*mvn2.pdf(xi[1:])*(1-p2)

def gamma_calculator_c1(xi):
    if xi[0] ==1:
        gamma_xi_c1 = pi1*mvn1.pdf(xi[1:])*p1/probability_calculator(xi)
        return gamma_xi_c1
    elif xi[0] ==0:
        gamma_xi_c1 = pi1*mvn1.pdf(xi[1:])*(1-p1)/probability_calculator(xi)
        return gamma_xi_c1
    else:
        raise ValueError('y1 must be 0 or 1')
    
def gamma_calculator_c2(xi):
    if xi[0] ==1:
        gamma_xi_c2 = pi2*mvn2.pdf(xi[1:])*p2/probability_calculator(xi)
        return gamma_xi_c2
    if xi[0] ==0:
        gamma_xi_c2 = pi2*mvn2.pdf(xi[1:])*(1-p2)/probability_calculator(xi)
        return gamma_xi_c2
    else:
        raise ValueError('y1 must be 0 or 1')

def prob_xi_sabendo_c1(xi):
    if x1[0] == 1:
        return mvn1.pdf(xi[1:])*p1/pi1
    elif x1[0] == 0:
        return mvn1.pdf(xi[1:])*(1-p1)/pi1
    else:
        raise ValueError('y1 must be 0 or 1')

def prob_xi_sabendo_c2(xi):
    if x1[0] == 1:
        return mvn2.pdf(xi[1:])*p2/pi2
    elif x1[0] == 0:
        return mvn2.pdf(xi[1:])*(1-p2)/pi2
    else:
        raise ValueError('y1 must be 0 or 1')

#Assign new observation to clusters
x_new = np.array([1,0.3,0.7])

p_xnew = probability_calculator(x_new)
print('p_xnew: ',p_xnew)
print('\n')
gamma_x_new_c1 = gamma_calculator_c1(x_new)
gamma_x_new_c2 = gamma_calculator_c2(x_new)
print('gamma_x_new_c1: ',gamma_x_new_c1)
print('')
print('gamma_x_new_c2: ',gamma_x_new_c2)
print('\n')

if gamma_x_new_c1 > gamma_x_new_c2:
    print('x_new belongs to cluster 1')
elif gamma_x_new_c1 < gamma_x_new_c2:
    print('x_new belongs to cluster 2')
else:
    print('x_new belongs to both clusters')

# Calculate p(x1|ck) for each observation and each cluster

p_x1_c1 = prob_xi_sabendo_c1(x1)
p_x1_c2 = prob_xi_sabendo_c2(x1)
p_x2_c1 = prob_xi_sabendo_c1(x2)
p_x2_c2 = prob_xi_sabendo_c2(x2)
p_x3_c1 = prob_xi_sabendo_c1(x3)
p_x3_c2 = prob_xi_sabendo_c2(x3)
p_x4_c1 = prob_xi_sabendo_c1(x4)
p_x4_c2 = prob_xi_sabendo_c2(x4)

print('p_x1_c1: ',p_x1_c1)
print('')
print('p_x1_c2: ',p_x1_c2)
print('')
print('p_x2_c1: ',p_x2_c1)
print('')
print('p_x2_c2: ',p_x2_c2)
print('')
print('p_x3_c1: ',p_x3_c1)
print('')
print('p_x3_c2: ',p_x3_c2)
print('')
print('p_x4_c1: ',p_x4_c1)
print('')
print('p_x4_c2: ',p_x4_c2)
print('\n')

points_in_cluster1 = []
points_in_cluster2 = []

if p_x1_c1 > p_x1_c2: points_in_cluster1.append(x1)
elif p_x1_c1 < p_x1_c2: points_in_cluster2.append(x1)
else:
    points_in_cluster1.append(x1)
    points_in_cluster2.append(x1)
if p_x2_c1 > p_x2_c2: points_in_cluster1.append(x2)
elif p_x2_c1 < p_x2_c2: points_in_cluster2.append(x2)
else:
    points_in_cluster1.append(x2)
    points_in_cluster2.append(x2)
if p_x3_c1 > p_x3_c2: points_in_cluster1.append(x3)
elif p_x3_c1 < p_x3_c2: points_in_cluster2.append(x3)
else:
    points_in_cluster1.append(x3)
    points_in_cluster2.append(x3)
if p_x4_c1 > p_x4_c2: points_in_cluster1.append(x4)
elif p_x4_c1 < p_x4_c2: points_in_cluster2.append(x4)
else:
    points_in_cluster1.append(x4)
    points_in_cluster2.append(x4)

points_in_cluster1 = np.array(points_in_cluster1)
points_in_cluster2 = np.array(points_in_cluster2)
    
def manhattan_distance(x,y): return np.sum(np.abs(x-y))

#Calculate a's
if any(np.array_equal(x1,point) for point in points_in_cluster1):
    soma_a = np.sum([manhattan_distance(x1,point) for point in points_in_cluster1])
    if soma_a == 0:
        a_x1 = 0
    else:
        a_x1 = soma_a/(len(points_in_cluster1)-1)
    b_x1 = np.sum([manhattan_distance(x1,point) for point in points_in_cluster2])/(len(points_in_cluster2))
else:
    a_x1 = np.sum([manhattan_distance(x1,point) for point in points_in_cluster2])/(len(points_in_cluster2)-1)
    b_x1 = np.sum([manhattan_distance(x1,point) for point in points_in_cluster1])/(len(points_in_cluster1))

if any(np.array_equal(x2,point) for point in points_in_cluster1):
    soma_a = np.sum([manhattan_distance(x2,point) for point in points_in_cluster1])
    if soma_a == 0:
        a_x2 = 0
    else:
        a_x2 = soma_a/(len(points_in_cluster1)-1)
    b_x2 = np.sum([manhattan_distance(x2,point) for point in points_in_cluster2])/(len(points_in_cluster2))
else:
    a_x2 = np.sum([manhattan_distance(x2,point) for point in points_in_cluster2])/(len(points_in_cluster2)-1)
    b_x2 = np.sum([manhattan_distance(x2,point) for point in points_in_cluster1])/(len(points_in_cluster1))

if any(np.array_equal(x3,point) for point in points_in_cluster1):
    soma_a = np.sum([manhattan_distance(x3,point) for point in points_in_cluster1])
    if soma_a == 0:
        a_x3 = 0
    else:
        a_x3 = soma_a/(len(points_in_cluster1)-1)
    b_x3 = np.sum([manhattan_distance(x3,point) for point in points_in_cluster2])/(len(points_in_cluster2))
else:
    a_x3 = np.sum([manhattan_distance(x3,point) for point in points_in_cluster2])/(len(points_in_cluster2)-1)
    b_x3 = np.sum([manhattan_distance(x3,point) for point in points_in_cluster1])/(len(points_in_cluster1))

if any(np.array_equal(x4,point) for point in points_in_cluster1):
    soma_a = np.sum([manhattan_distance(x4,point) for point in points_in_cluster1])
    if soma_a == 0:
        a_x4 = 0
    else:
        a_x4 = soma_a/(len(points_in_cluster1)-1)
    b_x4 = np.sum([manhattan_distance(x4,point) for point in points_in_cluster2])/(len(points_in_cluster2))
else:
    a_x4 = np.sum([manhattan_distance(x4,point) for point in points_in_cluster2])/(len(points_in_cluster2)-1)
    b_x4 = np.sum([manhattan_distance(x4,point) for point in points_in_cluster1])/(len(points_in_cluster1))

def silhouette(a,b):
    if a >= b: return (b-a)/a
    else: return (b-a)/b

#Calculate s's
s_x1 = silhouette(a_x1,b_x1)
s_x2 = silhouette(a_x2,b_x2)
s_x3 = silhouette(a_x3,b_x3)
s_x4 = silhouette(a_x4,b_x4)


print('s_x1: ',s_x1)
print('')
print('s_x2: ',s_x2)
print('')
print('s_x3: ',s_x3)
print('')
print('s_x4: ',s_x4)
print('\n')

#Calculate silhouette of each cluster
soma1 = 0
if any(np.array_equal(x4,point) for point in points_in_cluster1):
    soma1 += s_x4
if any(np.array_equal(x3,point) for point in points_in_cluster1):
    soma1 += s_x3
if any(np.array_equal(x2,point) for point in points_in_cluster1):
    soma1 += s_x2
if any(np.array_equal(x1,point) for point in points_in_cluster1):
    soma1 += s_x1
s1 = soma1/len(points_in_cluster1)

soma2 = 0
if any(np.array_equal(x4,point) for point in points_in_cluster2):
    soma2 += s_x4
if any(np.array_equal(x3,point) for point in points_in_cluster2):
    soma2 += s_x3
if any(np.array_equal(x2,point) for point in points_in_cluster2):
    soma2 += s_x2
if any(np.array_equal(x1,point) for point in points_in_cluster2):
    soma2 += s_x1
s2 = soma2/len(points_in_cluster2)


print('s1: ',s1)
print('')
print('s2: ',s2)
print('\n')

#Calculate silhouette of the whole clustering
s = np.mean([s1,s2])
print('s: ',s)
print('\n')

