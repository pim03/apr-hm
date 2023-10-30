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

obss = np.array([x1,x2,x3,x4])

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
p_xi = []
for xi in obss:
    p_xi.append(probability_calculator(xi))
p_xi = np.array(p_xi)

print('p_x1: ',p_xi[0])
print('')
print('p_x2: ',p_xi[1])
print('')
print('p_x3: ',p_xi[2])
print('')
print('p_x4: ',p_xi[3])
print('\n')

#Calculate gamma of each x:
gamma_xi_c1 = []
gamma_xi_c2 = []
for xi in obss:
    gamma_xi_c1.append(gamma_calculator_c1(xi))
    gamma_xi_c2.append(gamma_calculator_c2(xi))
gamma_xi_c1 = np.array(gamma_xi_c1)
gamma_xi_c2 = np.array(gamma_xi_c2)

print('gamma_x1_c1: ',gamma_xi_c1[0])
print('')
print('gamma_x1_c2: ',gamma_xi_c2[0])
print('')
print('gamma_x2_c1: ',gamma_xi_c1[1])
print('')
print('gamma_x2_c2: ',gamma_xi_c2[1])
print('')
print('gamma_x3_c1: ',gamma_xi_c1[2])
print('')
print('gamma_x3_c2: ',gamma_xi_c2[2])
print('')
print('gamma_x4_c1: ',gamma_xi_c1[3])
print('')
print('gamma_x4_c2: ',gamma_xi_c2[3])
print('\n')

#Update parameters 

#Update pi
pi1 = np.sum(gamma_xi_c1)/4
pi2 = np.sum(gamma_xi_c2)/4

print('pi1_new: ',pi1)
print('')
print('pi2_new: ',pi2)
print('\n')

#Update p (probability of y1 = 1)
p1 = 0
p2 = 0
for i in range(len(obss)):
    p1 += gamma_xi_c1[i]*obss[i][0]
    p2 += gamma_xi_c2[i]*obss[i][0]
p1 = p1/np.sum(gamma_xi_c1)
p2 = p2/np.sum(gamma_xi_c2)

print('p1_new: ',p1)
print('')
print('p2_new: ',p2)
print('\n')

#Update u
u1 = [0,0]
u2 = [0,0]
for i in range(len(obss)):
    u1 += gamma_xi_c1[i]*(obss[i][1:])
    u2 += gamma_xi_c2[i]*(obss[i][1:])
u1 = np.array(u1)/np.sum(gamma_xi_c1)
u2 = np.array(u2)/np.sum(gamma_xi_c2)
print('u1_new: ',u1)
print('')
print('u2_new: ',u2)
print('\n')

#Update Sigma
Sigma1 = [[0,0],[0,0]]
Sigma2 = [[0,0],[0,0]]
for i in range(len(obss)):
    Sigma1 += gamma_xi_c1[i]*np.outer(obss[i][1:]-u1,obss[i][1:]-u1)
    Sigma2 += gamma_xi_c2[i]*np.outer(obss[i][1:]-u2,obss[i][1:]-u2)
Sigma1 = np.array(Sigma1)/np.sum(gamma_xi_c1)
Sigma2 = np.array(Sigma2)/np.sum(gamma_xi_c2)
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
    if xi[0] == 1:
        return mvn1.pdf(xi[1:])*p1
    elif xi[0] == 0:
        return mvn1.pdf(xi[1:])*(1-p1)
    else:
        raise ValueError('y1 must be 0 or 1')

def prob_xi_sabendo_c2(xi):
    if xi[0] == 1:
        return mvn2.pdf(xi[1:])*p2
    elif xi[0] == 0:
        return mvn2.pdf(xi[1:])*(1-p2)
    else:
        raise ValueError('y1 must be 0 or 1')

print('multivar normal: ',mvn1.pdf(x1[1:])*p1)

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

# Calculate p(xi|ck) for each observation and each cluster
p_xi_c1 = []
p_xi_c2 = []
for xi in obss:
    p_xi_c1.append(prob_xi_sabendo_c1(xi))
    p_xi_c2.append(prob_xi_sabendo_c2(xi))
p_xi_c1 = np.array(p_xi_c1)
p_xi_c2 = np.array(p_xi_c2)

print('p_x1_c1: ',p_xi_c1[0])
print('')
print('p_x1_c2: ',p_xi_c2[0])
print('')
print('p_x2_c1: ',p_xi_c1[1])
print('')
print('p_x2_c2: ',p_xi_c2[1])
print('')
print('p_x3_c1: ',p_xi_c1[2])
print('')
print('p_x3_c2: ',p_xi_c2[2])
print('')
print('p_x4_c1: ',p_xi_c1[3])
print('')
print('p_x4_c2: ',p_xi_c2[3])
print('\n')

points_in_cluster1 = []
points_in_cluster2 = []

for i in range(len(obss)):
    if p_xi_c1[i] > p_xi_c2[i]:
        points_in_cluster1.append(obss[i])
    elif p_xi_c1[i] < p_xi_c2[i]:
        points_in_cluster2.append(obss[i])
    else:
        points_in_cluster1.append(obss[i])
        points_in_cluster2.append(obss[i])

points_in_cluster1 = np.array(points_in_cluster1)
points_in_cluster2 = np.array(points_in_cluster2)
    
def manhattan_distance(x,y): return np.sum(np.abs(x-y))

#Calculate a's (average distance between point and other points in same cluster) and b's (average distance between point and other points in other cluster)
a_xi = []
b_xi = []
for xi in obss:
    if any(np.array_equal(xi,point) for point in points_in_cluster1):
        distancias_a = [manhattan_distance(xi,point) for point in points_in_cluster1 if not np.array_equal(xi,point)]
        if distancias_a: a_xi.append(np.mean(distancias_a))
        else: a_xi.append(0)
        distancias_b = [manhattan_distance(xi,point) for point in points_in_cluster2]
        b_xi.append(np.mean(distancias_b))
    elif any(np.array_equal(xi,point) for point in points_in_cluster2):
        distancias_a = [manhattan_distance(xi,point) for point in points_in_cluster2 if not np.array_equal(xi,point)]
        if distancias_a: a_xi.append(np.mean(distancias_a))
        else: a_xi.append(0)
        distancias_b = [manhattan_distance(xi,point) for point in points_in_cluster1]
        b_xi.append(np.mean(distancias_b))
    else:
        raise ValueError('xi must belong to cluster 1 or 2')
a_xi = np.array(a_xi)
b_xi = np.array(b_xi)

print('a_x1: ',a_xi[0])
print('')
print('a_x2: ',a_xi[1])
print('')
print('a_x3: ',a_xi[2])
print('')
print('a_x4: ',a_xi[3])
print('\n')
print('b_x1: ',b_xi[0])
print('')
print('b_x2: ',b_xi[1])
print('')
print('b_x3: ',b_xi[2])
print('')
print('b_x4: ',b_xi[3])
print('\n')


def silhouette(a,b):
    if a >= b: return (b-a)/a
    else: return (b-a)/b

#Calculate s's
s_xi = []
for i in range(len(obss)):
    s_xi.append(silhouette(a_xi[i],b_xi[i]))
s_xi = np.array(s_xi)

print('s_x1: ',s_xi[0])
print('')
print('s_x2: ',s_xi[1])
print('')
print('s_x3: ',s_xi[2])
print('')
print('s_x4: ',s_xi[3])
print('\n')

#Calculate silhouette of each cluster
soma1 = 0
soma2 = 0
for i in range(len(obss)):
    if any(np.array_equal(obss[i],point) for point in points_in_cluster1):
        soma1 += s_xi[i]
    elif any(np.array_equal(obss[i],point) for point in points_in_cluster2):
        soma2 += s_xi[i]
    else:
        raise ValueError('xi must belong to cluster 1 or 2')
s1 = soma1/len(points_in_cluster1)
s2 = soma2/len(points_in_cluster2)

print('s1: ',s1)
print('')
print('s2: ',s2)
print('\n')

#Calculate silhouette of the whole clustering
s = np.mean([s1,s2])
print('s: ',s)
print('\n')

