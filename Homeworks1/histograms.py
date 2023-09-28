import matplotlib.pyplot as plt
import numpy as np

# y1 = [3,2,1,5,4,10,12,11,7,9,6,8]
# y2 = [8,11,3.5,3.5,3.5,11,3.5,11,8,3.5,8,3.5]
# cima = 0
# baixo_dir = 0
# baixo_esq = 0
# soma1 = 0
# soma2 = 0
# for i in range(len(y1)): 
#     soma1 += y1[i]
#     soma2 += y2[i]
# mean1 = soma1/len(y1)
# mean2 = soma2/len(y2)
# for i in range(len(y1)):
#     cima += (y1[i]-mean1)*(y2[i]-mean2)
#     baixo_dir += (y1[i]-mean1)*(y1[i]-mean1)
#     baixo_esq += (y2[i]-mean2)*(y2[i]-mean2)

# pearson = cima/(np.sqrt(baixo_dir)*np.sqrt(baixo_esq))
# print(pearson)

data1 = [0.24,0.68,0.9,0.76]
plt.hist(data1, bins=bins, edgecolor='k', color='skyblue', density=True)  # Set density=True
plt.xlabel('y1',fontsize=14)
plt.title('Histograma de y1 condicionado a A',fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

data2 = [0.06,0.04,0.46,0.62]
plt.hist(data2, bins=bins, edgecolor='k', color= '#FF7F50', density=True)  # Set density=True
plt.xlabel('y1',fontsize=14)
plt.title('Histograma de y1 condicionado a B',fontsize=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

data3 = [0.32,0.36,0.44,0.52]
plt.hist(data3, bins=bins, edgecolor='k', color='#98FF98', density=True)  # Set density=True
plt.xlabel('y1',fontsize=14)
plt.title('Histograma de y1 condicionado a C',fontsize=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

plt.hist(data1, bins=bins, edgecolor='k', color='skyblue', density=True, alpha=0.5, label='A')
plt.hist(data2, bins=bins, edgecolor='k', color='#FF7F50', density=True, alpha=0.5, label='B')
plt.hist(data3, bins=bins, edgecolor='k', color='#98FF98', density=True, alpha=0.5, label='C')

plt.xlabel('y1', fontsize=14)
plt.title('Histogramas de y1 condicionados sobrepostos', fontsize=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.show()



