import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab
from numpy import genfromtxt
import csv


my_data = np.genfromtxt('wine_data.csv', delimiter=',')
data = my_data[:,1:]
target= my_data[:,0] # Class of each instance (1, 2 or 3)
print("Size of the data {} ".format(data.shape))

# Draw the data in 3/13 dimensions
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111, projection='3d')
ax1.scatter(data[:,3],data[:,1],data[:,2], c=target)
ax1.set_xlabel('1st dimension')
ax1.set_ylabel('2nd dimension')
ax1.set_zlabel('3rd dimension')
ax1.set_title("Vizualization of the dataset (3 out of 13 dimensions)")


# Data standarized
mean = np.mean(data, axis=0)
std = np.std(data, axis=0, ddof=1)  
X_standardized = (data - mean) / std

# cov matrix
cov_matrix = np.cov(X_standardized, rowvar=False)  # Covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

sorted_indices = np.argsort(eigenvalues)[::-1]  
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]


P2 = eigenvectors[:, :2]  # Projection matrix for 2 components
P3 = eigenvectors[:, :3]  # Projection matrix for 3 components


newData2 = X_standardized @ P2  
newData3 = X_standardized @ P3  


variance_preserved_2 = np.sum(eigenvalues[:2]) / np.sum(eigenvalues)
variance_preserved_3 = np.sum(eigenvalues[:3]) / np.sum(eigenvalues)
print(f"Variance preserved (2 components): {variance_preserved_2:.4f}")
print(f"Variance preserved (3 components): {variance_preserved_3:.4f}")

#=============================================================================

# first two principal components 
plt.figure(2)
plt.scatter(newData2[:,0],newData2[:,1], c=target)
plt.xlabel('1st Principal Component')
plt.ylabel('2nd Principal Component')
plt.title("Projection to the top-2 Principal Components")
plt.draw()

# first three principal components 
fig = plt.figure(3)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(newData3[:,0],newData3[:,1], newData3[:,2], c=target)
ax.set_xlabel('1st Principal Component')
ax.set_ylabel('2nd Principal Component')
ax.set_zlabel('3rd Principal Component')
ax.set_title("Projection to the top-3 Principal Components")
plt.show()  
