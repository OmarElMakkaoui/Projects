import numpy as np
import scipy.linalg as linalg

def my_LDA(X, Y):
    classLabels = np.unique(Y)
    classNum = len(classLabels)
    datanum, dim = X.shape
    totalMean = np.mean(X, axis=0)

    S_W = np.zeros((dim, dim))
    class_means = {}

    for c in classLabels:
        X_c = X[Y == c]
        mean_c = np.mean(X_c, axis=0)
        class_means[c] = mean_c
        S_W += np.dot((X_c - mean_c).T, (X_c - mean_c))

    S_B = np.zeros((dim, dim))

    for c in classLabels:
        N_c = X[Y == c].shape[0]
        mean_c = class_means[c].reshape(dim, 1)
        totalMean_col = totalMean.reshape(dim, 1)
        S_B += N_c * np.dot((mean_c - totalMean_col), (mean_c - totalMean_col).T)

    eigvals, eigvecs = linalg.eig(np.linalg.pinv(S_W).dot(S_B))
    eiglist = sorted([(eigvals[i], eigvecs[:, i]) for i in range(len(eigvals))], key=lambda x: x[0].real, reverse=True)
    
    W = np.array([eiglist[i][1] for i in range(classNum - 1)]).real.T
    X_lda = np.dot(X, W)
    projected_centroid = np.array([np.mean(X_lda[Y == c], axis=0) for c in classLabels])

    return W, projected_centroid, X_lda