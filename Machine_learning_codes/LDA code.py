import numpy as np
import scipy.linalg as linalg

np.random.seed(42)

dataset = np.genfromtxt('wine_data.csv', delimiter=',')
train_size = 100

np.random.shuffle(dataset)
train_data = dataset[:train_size, 1:]
train_labels = dataset[:train_size, 0]

test_data = dataset[train_size:, 1:]
test_labels = dataset[train_size:, 0]

def lda_train(X, Y):
    labels = np.unique(Y)
    num_classes = len(labels)
    num_samples, dim = X.shape
    mean_total = np.mean(X, axis=0)

    class_indices = [np.where(Y == label)[0] for label in labels]
    class_means = [(np.mean(X[idx], axis=0), len(idx)) for idx in class_indices]

    Sw = np.zeros((dim, dim))
    for idx in class_indices:
        Sw += np.cov(X[idx], rowvar=False) * (len(idx) - 1)

    Sb = np.zeros((dim, dim))
    for mu, class_size in class_means:
        diff = mu - mean_total
        diff = diff.reshape(dim, 1)
        Sb += class_size * np.dot(diff, diff.T)

    try:
        S = np.dot(linalg.inv(Sw), Sb)
        eigval, eigvec = linalg.eig(S)
    except:
        print("Matrix is singular, using alternative method")
        eigval, eigvec = linalg.eig(Sb, Sw + Sb)

    idx = eigval.argsort()[::-1]
    eigvec = eigvec[:, idx]
    W = np.real(eigvec[:, :num_classes - 1])

    projected_centroids = [np.dot(mu, W) for mu, _ in class_means]
    X_lda = np.real(np.dot(X, W))

    return W, projected_centroids, X_lda

W, projected_centroids, X_lda = lda_train(train_data, train_labels)

def lda_predict(X, projected_centroids, W):
    projected_data = np.dot(X, W)
    dist = [linalg.norm(data - centroid) for data in projected_data for centroid in projected_centroids]
    Y_pred = np.reshape(np.array(dist), (len(X), len(projected_centroids)))
    return Y_pred.argmin(axis=1) + 1  # Adjust labels to match the dataset

predicted_labels = lda_predict(test_data, projected_centroids, W)

accuracy = np.mean(predicted_labels == test_labels) * 100
print(f'Accuracy of LDA: {accuracy:.2f}%')