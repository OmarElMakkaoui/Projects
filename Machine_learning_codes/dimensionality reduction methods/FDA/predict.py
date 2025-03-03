import numpy as np

def predict(X, projected_centroid, W):
    
    projected_data = np.dot(X, W)
    distances = np.linalg.norm(projected_data[:, np.newaxis, :] - projected_centroid, axis=2)
    label = np.argmin(distances, axis=1)
    
    return label