# VSS/clustering/kmeans.py
import numpy as np

class KMeans:
    def __init__(self, n_clusters, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, vectors):
        n_samples, n_features = vectors.shape
        # Initialize centroids randomly
        centroids = vectors[np.random.choice(n_samples, self.n_clusters, replace=False)]

        for _ in range(self.max_iter):
            # Assign vectors to the nearest centroid
            distances = np.linalg.norm(vectors[:, np.newaxis] - centroids, axis=2)
            labels = distances.argmin(axis=1)

            # Recompute centroids
            new_centroids = np.array([vectors[labels == i].mean(axis=0) for i in range(self.n_clusters)])
            if np.all(centroids == new_centroids):
                break
            centroids = new_centroids

        self.centroids = centroids
        self.labels = labels

    def predict(self, vectors):
        distances = np.linalg.norm(vectors[:, np.newaxis] - self.centroids, axis=2)
        return distances.argmin(axis=1)
