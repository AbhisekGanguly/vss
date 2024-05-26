# VSS/clustering/kmeans_plus.py
import numpy as np

class KMeansPlusPlus:
    def __init__(self, n_clusters, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def _initialize_centroids(self, vectors):
        n_samples = vectors.shape[0]
        centroids = []
        centroids.append(vectors[np.random.choice(n_samples)])
        for _ in range(1, self.n_clusters):
            distances = np.min([np.linalg.norm(vectors - c, axis=1) for c in centroids], axis=0)
            prob = distances / distances.sum()
            cumulative_prob = np.cumsum(prob)
            r = np.random.rand()
            for j, p in enumerate(cumulative_prob):
                if r < p:
                    centroids.append(vectors[j])
                    break
        return np.array(centroids)

    def fit(self, vectors):
        centroids = self._initialize_centroids(vectors)
        for _ in range(self.max_iter):
            distances = np.linalg.norm(vectors[:, np.newaxis] - centroids, axis=2)
            labels = distances.argmin(axis=1)
            new_centroids = np.array([vectors[labels == i].mean(axis=0) for i in range(self.n_clusters)])
            if np.all(centroids == new_centroids):
                break
            centroids = new_centroids
        self.centroids = centroids
        self.labels = labels

    def predict(self, vectors):
        distances = np.linalg.norm(vectors[:, np.newaxis] - self.centroids, axis=2)
        return distances.argmin(axis=1)
