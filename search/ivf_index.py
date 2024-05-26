# VSS/search/ivf_index.py
import numpy as np
from clustering.kmeans import KMeans

class IVFIndex:
    def __init__(self, n_clusters):
        self.kmeans = KMeans(n_clusters)
        self.inverted_lists = {}

    def fit(self, vectors):
        self.kmeans.fit(vectors)
        for i in range(self.kmeans.n_clusters):
            self.inverted_lists[i] = vectors[self.kmeans.labels == i]

    def add(self, vectors):
        labels = self.kmeans.predict(vectors)
        for i, vector in zip(labels, vectors):
            if i in self.inverted_lists:
                self.inverted_lists[i] = np.vstack([self.inverted_lists[i], vector])
            else:
                self.inverted_lists[i] = np.array([vector])

    def search(self, query_vector, k):
        cluster_idx = self.kmeans.predict(np.array([query_vector]))[0]
        candidates = self.inverted_lists.get(cluster_idx, np.array([]))
        if candidates.size == 0:
            return np.array([]), np.array([])

        # Brute-force search within the cluster
        distances = np.linalg.norm(candidates - query_vector, axis=1)
        nearest_indices = distances.argsort()[:k]
        return nearest_indices, distances[nearest_indices]
