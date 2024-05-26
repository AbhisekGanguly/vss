# VSS/search/brute_force_search.py
import numpy as np
from data.vector_data import VectorData

class BruteForceSearch:
    def __init__(self, vector_data, distance_metric='euclidean'):
        self.vector_data = vector_data
        self.distance_metric = distance_metric

    def _calculate_distance(self, vector1, vector2):
        if self.distance_metric == 'euclidean':
            return VectorData.euclidean_distance(vector1, vector2)
        elif self.distance_metric == 'cosine':
            return 1 - VectorData.cosine_similarity(vector1, vector2)
        elif self.distance_metric == 'manhattan':
            return VectorData.manhattan_distance(vector1, vector2)
        elif self.distance_metric == 'jaccard':
            return 1 - VectorData.jaccard_similarity(vector1, vector2)
        elif self.distance_metric == 'hamming':
            return VectorData.hamming_distance(vector1, vector2)
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")

    def search(self, query_vector, k):
        distances = []
        for vector in self.vector_data.vectors:
            dist = self._calculate_distance(query_vector, vector)
            distances.append(dist)
        distances = np.array(distances)
        nearest_indices = distances.argsort()[:k]
        return nearest_indices, distances[nearest_indices]
