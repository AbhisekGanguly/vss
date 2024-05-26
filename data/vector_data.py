# VSS/data/vector_data.py
import numpy as np

class VectorData:
    def __init__(self, vectors):
        self.vectors = np.array(vectors)

    def add_vector(self, vector):
        self.vectors = np.vstack([self.vectors, vector])

    # A method to calculate the Euclidean distance between two vectors
    @staticmethod
    def euclidean_distance(self, vector1, vector2):
        return np.linalg.norm(vector1 - vector2)

    # A method to calculate the cosine similarity between two vectors
    @staticmethod
    def cosine_similarity(self, vector1, vector2):
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

    # A method to calculate the Manhattan Distance between two vectors
    @staticmethod
    def manhattan_distance(self, vector1, vector2):
        return np.sum(np.abs(vector1 - vector2))

    # A method to calculate the Jaccard similarity between two vectors
    @staticmethod
    def jaccard_similarity(self, vector1, vector2):
        intersection = np.sum(np.minimum(vector1, vector2))
        union = np.sum(np.maximum(vector1, vector2))
        return intersection / union

    # A method to calculate the Hamming distance between two vectors
    @staticmethod
    def hamming_distance(self, vector1, vector2):
        return np.sum(vector1 != vector2)
