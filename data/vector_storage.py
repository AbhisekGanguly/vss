# VSS/data/vector_storage.py
import numpy as np
import pickle

class VectorStorage:
    def __init__(self):
        self.vectors = []

    def add_vector(self, vector):
        self.vectors.append(vector)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.vectors, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.vectors = pickle.load(f)

    def get_all_vectors(self):
        return np.array(self.vectors)