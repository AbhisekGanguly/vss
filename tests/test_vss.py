# tests/test_vss.py
import unittest
from VSS import VectorData, BruteForceSearch, KMeans, IVFIndex, HNSWIndex, KMeansPlusPlus, VectorStorage, BatchProcessor
import numpy as np

class TestVSS(unittest.TestCase):
    def setUp(self):
        self.data_vectors = np.random.random((1000, 64)).astype('float32')
        self.query_vector = np.random.random((64,)).astype('float32')
        self.vector_data = VectorData(self.data_vectors)

    def test_brute_force_search(self):
        bfs = BruteForceSearch(self.vector_data)
        indices, distances = bfs.search(self.query_vector, k=5)
        self.assertEqual(len(indices), 5)

    def test_ivf_index(self):
        ivf = IVFIndex(n_clusters=10)
        ivf.fit(self.data_vectors)
        ivf.add(np.random.random((100, 64)).astype('float32'))
        indices, distances = ivf.search(self.query_vector, k=5)
        self.assertEqual(len(indices), 5)

    def test_hnsw_index(self):
        hnsw = HNSWIndex()
        hnsw.fit(self.data_vectors)
        hnsw.add(np.random.random((100, 64)).astype('float32'))
        indices, distances = hnsw.search(self.query_vector, k=5)
        self.assertEqual(len(indices), 5)

    def test_kmeans_plus(self):
        kmeans = KMeansPlusPlus(n_clusters=10)
        kmeans.fit(self.data_vectors)
        labels = kmeans.predict(self.query_vector[np.newaxis, :])
        self.assertEqual(labels.shape, (1,))

    def test_vector_storage(self):
        storage = VectorStorage()
        storage.add_vector(self.query_vector)
        storage.save('test_vectors.pkl')
        storage.load('test_vectors.pkl')
        vectors = storage.get_all_vectors()
        self.assertEqual(vectors.shape[0], 1)

    def test_batch_processing(self):
        def dummy_func(batch):
            self.assertTrue(isinstance(batch, np.ndarray))

        processor = BatchProcessor(batch_size=100)
        processor.process_batches(self.data_vectors, dummy_func)

if __name__ == '__main__':
    unittest.main()
