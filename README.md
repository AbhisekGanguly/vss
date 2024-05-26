# VSS (Vector Similarity Search)

## Overview
VSS is a Python library designed for efficient similarity search and clustering of dense vectors. It supports multiple distance metrics, various search algorithms, and clustering methods to cater to a wide range of applications.

## Features
- Multiple distance metrics:
  - Euclidean
  - Cosine
  - Manhattan
  - Hamming
  - Jaccard
- Brute-force search
- Inverted File Index (IVF)
- HNSW (Hierarchical Navigable Small World) Index
- KMeans and KMeans++ clustering
- Efficient vector storage
- Batch processing for large datasets

## Installation
To install the VSS library, simply use pip:
```
pip install VSS
```

## Usage

### Brute-Force Search
```python
from VSS import VectorData, BruteForceSearch
import numpy as np

# Create sample data
data_vectors = np.random.random((1000, 64)).astype('float32')
query_vector = np.random.random((64,)).astype('float32')

# Initialize vector data
vector_data = VectorData(data_vectors)

# Initialize brute-force search with Euclidean distance
bfs = BruteForceSearch(vector_data, distance_metric='euclidean')

# Perform search
indices, distances = bfs.search(query_vector, k=5)
print("Indices:", indices)
print("Distances:", distances)
```

### IVF Index
```python
from VSS import IVFIndex
import numpy as np

# Create sample data
data_vectors = np.random.random((1000, 64)).astype('float32')

# Initialize IVF index with 10 clusters
ivf = IVFIndex(n_clusters=10)

# Fit the index with data vectors
ivf.fit(data_vectors)

# Add additional vectors to the index
additional_vectors = np.random.random((100, 64)).astype('float32')
ivf.add(additional_vectors)

# Perform search
query_vector = np.random.random((64,)).astype('float32')
indices, distances = ivf.search(query_vector, k=5)
print("Indices:", indices)
print("Distances:", distances)
```

### HNSW Index
```python
from VSS import HNSWIndex
import numpy as np

# Create sample data
data_vectors = np.random.random((1000, 64)).astype('float32')

# Initialize HNSW index
hnsw = HNSWIndex()

# Fit the index with data vectors
hnsw.fit(data_vectors)

# Add additional vectors to the index
additional_vectors = np.random.random((100, 64)).astype('float32')
hnsw.add(additional_vectors)

# Perform search
query_vector = np.random.random((64,)).astype('float32')
indices, distances = hnsw.search(query_vector, k=5)
print("Indices:", indices)
print("Distances:", distances)
```

### KMeans++
```python
from VSS import KMeansPlusPlus
import numpy as np

# Create sample data
data_vectors = np.random.random((1000, 64)).astype('float32')

# Initialize KMeans++ with 10 clusters
kmeans = KMeansPlusPlus(n_clusters=10)

# Fit the KMeans++ algorithm with data vectors
kmeans.fit(data_vectors)

# Predict the cluster labels for the data vectors
labels = kmeans.predict(data_vectors)
print("Labels:", labels)
```

### Vector Storage
```python
from VSS import VectorStorage
import numpy as np

# Create a sample vector
vector = np.random.random((64,)).astype('float32')

# Initialize vector storage
storage = VectorStorage()

# Add the vector to storage
storage.add_vector(vector)

# Save the vectors to a file
storage.save('vectors.pkl')

# Load the vectors from a file
storage.load('vectors.pkl')

# Retrieve all stored vectors
vectors = storage.get_all_vectors()
print("Stored Vectors:", vectors)
```

### Batch Processing
```python
from VSS import BatchProcessor
import numpy as np

# Create sample data
data_vectors = np.random.random((1000, 64)).astype('float32')

# Define a dummy processing function
def process_batch(batch):
    print("Processing batch of size:", len(batch))

# Initialize batch processor with batch size 100
processor = BatchProcessor(batch_size=100)

# Process data vectors in batches
processor.process_batches(data_vectors, process_batch)
```

## Testing
To ensure the library works as expected, run the unit tests provided in the `tests` directory:
```
python -m unittest discover tests
```

## Distance Metrics

### Euclidean Distance
Measures the straight-line distance between two vectors.

### Cosine Similarity
Measures the cosine of the angle between two vectors, indicating their orientation rather than magnitude.

### Manhattan Distance
Measures the distance between two vectors by summing the absolute differences of their coordinates.

### Hamming Distance
Measures the number of positions at which the corresponding elements are different, used for binary vectors.

### Jaccard Similarity
Measures the similarity between two sets by dividing the size of their intersection by the size of their union.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request or open an issue.

## License
This project is licensed under the MIT License.

## Acknowledgements
This library was inspired by the need for efficient vector similarity search and clustering in various machine learning applications.