# VSS/search/hnsw_index.py
import numpy as np
import heapq
from data.vector_data import VectorData

class HNSWIndex:
    def __init__(self, m=16, ef=200, distance_metric='euclidean'):
        self.m = m
        self.ef = ef
        self.distance_metric = distance_metric
        self.graph = []
        self.entry_point = None

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

    def fit(self, vectors):
        self.graph = []
        # Implementation of building the HNSW graph
        # Create the entry point
        self.entry_point = vectors[0]
        
        # Add the entry point to the graph
        self.graph.append([self.entry_point, []])
        
        # Iterate over the remaining vectors
        for vector in vectors[1:]:
            # Find the nearest neighbors for the current vector
            nearest_neighbors = self._find_nearest_neighbors(vector)
            
            # Add the vector to the graph
            self.graph.append([vector, nearest_neighbors])
            
            # Update the nearest neighbors of the existing vectors
            self._update_nearest_neighbors(vector, nearest_neighbors)

    def add(self, vectors):
        # Adding vectors to the HNSW graph
        # Iterate over the vectors to be added
        for vector in vectors:
            # Find the nearest neighbors for the current vector
            nearest_neighbors = self._find_nearest_neighbors(vector)
            
            # Add the vector to the graph
            self.graph.append([vector, nearest_neighbors])
            
            # Update the nearest neighbors of the existing vectors
            self._update_nearest_neighbors(vector, nearest_neighbors)

    def search(self, query_vector, k):
        # Search for the nearest neighbors of the query vector
        # Initialize the priority queue for the search
        priority_queue = []
        heapq.heappush(priority_queue, (0, 0))

        # Initialize the list of visited nodes
        visited = set()

        # Initialize the list of nearest neighbors
        nearest_neighbors = []

        # Perform the search
        while priority_queue:
            # Get the top element from the priority queue
            dist, node_id = heapq.heappop(priority_queue)

            # Check if the node has already been visited
            if node_id in visited:
                continue

            # Mark the node as visited
            visited.add(node_id)

            # Get the vector corresponding to the node
            vector, neighbors = self.graph[node_id]

            # Calculate the distance between the query vector and the current vector
            dist = self._calculate_distance(query_vector, vector)

            # Add the current vector to the list of nearest neighbors
            nearest_neighbors.append((vector, dist))

            # Update the priority queue with the neighbors of the current vector
            for neighbor_id in neighbors:
                neighbor_vector, _ = self.graph[neighbor_id]
                neighbor_dist = self._calculate_distance(query_vector, neighbor_vector)
                heapq.heappush(priority_queue, (neighbor_dist, neighbor_id))
