# VSS/data/batch_processing.py

import numpy as np

class BatchProcessor:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def process_batches(self, data, func):
        n_samples = len(data)
        for i in range(0, n_samples, self.batch_size):
            batch = data[i:i+self.batch_size]
            func(batch)
