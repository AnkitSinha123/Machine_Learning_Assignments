import numpy as np

class KMeans:
    def __init__(self, k: int, epsilon: float = 1e-6) -> None:
        self.num_clusters = k
        self.cluster_centers = None
        self.epsilon = epsilon
    
    def fit(self, X: np.ndarray, max_iter: int = 100) -> None:
        # Initialize cluster centers (need to be careful with the initialization,
        # otherwise you might see that none of the pixels are assigned to some
        # of the clusters, which will result in a division by zero error)
        
        
        
        self.cluster_centers = np.unique(X,axis=0)
        sample_points_ids = np.random.choice(self.cluster_centers.shape[0],size=self.num_clusters,replace=False)
        self.cluster_centers =self.cluster_centers[sample_points_ids,:]
        
        
        for _ in range(max_iter):
            # Assign each sample to the closest prototype
            
            self.labels = self.predict(X)
            
            # Update prototypes
            
            new_clusters = np.array([X[self.labels == j].mean(axis=0) for j in range(self.num_clusters)])
            if (np.max(self.cluster_centers-new_clusters) < self.epsilon):
                break
            
            self.cluster_centers = new_clusters
            
            #pass

    
    def predict(self, X: np.ndarray) -> np.ndarray:
        # Predicts the index of the closest cluster center for each data point
        distances = np.linalg.norm(X[:, np.newaxis,:] - self.cluster_centers, axis=2)
        return np.argmin(distances, axis=1)
    
    def fit_predict(self, X: np.ndarray, max_iter: int = 100) -> np.ndarray:
        self.fit(X, max_iter)
        return self.predict(X)
    
    def replace_with_cluster_centers(self, X: np.ndarray) -> np.ndarray:
        # Returns an ndarray of the same shape as X
        # Each row of the output is the cluster center closest to the corresponding row in X
        
        
        new_X = np.zeros_like(X)
        for i, cluster_idx in enumerate(self.labels):
            new_X[i] = self.cluster_centers[cluster_idx]
    
        return new_X
        