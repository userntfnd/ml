# ml
# Expt 9 pca
```
import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # Compute mean of the data
        self.mean = np.mean(X, axis=0)

        # Center the data
        centered_data = X - self.mean

        # Compute covariance matrix
        cov_matrix = np.cov(centered_data, rowvar=False)

        # Compute eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort eigenvectors based on eigenvalues
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        # Select top n_components eigenvectors
        self.components = sorted_eigenvectors[:, :self.n_components]

    def transform(self, X):
        # Center the data
        centered_data = X - self.mean

        # Project data onto principal components
        transformed_data = np.dot(centered_data, self.components)

        return transformed_data

# Example usage
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

pca = PCA(n_components=2)
pca.fit(data)
transformed_data = pca.transform(data)

print("Original Data:")
print(data)
print("\nTransformed Data:")
print(transformed_data)



from sklearn.decomposition import PCA
import numpy as np

# Sample data
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Create PCA instance
pca = PCA(n_components=2)

# Fit the model with data
pca.fit(data)

# Transform the data onto the new feature space
transformed_data = pca.transform(data)

print("Original Data:")
print(data)
print("\nTransformed Data:")
print(transformed_data)
```

