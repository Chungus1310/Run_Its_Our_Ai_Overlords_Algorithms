# Principal Component Analysis (PCA)
# PCA is used for dimensionality reduction while preserving as much variance as possible.
# It finds the directions (principal components) where the data varies the most.

import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

try:
    # Step 1: Load and prepare our data
    print("🌸 Loading the Iris dataset...")
    iris = load_iris()
    X = iris.data
    
    # Step 2: Scale the data
    print("📏 Scaling the data...")
    X_scaled = StandardScaler().fit_transform(X)
    
    # Step 3: Create and train PCA model
    print("🔧 Performing PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Step 4: Calculate explained variance
    explained_variance = pca.explained_variance_ratio_
    print("\n📊 Explained variance ratio by components:")
    print(f"First component: {explained_variance[0]:.2%}")
    print(f"Second component: {explained_variance[1]:.2%}")
    print(f"Total variance explained: {sum(explained_variance):.2%}")
    
    # Step 5: Visualize results
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                         c=iris.target, 
                         cmap='viridis',
                         alpha=0.6)
    plt.colorbar(scatter, label='Iris Classes')
    plt.title("PCA Reduction of Iris Dataset")
    plt.xlabel(f"First Principal Component\n{explained_variance[0]:.1%} variance explained")
    plt.ylabel(f"Second Principal Component\n{explained_variance[1]:.1%} variance explained")
    
    print("🎉 Done! Close the plot window to continue...")
    plt.show()

except Exception as e:
    print(f"❌ An error occurred: {str(e)}")







"""
---Output---
🌸 Loading the Iris dataset...
📏 Scaling the data...
🔧 Performing PCA...

📊 Explained variance ratio by components:
First component: 72.96%
Second component: 22.85%
Total variance explained: 95.81%
🎉 Done! Close the plot window to continue...

"""