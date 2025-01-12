# t-SNE (t-Distributed Stochastic Neighbor Embedding)
# This algorithm is great for visualizing high-dimensional data in 2D or 3D space.
# It's particularly good at preserving local relationships in the data.

import numpy as np
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

try:
    # Step 1: Load and prepare our data
    print("🌸 Loading the Iris dataset...")
    iris = load_iris()
    X = iris.data
    
    # Step 2: Scale our data - this is important because t-SNE is sensitive to scale!
    print("📏 Scaling the data to have zero mean and unit variance...")
    X_scaled = StandardScaler().fit_transform(X)
    
    # Step 3: Create and train the t-SNE model
    print("🔧 Creating t-SNE model...")
    tsne = TSNE(
        n_components=2,      # Reduce to 2D
        random_state=42,     # For reproducibility
        perplexity=30,       # Balance between local and global aspects
        n_iter=1000          # Number of iterations for optimization
    )
    
    # Step 4: Transform the data
    print("✨ Transforming the data using t-SNE...")
    X_tsne = tsne.fit_transform(X_scaled)
    
    # Step 5: Visualize the results
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                         c=iris.target, 
                         cmap='viridis',
                         alpha=0.6)
    plt.colorbar(scatter, label='Iris Classes')
    plt.title("t-SNE Visualization of Iris Dataset")
    plt.xlabel("First t-SNE Component")
    plt.ylabel("Second t-SNE Component")
    
    print("🎉 Done! Close the plot window to continue...")
    plt.show()

except Exception as e:
    print(f"❌ An error occurred: {str(e)}")



"""
---Output---
🌸 Loading the Iris dataset...
📏 Scaling the data to have zero mean and unit variance...
🔧 Creating t-SNE model...
✨ Transforming the data using t-SNE...
D:\Python\Lib\site-packages\sklearn\manifold\_t_sne.py:1164: FutureWarning: 'n_iter' was renamed to 'max_iter' in version 1.5 and will be removed in 1.7.
  warnings.warn(
🎉 Done! Close the plot window to continue...

"""