# Hierarchical Clustering
# This algorithm creates a tree of clusters, showing how data points are grouped
# at different levels of granularity.

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler

try:
    # Step 1: Load and prepare our data
    print("🌸 Loading the Iris dataset...")
    iris = load_iris()
    X = iris.data
    
    # Step 2: Scale the data
    print("📏 Scaling the data...")
    X_scaled = StandardScaler().fit_transform(X)
    
    # Step 3: Create and train Hierarchical Clustering model
    print("🔧 Performing Hierarchical clustering...")
    agg_clustering = AgglomerativeClustering(
        n_clusters=3,            # Number of clusters
        linkage='ward'           # Linkage criteria (ward minimizes variance within clusters)
    )
    
    # Step 4: Get cluster assignments
    labels = agg_clustering.fit_predict(X_scaled)
    
    # Step 5: Create visualizations
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot clusters
    scatter = ax1.scatter(X_scaled[:, 0], X_scaled[:, 1], 
                         c=labels, 
                         cmap='viridis',
                         alpha=0.6)
    ax1.set_title("Hierarchical Clustering Results")
    ax1.set_xlabel("Feature 1 (Scaled)")
    ax1.set_ylabel("Feature 2 (Scaled)")
    plt.colorbar(scatter, ax=ax1, label='Cluster Labels')
    
    # Create dendrogram
    print("📊 Creating dendrogram...")
    linked_matrix = linkage(X_scaled, method='ward')
    dendrogram(linked_matrix, 
              ax=ax2,
              orientation='top',
              distance_sort='descending',
              show_leaf_counts=True,
              leaf_rotation=90)
    ax2.set_title("Hierarchical Clustering Dendrogram")
    ax2.set_xlabel("Sample Index")
    ax2.set_ylabel("Distance")
    
    plt.tight_layout()
    print("🎉 Done! Close the plot window to continue...")
    plt.show()

except Exception as e:
    print(f"❌ An error occurred: {str(e)}")





"""
---Output---
🌸 Loading the Iris dataset...
📏 Scaling the data...
🔧 Performing Hierarchical clustering...
📊 Creating dendrogram...
🎉 Done! Close the plot window to continue...


"""