# K-Means Clustering
# This algorithm groups similar data points together into k clusters.
# Each cluster is represented by its centroid (center point).

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

try:
    # Step 1: Load and prepare our data
    print("🌸 Loading the Iris dataset...")
    iris = load_iris()
    X = iris.data
    
    # Step 2: Scale the data
    print("📏 Scaling the data...")
    X_scaled = StandardScaler().fit_transform(X)
    
    # Step 3: Create and train K-means model
    print("🔧 Performing K-means clustering...")
    kmeans = KMeans(
        n_clusters=3,        # Number of clusters
        random_state=42,     # For reproducibility
        n_init=10           # Number of times to run with different centroid seeds
    )
    
    # Step 4: Get cluster assignments
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Step 5: Evaluate clustering quality
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    print(f"\n📊 Clustering Quality:")
    print(f"Silhouette Score: {silhouette_avg:.3f} (-1 to 1, higher is better)")
    print(f"Inertia (Within-cluster sum of squares): {kmeans.inertia_:.3f}")
    
    # Step 6: Visualize results
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], 
                         c=cluster_labels, 
                         cmap='viridis',
                         alpha=0.6)
    
    # Plot centroids
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], 
               marker='x', s=200, linewidths=3, 
               color='r', label='Centroids')
    
    plt.colorbar(scatter, label='Cluster Labels')
    plt.title("K-means Clustering of Iris Dataset")
    plt.xlabel("Feature 1 (Scaled)")
    plt.ylabel("Feature 2 (Scaled)")
    plt.legend()
    
    print("🎉 Done! Close the plot window to continue...")
    plt.show()

except Exception as e:
    print(f"❌ An error occurred: {str(e)}")









"""
---Output---
🌸 Loading the Iris dataset...
📏 Scaling the data...
🔧 Performing K-means clustering...

📊 Clustering Quality:
Silhouette Score: 0.460 (-1 to 1, higher is better)
Inertia (Within-cluster sum of squares): 139.820
🎉 Done! Close the plot window to continue...


"""