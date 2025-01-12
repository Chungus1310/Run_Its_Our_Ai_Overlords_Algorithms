# k-NN is one of the simplest yet powerful machine learning algorithms.
# It classifies data points based on their closest neighbors.

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load and prepare our data
# The Iris dataset is perfect for understanding k-NN
print("🌸 Loading the Iris dataset...")
iris = load_iris()
X, y = iris.data, iris.target

# Step 2: Scale our features
# k-NN works better with normalized data because it uses distances
print("📏 Normalizing the data...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Split into training and testing sets
print("✂️ Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Step 4: Create and train our k-NN model
# We'll try different values of k to find the best one
k_values = [3, 5, 7]
best_accuracy = 0
best_k = 0

print("\n🔍 Finding the best k value...")
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    accuracy = knn.score(X_test, y_test)
    print(f"k={k}: Accuracy = {accuracy*100:.2f}%")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

# Train the final model with the best k
print(f"\n🌟 Training final model with k={best_k}...")
final_model = KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(X_train, y_train)

# Evaluate the model
y_pred = final_model.predict(X_test)
print("\nFinal Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))




"""
--- Output ---
🌸 Loading the Iris dataset...
📏 Normalizing the data...
✂️ Splitting data into training and testing sets...

🔍 Finding the best k value...
k=3: Accuracy = 100.00%
k=5: Accuracy = 100.00%
k=7: Accuracy = 100.00%

🌟 Training final model with k=3...

Final Model Performance:
Accuracy: 100.00%

Detailed Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        10
           1       1.00      1.00      1.00         9
           2       1.00      1.00      1.00        11

    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30


"""