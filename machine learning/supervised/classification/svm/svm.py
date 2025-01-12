"""
This script demonstrates how to use SVM for classification using the famous Iris dataset.
SVMs work by finding the best boundary (hyperplane) that separates different classes.
"""

# First, let's import all the tools we need
from sklearn.svm import SVC  # SVC stands for Support Vector Classification
from sklearn.datasets import load_iris  # We'll use the classic Iris flower dataset
from sklearn.model_selection import train_test_split  # Helps us split our data
from sklearn.preprocessing import StandardScaler  # Helps normalize our data
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load and prepare our data
# The Iris dataset contains measurements of 3 different types of Iris flowers
iris = load_iris()
X, y = iris.data, iris.target

# Step 2: Scale our data
# SVMs work much better with scaled data! This ensures all features are on the same scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Split into training and testing sets
# We use 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, 
                                                    test_size=0.2, 
                                                    random_state=42)

# Step 4: Create and train our SVM model
# We're using a linear kernel for simplicity, but 'rbf' often works better!
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)

# Step 5: Make predictions and evaluate our model
y_pred = svm.predict(X_test)

# Print detailed results
print("🎯 Model Performance Summary:")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))



"""
---Output---
🎯 Model Performance Summary:
Accuracy: 96.67%

Detailed Classification Report:
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       1.00      0.89      0.94         9
   virginica       0.92      1.00      0.96        11

    accuracy                           0.97        30
   macro avg       0.97      0.96      0.97        30
weighted avg       0.97      0.97      0.97        30

"""