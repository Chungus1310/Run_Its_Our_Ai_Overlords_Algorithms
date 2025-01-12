"""
Random Forests are like a team of decision trees working together to make predictions.
Each tree votes on the outcome, and the majority vote wins!
"""

# Import our needed tools
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Get our data ready
print("🌸 Loading the Iris flower dataset...")
iris = load_iris()
X, y = iris.data, iris.target

# Step 2: Split our data
# We keep some data hidden for testing - like holding back quiz questions
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42)

# Step 3: Create our Random Forest
# Think of this as creating a forest with 100 trees, each with a maximum depth of 3 levels
print("🌲 Growing our random forest...")
rf = RandomForestClassifier(
    n_estimators=100,  # Number of trees in our forest
    max_depth=3,       # How deep each tree can grow
    random_state=42    # For reproducible results
)

# Step 4: Train our forest
rf.fit(X_train, y_train)

# Step 5: Make predictions and evaluate
y_pred = rf.predict(X_test)

# Print friendly results
print("\n🎯 Let's see how our forest performed!")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Show feature importance
print("\n🌟 Feature Importance:")
for feature, importance in zip(iris.feature_names, rf.feature_importances_):
    print(f"{feature}: {importance:.4f}")






"""
---Output---
🌸 Loading the Iris flower dataset...
🌲 Growing our random forest...

🎯 Let's see how our forest performed!
Accuracy: 100.00%

🌟 Feature Importance:
sepal length (cm): 0.1062
sepal width (cm): 0.0099
petal length (cm): 0.4522
petal width (cm): 0.4317

"""