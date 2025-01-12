# Decision Trees are like playing a game of "20 Questions" with your data.
# They make decisions by splitting data based on features.

from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load our dataset
print("🌸 Loading the Iris dataset...")
iris = load_iris()
X, y = iris.data, iris.target

# Step 2: Split the data
# No need to scale data for Decision Trees - they're invariant to scaling!
print("✂️ Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: Create and train our Decision Tree
# We limit the depth to prevent overfitting
print("🌳 Growing our Decision Tree...")
dt = DecisionTreeClassifier(
    max_depth=3,
    min_samples_split=5,
    criterion='entropy',  # Using entropy for information gain
    random_state=42
)
dt.fit(X_train, y_train)

# Step 4: Make predictions and evaluate
print("\n📊 Evaluating our model...")
y_pred = dt.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Print results
print(f"\nModel Performance:")
print(f"Accuracy: {accuracy*100:.2f}%")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# Visualize the tree (optional but very educational!)
plt.figure(figsize=(20,10))
plot_tree(dt, feature_names=iris.feature_names, 
         class_names=iris.target_names, 
         filled=True, rounded=True)
plt.savefig('decision_tree.png')
print("\n🖼️ Decision tree visualization saved as 'decision_tree.png'")

# Feature importance
print("\n🎯 Feature Importance:")
for name, importance in zip(iris.feature_names, dt.feature_importances_):
    print(f"{name}: {importance*100:.2f}%")






"""
----Output----
🌸 Loading the Iris dataset...
✂️ Splitting data into training and testing sets...
🌳 Growing our Decision Tree...

📊 Evaluating our model...

PS C:\Users\Administrator\Desktop\machine learning\supervised\classification> & D:/Python/python.exe "c:/Users/Administrator/Desktop/machine learning/supervised/classification/DecisionTrees.py"
🌸 Loading the Iris dataset...
✂️ Splitting data into training and testing sets...
🌳 Growing our Decision Tree...

📊 Evaluating our model...

Model Performance:
Accuracy: 100.00%
PS C:\Users\Administrator\Desktop\machine learning\supervised\classification> & D:/Python/python.exe "c:/Users/Administrator/Desktop/machine learning/supervised/classification/DecisionTrees.py"
🌸 Loading the Iris dataset...
✂️ Splitting data into training and testing sets...
🌳 Growing our Decision Tree...

📊 Evaluating our model...

Model Performance:
PS C:\Users\Administrator\Desktop\machine learning\supervised\classification> & D:/Python/python.exe "c:/Users/Administrator/Desktop/machine learning/supervised/classification/DecisionTrees.py"
🌸 Loading the Iris dataset...
✂️ Splitting data into training and testing sets...
🌳 Growing our Decision Tree...

📊 Evaluating our model...

Model Performance:
Accuracy: 100.00%

🌸 Loading the Iris dataset...
✂️ Splitting data into training and testing sets...
🌳 Growing our Decision Tree...

📊 Evaluating our model...

Model Performance:
Accuracy: 100.00%


📊 Evaluating our model...

Model Performance:
Accuracy: 100.00%

Detailed Classification Report:
Model Performance:
Accuracy: 100.00%

Detailed Classification Report:
Detailed Classification Report:
              precision    recall  f1-score   support
              precision    recall  f1-score   support


           0       1.00      1.00      1.00        10
           1       1.00      1.00      1.00         9
           2       1.00      1.00      1.00        11

    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30


🖼️ Decision tree visualization saved as 'decision_tree.png'

🎯 Feature Importance:
sepal length (cm): 0.00%
sepal width (cm): 0.00%
petal length (cm): 91.81%
petal width (cm): 8.19%


"""