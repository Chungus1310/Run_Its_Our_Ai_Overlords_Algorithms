"""
Naive Bayes is like a probability calculator - it predicts by calculating
the likelihood of each possible outcome based on the features it sees.
"""

# Import our friendly tools
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load our data
print("🌸 Loading the Iris dataset...")
iris = load_iris()
X, y = iris.data, iris.target

# Step 2: Split our dataset
# Like dealing cards, we shuffle and split our data
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42)

# Step 3: Create and train our Naive Bayes model
print("🧮 Training our Naive Bayes classifier...")
nb = GaussianNB()  # We use Gaussian NB because our features are continuous
nb.fit(X_train, y_train)

# Step 4: Make predictions
y_pred = nb.predict(X_test)

# Step 5: Evaluate our model
accuracy = accuracy_score(y_test, y_pred)
print("\n📊 Results:")
print(f"Accuracy: {accuracy * 100:.2f}%")

# Create a nice confusion matrix visualization
print("\n📈 Plotting confusion matrix...")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()




"""
---Output---
🌸 Loading the Iris dataset...
🧮 Training our Naive Bayes classifier...

📊 Results:
Accuracy: 100.00%

📈 Plotting confusion matrix...

"""