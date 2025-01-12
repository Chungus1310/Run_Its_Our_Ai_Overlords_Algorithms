# This is one of the fundamental algorithms for binary classification problems.
# We'll use the famous Iris dataset to demonstrate how it works.

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris

# Step 1: Data Preparation
# The Iris dataset is perfect for beginners - it contains measurements of different iris flowers
iris = load_iris()
X, y = iris.data, iris.target

# Step 2: Data Preprocessing
# We normalize our data to make sure all features are on the same scale
# This is crucial for Logistic Regression to work well!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Split our data into training and testing sets
# We use 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Step 4: Create our Logistic Regression model
# We're using TensorFlow/Keras for a more modern implementation
model = Sequential([
    # Input layer matches our feature count, output uses sigmoid for binary classification
    Dense(3, activation='softmax', input_shape=(X_train.shape[1],))
])

# Step 5: Compile the model
# We use categorical_crossentropy for multi-class classification
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Step 6: Train the model
# Epochs determine how many times we'll look at the entire dataset
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=8,
    validation_split=0.2,
    verbose=1
)

# Step 7: Evaluate our model
y_pred = model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=1)
accuracy = accuracy_score(y_test, y_pred_classes)

# Print detailed results
print("\nModel Performance:")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred_classes))




"""
---Output---
Epoch 1/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 1s 19ms/step - accuracy: 0.3425 - loss: 1.0349 - val_accuracy: 0.4583 - val_loss: 0.9205
Epoch 2/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - accuracy: 0.3763 - loss: 0.9798 - val_accuracy: 0.4583 - val_loss: 0.8987
Epoch 3/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.5390 - loss: 0.9135 - val_accuracy: 0.5417 - val_loss: 0.8783
Epoch 4/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - accuracy: 0.4358 - loss: 0.9590 - val_accuracy: 0.5417 - val_loss: 0.8589
Epoch 5/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - accuracy: 0.5131 - loss: 0.9666 - val_accuracy: 0.5833 - val_loss: 0.8405
Epoch 6/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - accuracy: 0.5487 - loss: 0.8902 - val_accuracy: 0.6250 - val_loss: 0.8231
Epoch 7/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.5543 - loss: 0.8287 - val_accuracy: 0.6250 - val_loss: 0.8070
Epoch 8/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6250 - loss: 0.8246 - val_accuracy: 0.6250 - val_loss: 0.7913
Epoch 9/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6697 - loss: 0.7870 - val_accuracy: 0.6250 - val_loss: 0.7770
Epoch 10/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - accuracy: 0.6503 - loss: 0.7773 - val_accuracy: 0.6667 - val_loss: 0.7628
Epoch 11/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.5916 - loss: 0.7917 - val_accuracy: 0.6667 - val_loss: 0.7499
Epoch 12/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.5926 - loss: 0.8122 - val_accuracy: 0.7083 - val_loss: 0.7373
Epoch 13/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.6892 - loss: 0.7199 - val_accuracy: 0.7083 - val_loss: 0.7253
Epoch 14/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.6999 - loss: 0.7036 - val_accuracy: 0.7083 - val_loss: 0.7140
Epoch 15/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.6410 - loss: 0.7412 - val_accuracy: 0.7500 - val_loss: 0.7035
Epoch 16/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.6492 - loss: 0.7491 - val_accuracy: 0.7500 - val_loss: 0.6937
Epoch 17/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - accuracy: 0.6648 - loss: 0.7247 - val_accuracy: 0.7500 - val_loss: 0.6839
Epoch 18/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.6629 - loss: 0.7053 - val_accuracy: 0.7500 - val_loss: 0.6745
Epoch 19/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step - accuracy: 0.7493 - loss: 0.6321 - val_accuracy: 0.7500 - val_loss: 0.6659
Epoch 20/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step - accuracy: 0.7150 - loss: 0.6120 - val_accuracy: 0.7500 - val_loss: 0.6576
Epoch 21/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.7897 - loss: 0.5933 - val_accuracy: 0.7917 - val_loss: 0.6494
Epoch 22/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.6971 - loss: 0.6559 - val_accuracy: 0.7917 - val_loss: 0.6416
Epoch 23/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.7983 - loss: 0.5983 - val_accuracy: 0.7917 - val_loss: 0.6343
Epoch 24/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.7265 - loss: 0.6036 - val_accuracy: 0.7917 - val_loss: 0.6270
Epoch 25/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.7631 - loss: 0.5991 - val_accuracy: 0.7917 - val_loss: 0.6205
Epoch 26/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step - accuracy: 0.7220 - loss: 0.6154 - val_accuracy: 0.7917 - val_loss: 0.6138
Epoch 27/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - accuracy: 0.7458 - loss: 0.5931 - val_accuracy: 0.7917 - val_loss: 0.6077
Epoch 28/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step - accuracy: 0.7861 - loss: 0.5478 - val_accuracy: 0.7917 - val_loss: 0.6015
Epoch 29/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.7320 - loss: 0.6296 - val_accuracy: 0.7917 - val_loss: 0.5957
Epoch 30/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - accuracy: 0.7479 - loss: 0.6112 - val_accuracy: 0.7917 - val_loss: 0.5899
Epoch 31/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step - accuracy: 0.7256 - loss: 0.5740 - val_accuracy: 0.7917 - val_loss: 0.5847
Epoch 32/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.7174 - loss: 0.5808 - val_accuracy: 0.7917 - val_loss: 0.5798
Epoch 33/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.7474 - loss: 0.5460 - val_accuracy: 0.7917 - val_loss: 0.5748
Epoch 34/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.7638 - loss: 0.5479 - val_accuracy: 0.7917 - val_loss: 0.5696
Epoch 35/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - accuracy: 0.8061 - loss: 0.5373 - val_accuracy: 0.7917 - val_loss: 0.5646
Epoch 36/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.8004 - loss: 0.5177 - val_accuracy: 0.7917 - val_loss: 0.5598
Epoch 37/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - accuracy: 0.7315 - loss: 0.5220 - val_accuracy: 0.7917 - val_loss: 0.5557
Epoch 38/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - accuracy: 0.7465 - loss: 0.5575 - val_accuracy: 0.7917 - val_loss: 0.5510
Epoch 39/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.7673 - loss: 0.5268 - val_accuracy: 0.7917 - val_loss: 0.5467
Epoch 40/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step - accuracy: 0.8097 - loss: 0.5162 - val_accuracy: 0.7917 - val_loss: 0.5424
Epoch 41/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.7558 - loss: 0.5221 - val_accuracy: 0.7917 - val_loss: 0.5383
Epoch 42/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.8063 - loss: 0.4949 - val_accuracy: 0.7917 - val_loss: 0.5346
Epoch 43/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - accuracy: 0.7518 - loss: 0.4990 - val_accuracy: 0.7917 - val_loss: 0.5305
Epoch 44/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.7456 - loss: 0.5455 - val_accuracy: 0.7917 - val_loss: 0.5265
Epoch 45/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step - accuracy: 0.7687 - loss: 0.4897 - val_accuracy: 0.7917 - val_loss: 0.5227
Epoch 46/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.8176 - loss: 0.4375 - val_accuracy: 0.7917 - val_loss: 0.5190
Epoch 47/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.8431 - loss: 0.4724 - val_accuracy: 0.8333 - val_loss: 0.5151
Epoch 48/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.8606 - loss: 0.4727 - val_accuracy: 0.8333 - val_loss: 0.5121
Epoch 49/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.7799 - loss: 0.5028 - val_accuracy: 0.8333 - val_loss: 0.5090
Epoch 50/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.8083 - loss: 0.4682 - val_accuracy: 0.8333 - val_loss: 0.5053
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step

Model Performance:
Accuracy: 93.33%

Detailed Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        10
           1       1.00      0.78      0.88         9
           2       0.85      1.00      0.92        11

    accuracy                           0.93        30
   macro avg       0.95      0.93      0.93        30
weighted avg       0.94      0.93      0.93        30

"""