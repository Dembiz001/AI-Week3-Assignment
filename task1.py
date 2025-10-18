# =============================================================================
# TASK 1: Classical ML with Scikit-learn - Iris Species Dataset
# =============================================================================

# Import required libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

print("🚀 Starting Task 1: Iris Species Classification")

# =============================================================================
# GOAL 1: Preprocess the data (handle missing values, encode labels)
# =============================================================================

# Load the dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Labels (already encoded as 0,1,2)

print("📊 Original Data Info:")
print(f"Features: {iris.feature_names}")
print(f"Target names: {iris.target_names}")
print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Check for missing values
print(f"\n🔍 Checking for missing values...")
print(f"Missing values in features: {np.isnan(X).sum()}")
print(f"Missing values in target: {np.isnan(y).sum()}")

# Check if labels need encoding (they're already encoded as 0,1,2)
print(f"\n📝 Label encoding status:")
print(f"Unique labels: {np.unique(y)}")
print(f"Label names: {iris.target_names}")
print("✅ Labels are already encoded - no encoding needed")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3, 
    random_state=42
)

print(f"\n🎯 Data split completed:")
print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

# =============================================================================
# GOAL 2: Train a decision tree classifier to predict iris species
# =============================================================================

print("\n🌳 Training Decision Tree Classifier...")

# Create the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the model
clf.fit(X_train, y_train)

print("✅ Model training completed!")

# =============================================================================
# GOAL 3: Evaluate using accuracy, precision, and recall
# =============================================================================

print("\n📊 Evaluating model performance...")

# Make predictions
y_pred = clf.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

# Display results
print("🎯 EVALUATION RESULTS:")
print("=" * 40)
print(f"Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print("=" * 40)

# Show some prediction examples
print("\n🔮 Prediction Examples:")
print("Actual vs Predicted (First 15 samples):")
for i in range(15):
    actual_name = iris.target_names[y_test[i]]
    predicted_name = iris.target_names[y_pred[i]]
    correct = "✓" if y_test[i] == y_pred[i] else "✗"
    print(f"Sample {i+1:2d}: Actual={actual_name:9s} Predicted={predicted_name:9s} {correct}")

# Final summary
print(f"\n🎉 TASK 1 COMPLETED!")
print(f"✅ Data preprocessing: Checked for missing values and label encoding")
print(f"✅ Model trained: Decision Tree Classifier")
print(f"✅ Evaluation: Accuracy={accuracy*100:.2f}%, Precision={precision:.4f}, Recall={recall:.4f}")
