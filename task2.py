# =============================================================================
# TASK 2: Deep Learning with TensorFlow - MNIST Handwritten Digits
# =============================================================================

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import random

print("🚀 Starting Task 2: MNIST Handwritten Digits Classification")

# =============================================================================
# Load and Prepare the MNIST Dataset
# =============================================================================

print("\n📥 Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print("📊 Dataset Information:")
print(f"Training data shape: {x_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Testing data shape: {x_test.shape}")
print(f"Testing labels shape: {y_test.shape}")
print(f"Unique labels: {np.unique(y_train)}")

# =============================================================================
# Data Preprocessing
# =============================================================================

print("\n🔧 Preprocessing data...")

# Normalize pixel values from 0-255 to 0-1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Reshape data to add channel dimension (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print("✅ Data preprocessing completed:")
print(f"Training shape after reshaping: {x_train.shape}")
print(f"Testing shape after reshaping: {x_test.shape}")

# =============================================================================
# Build CNN Model Architecture
# =============================================================================

print("\n🏗️ Building CNN Model Architecture...")

model = keras.Sequential([
    # First Convolutional Block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    
    # Second Convolutional Block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Classification Block
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Regularization to prevent overfitting
    layers.Dense(10, activation='softmax')  # 10 classes for digits 0-9
])

print("✅ CNN Model Architecture:")
model.summary()

# =============================================================================
# Compile the Model
# =============================================================================

print("\n⚙️ Compiling the model...")
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ Model compiled with:")
print(f"Optimizer: Adam")
print(f"Loss function: Sparse Categorical Crossentropy")
print(f"Metrics: Accuracy")

# =============================================================================
# Train the Model
# =============================================================================

print("\n🎯 Training the model...")
print("Goal: Achieve >95% test accuracy")

history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=10,
    validation_split=0.1,  # Use 10% of training data for validation
    verbose=1
)

print("✅ Model training completed!")

# =============================================================================
# Evaluate the Model
# =============================================================================

print("\n📊 Evaluating model on test set...")
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

print("🎯 TEST SET RESULTS:")
print("=" * 50)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

# Check if we achieved the goal
if test_accuracy > 0.95:
    print("🎉 SUCCESS: Achieved >95% test accuracy!")
else:
    print("⚠️  Goal not yet achieved. Consider training for more epochs or tuning hyperparameters.")

print("=" * 50)

# =============================================================================
# Visualize Model's Predictions on 5 Sample Images
# =============================================================================

print("\n🖼️ Visualizing predictions on 5 sample images...")

# Select 5 random test samples
sample_indices = random.sample(range(len(x_test)), 5)

# Create visualization
plt.figure(figsize=(15, 8))

for i, idx in enumerate(sample_indices):
    # Get the sample
    sample_image = x_test[idx]
    true_label = y_test[idx]
    
    # Make prediction
    prediction = model.predict(np.expand_dims(sample_image, axis=0), verbose=0)
    predicted_label = np.argmax(prediction)
    confidence = np.max(prediction)
    
    # Plot the image
    plt.subplot(2, 5, i + 1)
    plt.imshow(sample_image.squeeze(), cmap='gray')
    plt.title(f'True: {true_label}, Pred: {predicted_label}\nConf: {confidence:.2f}', 
              color='green' if true_label == predicted_label else 'red')
    plt.axis('off')
    
    # Plot probability distribution
    plt.subplot(2, 5, i + 6)
    bars = plt.bar(range(10), prediction[0], color=['red' if j == predicted_label else 'blue' for j in range(10)])
    plt.xticks(range(10))
    plt.ylabel('Probability')
    plt.title('Prediction Probabilities')
    
    # Add value labels on bars
    for bar, prob in zip(bars, prediction[0]):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{prob:.2f}', ha='center', va='bottom', fontsize=8)

plt.suptitle('MNIST Digit Classification - Predictions on 5 Sample Images', fontsize=16)
plt.tight_layout()
plt.savefig('mnist_predictions.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# Training History Visualization
# =============================================================================

print("\n📈 Plotting training history...")

plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# Final Summary
# =============================================================================

print("\n" + "="*60)
print("🎉 TASK 2 COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"✅ CNN Model Built: {len(model.layers)} layers")
print(f"✅ Model Trained: {len(history.history['accuracy'])} epochs")
print(f"✅ Test Accuracy: {test_accuracy*100:.2f}%")
print(f"✅ Goal Achieved: {'YES' if test_accuracy > 0.95 else 'NO'}")
print(f"✅ Predictions Visualized: 5 sample images")
print("="*60)

# Additional: Show some misclassified examples for analysis
print("\n🔍 Analyzing misclassifications (if any)...")
predictions = model.predict(x_test, verbose=0)
predicted_labels = np.argmax(predictions, axis=1)

# Find misclassified samples
misclassified = np.where(predicted_labels != y_test)[0]
if len(misclassified) > 0:
    print(f"Number of misclassified samples: {len(misclassified)}/{len(y_test)}")
    print("First 5 misclassified examples:")
    
    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(misclassified[:5]):
        plt.subplot(1, 5, i + 1)
        plt.imshow(x_test[idx].squeeze(), cmap='gray')
        plt.title(f'True: {y_test[idx]}, Pred: {predicted_labels[idx]}')
        plt.axis('off')
    plt.suptitle('Misclassified Examples', fontsize=16)
    plt.tight_layout()
    plt.show()
else:
    print("🎉 No misclassifications - perfect performance!")
