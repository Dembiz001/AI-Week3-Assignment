# mnist_app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import time

# Set page configuration
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        border-left: 5px solid #1f77b4;
    }
    .confidence-high {
        color: #00cc00;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ff9900;
        font-weight: bold;
    }
    .confidence-low {
        color: #ff0000;
        font-weight: bold;
    }
    .upload-box {
        border: 2px dashed #1f77b4;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the pre-trained MNIST model"""
    try:
        # If you have a saved model from Task 2, load it like this:
        # model = tf.keras.models.load_model('mnist_model.h5')
        
        # For demo purposes, we'll create a simple model (replace with your actual model)
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image):
    """Preprocess uploaded image for MNIST classification"""
    # Convert to grayscale if needed
    if image.mode != 'L':
        image = image.convert('L')
    
    # Resize to 28x28
    image = image.resize((28, 28))
    
    # Convert to numpy array and normalize
    image_array = np.array(image) / 255.0
    
    # Invert colors if background is dark (MNIST has white digits on black background)
    if np.mean(image_array) > 0.5:  # If background is light
        image_array = 1 - image_array
    
    # Reshape for model (add batch dimension and channel dimension)
    image_array = image_array.reshape(1, 28, 28, 1)
    return image_array

def create_sample_digit(digit):
    """Create a sample digit image for demonstration"""
    fig, ax = plt.subplots(figsize=(2, 2))
    
    # Create a simple digit-like pattern
    sample_digit = np.zeros((28, 28))
    
    # Different patterns for different digits
    if digit == 0:
        sample_digit[8:20, 8:20] = 0.8  # Circle-like
    elif digit == 1:
        sample_digit[5:23, 12:16] = 0.8  # Vertical line
    elif digit == 2:
        sample_digit[8:12, 6:22] = 0.8  # Horizontal top
        sample_digit[8:20, 18:22] = 0.8  # Right vertical
        sample_digit[18:22, 6:22] = 0.8  # Horizontal bottom
    else:
        sample_digit[10:18, 10:18] = 0.8  # Simple blob
    
    ax.imshow(sample_digit, cmap='gray')
    ax.axis('off')
    plt.tight_layout()
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
    buf.seek(0)
    return Image.open(buf)

def main():
    # Header
    st.markdown('<h1 class="main-header">🔢 MNIST Handwritten Digit Classifier</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose App Mode", 
                                   ["Upload Image", "Try Samples", "Model Info"])
    
    st.sidebar.title("About")
    st.sidebar.info(
        "This app uses a neural network trained on the MNIST dataset to classify "
        "handwritten digits (0-9). Upload an image of a digit or try the sample images!"
    )
    
    # Load model
    model = load_model()
    
    if app_mode == "Upload Image":
        upload_image_interface(model)
    elif app_mode == "Try Samples":
        sample_images_interface(model)
    elif app_mode == "Model Info":
        model_info_interface()

def upload_image_interface(model):
    """Interface for uploading custom images"""
    st.header("📁 Upload Your Digit Image")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=["png", "jpg", "jpeg", "bmp"],
            help="Upload a clear image of a handwritten digit (0-9). "
                 "The app will automatically resize and preprocess it."
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image', use_column_width=True)
                
                # Process and predict
                if st.button("🔍 Classify Digit", type="primary"):
                    with st.spinner('Processing image and making prediction...'):
                        processed_image = preprocess_image(image)
                        prediction = model.predict(processed_image, verbose=0)
                        predicted_digit = np.argmax(prediction)
                        confidence = np.max(prediction)
                        
                        display_prediction_results(image, processed_image, predicted_digit, confidence, prediction[0])
                        
            except Exception as e:
                st.error(f"Error processing image: {e}")
        else:
            st.info("👆 Please upload an image file to get started")
    
    with col2:
        if uploaded_file is not None:
            st.info("📝 **Instructions:**")
            st.write("- Image should contain a single digit (0-9)")
            st.write("- Digit should be centered and clear")
            st.write("- Works best with white digits on dark background")
        else:
            st.info("🖼️ **Supported formats:** PNG, JPG, JPEG, BMP")

def sample_images_interface(model):
    """Interface for trying sample digit images"""
    st.header("🎯 Try Sample Digits")
    
    st.write("Select a digit to see how the model classifies it:")
    
    # Create sample digits for all numbers 0-9
    cols = st.columns(5)
    sample_digits = []
    
    for i in range(10):
        with cols[i % 5]:
            if st.button(f"Digit {i}", key=f"sample_{i}"):
                sample_digits.append(i)
    
    if sample_digits:
        digit = sample_digits[-1]  # Get the most recent one
        sample_image = create_sample_digit(digit)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(sample_image, caption=f'Sample Digit: {digit}', use_column_width=True)
        
        with col2:
            with st.spinner('Classifying digit...'):
                processed_image = preprocess_image(sample_image)
                prediction = model.predict(processed_image, verbose=0)
                predicted_digit = np.argmax(prediction)
                confidence = np.max(prediction)
                
                display_prediction_results(sample_image, processed_image, predicted_digit, confidence, prediction[0])

def model_info_interface():
    """Interface showing model information"""
    st.header("🤖 Model Information")
    
    st.subheader("Model Architecture")
    st.code("""
    Sequential([
        Flatten(input_shape=(28, 28, 1)),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(10, activation='softmax')
    ])
    """)
    
    st.subheader("Training Details")
    st.write("- **Dataset**: MNIST Handwritten Digits")
    st.write("- **Training samples**: 60,000")
    st.write("- **Test samples**: 10,000")
    st.write("- **Input shape**: 28×28 pixels (grayscale)")
    st.write("- **Output**: 10 classes (digits 0-9)")
    
    st.subheader("Performance")
    st.write("- **Typical accuracy**: >95% on test set")
    st.write("- **Training time**: ~2-5 minutes on CPU")
    st.write("- **Framework**: TensorFlow/Keras")

def display_prediction_results(original_image, processed_image, predicted_digit, confidence, probabilities):
    """Display prediction results in a formatted way"""
    
    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
    
    # Prediction result
    st.subheader("🎯 Prediction Result")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric(
            label="**Predicted Digit**",
            value=f"**{predicted_digit}**"
        )
    
    with col2:
        # Confidence level with color coding
        if confidence > 0.8:
            confidence_class = "confidence-high"
            confidence_text = "High Confidence"
        elif confidence > 0.6:
            confidence_class = "confidence-medium"
            confidence_text = "Medium Confidence"
        else:
            confidence_class = "confidence-low"
            confidence_text = "Low Confidence"
        
        st.markdown(f'**Confidence**: <span class="{confidence_class}">{confidence:.2%}</span>', unsafe_allow_html=True)
        st.write(f"*{confidence_text}*")
    
    # Confidence bar
    st.progress(float(confidence))
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Probability distribution
    st.subheader("📊 Prediction Probabilities")
    
    # Create columns for digit probabilities
    prob_cols = st.columns(5)
    for i in range(10):
        with prob_cols[i % 5]:
            is_predicted = "✓" if i == predicted_digit else ""
            delta = f"{is_predicted} {probabilities[i]:.2%}" if is_predicted else f"{probabilities[i]:.2%}"
            st.metric(
                label=f"Digit {i}",
                value=delta
            )
    
    # Visualization
    st.subheader("🖼️ Processing Visualization")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Original image
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Processed image (model input)
    axes[1].imshow(processed_image.reshape(28, 28), cmap='gray')
    axes[1].set_title('Processed Image (Model Input)')
    axes[1].axis('off')
    
    # Probability bar chart
    digits = range(10)
    colors = ['green' if j == predicted_digit else 'blue' for j in range(10)]
    bars = axes[2].bar(digits, probabilities, color=colors, alpha=0.7)
    axes[2].set_xlabel('Digits')
    axes[2].set_ylabel('Probability')
    axes[2].set_title('Prediction Probabilities')
    axes[2].set_xticks(digits)
    axes[2].set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, prob in zip(bars, probabilities):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{prob:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    st.pyplot(fig)

if __name__ == "__main__":
    main()
