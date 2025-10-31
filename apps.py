# streamlit_app.py

import streamlit as st
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import copy

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, transforms

# Pennylane
import pennylane as qml
from pennylane import numpy as np

# Plotting
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# Set page configuration
st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        background-color: #f0f2f6;
    }
    .confidence-bar {
        height: 20px;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        border-radius: 10px;
        margin: 5px 0;
    }
    .model-info {
        background-color: #e8f4fd;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🧠 Brain Tumor Classification</h1>', unsafe_allow_html=True)
st.markdown("""
This app uses a **hybrid quantum-classical neural network** to classify brain MRI images 
as either **Tumor** or **No Tumor**. The model combines ResNet18 with a quantum circuit 
for enhanced feature processing.
""")

# Sidebar for model information
with st.sidebar:
    st.header("Model Information")
    st.markdown("""
    **Architecture:**
    - ResNet18 (classical backbone)
    - Quantum variational circuit (4 qubits)
    - Fully connected layers
    
    **Training Results:**
    - Best accuracy: 94.29%
    - Best loss: 0.1920
    
    **Classes:**
    - No Tumor
    - Tumor
    """)
    
    st.markdown("---")
    st.header("About")
    st.markdown("""
    This hybrid model demonstrates the potential of 
    quantum machine learning for medical image analysis.
    
    Built with:
    - PyTorch
    - PennyLane
    - Streamlit
    """)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
st.sidebar.markdown(f"**Device:** {device}")

# Define the quantum components (same as original code)
@st.cache_resource
def setup_quantum_components():
    n_qubits = 4
    q_depth = 6
    q_delta = 0.005
    
    dev = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(dev, interface="torch")
    def quantum_net(q_input_features, q_weights):
        qml.AngleEmbedding(q_input_features, wires=range(n_qubits))
        qml.StronglyEntanglingLayers(q_weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(position)) for position in range(n_qubits)]
    
    return quantum_net, n_qubits, q_depth, q_delta

# Define the DressedQuantumNet class
class DressedQuantumNet(nn.Module):
    def __init__(self, n_qubits, q_depth, q_delta, quantum_net):
        super().__init__()
        self.n_qubits = n_qubits
        self.quantum_net = quantum_net
        self.pre_net = nn.Sequential(
            nn.Linear(512, 128), nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(128, n_qubits))
        self.q_params = nn.Parameter(q_delta * torch.randn(q_depth, n_qubits, 3))
        self.post_net = nn.Sequential(
            nn.Linear(n_qubits, 64), nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(64, 32), nn.GELU(), 
            nn.Dropout(0.2),
            nn.Linear(32, 2))

    def forward(self, input_features):
        self.q_params = self.q_params.to(device)
        input_features = input_features.to(device)
        pre_out = self.pre_net(input_features)
        q_in = torch.tanh(pre_out) * np.pi / 2.0
        q_out = torch.stack([
            torch.hstack(self.quantum_net(elem, self.q_params)) 
            for elem in q_in
        ]).float()
        return self.post_net(q_out)

# Load model function
@st.cache_resource
def load_model():
    try:
        # Initialize quantum components
        quantum_net, n_qubits, q_depth, q_delta = setup_quantum_components()
        
        # Create model architecture
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        model = torchvision.models.resnet18(weights=weights)
        
        # Freeze early layers
        for name, param in model.named_parameters():
            if "layer4" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # Replace final layer with quantum net
        model.fc = DressedQuantumNet(n_qubits, q_depth, q_delta, quantum_net)
        model = model.to(device)
        
        # Load trained weights
        if os.path.exists('best_model.pth'):
            model.load_state_dict(torch.load('best_model.pth', map_location=device))
            model.eval()
            return model
        else:
            st.error("Model file 'best_model.pth' not found. Please ensure it's in the same directory.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

# Prediction function
def predict_image(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        return predicted.item(), confidence.item(), probabilities.cpu().numpy()[0]

# Main app
def main():
    # Model loading section
    st.markdown("---")
    st.header("🚀 Model Deployment")
    
    with st.spinner("Loading quantum-classical model..."):
        model = load_model()
    
    if model is None:
        st.error("Failed to load model. Please check if 'best_model.pth' exists.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Image upload section
    st.markdown("---")
    st.header("📁 Upload MRI Image")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a brain MRI image", 
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload a brain MRI scan for tumor classification"
        )
    
    with col2:
        st.markdown("""
        **Supported formats:** JPG, JPEG, PNG, BMP
        
        **Expected input:** Brain MRI scans
        - Axial view preferred
        - Clear contrast
        - Minimal artifacts
        """)
    
    # Display and process uploaded image
    if uploaded_file is not None:
        try:
            # Load and display image
            image = Image.open(uploaded_file).convert('RGB')
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Uploaded Image")
                st.image(image, caption="Uploaded MRI Scan", use_column_width=True)
                
                # Image info
                st.markdown(f"""
                **Image Details:**
                - Format: {uploaded_file.type}
                - Size: {image.size}
                - Mode: {image.mode}
                """)
            
            with col2:
                st.subheader("🔍 Analysis Results")
                
                # Preprocess and predict
                with st.spinner("Analyzing image with quantum-classical model..."):
                    processed_image = preprocess_image(image).to(device)
                    prediction, confidence, probabilities = predict_image(model, processed_image)
                
                # Class names
                class_names = ['No Tumor', 'Tumor']
                result = class_names[prediction]
                
                # Display prediction
                if prediction == 1:  # Tumor
                    st.error(f"## 🚨 Prediction: {result}")
                    st.markdown("""
                    ⚠️ **Clinical Note:** This prediction indicates potential abnormalities. 
                    Please consult with a medical professional for proper diagnosis.
                    """)
                else:  # No Tumor
                    st.success(f"## ✅ Prediction: {result}")
                    st.markdown("""
                    💡 **Note:** This AI assessment suggests no tumor detected. 
                    Always follow up with qualified healthcare providers.
                    """)
                
                # Confidence metrics
                st.subheader("📊 Confidence Levels")
                
                for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
                    col_prob, col_bar = st.columns([2, 3])
                    
                    with col_prob:
                        st.write(f"{class_name}: {prob*100:.2f}%")
                    
                    with col_bar:
                        st.progress(float(prob))
                
                # Raw probabilities
                with st.expander("View Detailed Probabilities"):
                    st.write("Class probabilities from the quantum-classical model:")
                    for class_name, prob in zip(class_names, probabilities):
                        st.write(f"- {class_name}: {prob*100:.4f}%")
                
                # Model interpretation
                st.markdown("---")
                st.subheader("🤖 Model Interpretation")
                
                interpretation_text = """
                **How the hybrid model works:**
                1. **Classical Processing:** ResNet18 extracts spatial features from the MRI
                2. **Quantum Encoding:** Features are mapped to quantum state rotations
                3. **Quantum Processing:** Variational quantum circuit processes information
                4. **Classical Readout:** Quantum measurements are interpreted for classification
                
                **Note:** This is a research prototype. Always verify with medical experts.
                """
                st.markdown(interpretation_text)
        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.info("Please try with a different image format or check the file integrity.")
    
    else:
        # Demo section when no image is uploaded
        st.markdown("---")
        st.header("🎯 How to Use")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 1. Upload Image
            Click 'Browse files' and select a brain MRI image in JPG, PNG, or BMP format.
            """)
        
        with col2:
            st.markdown("""
            ### 2. Automatic Analysis
            The quantum-classical model will process the image and extract features.
            """)
        
        with col3:
            st.markdown("""
            ### 3. Get Results
            View the prediction with confidence scores and model interpretation.
            """)
        
        # Example images
        st.markdown("---")
        st.header("📸 Example MRI Scans")
        
        st.markdown("""
        For testing purposes, you can use sample brain MRI images from medical datasets.
        The model is trained to distinguish between:
        - **Normal brain tissue**
        - **Tumor-affected regions**
        """)
        
        # Technical details
        with st.expander("🔧 Technical Specifications"):
            st.markdown("""
            **Model Architecture:**
            - Backbone: ResNet18 (pretrained on ImageNet)
            - Quantum Circuit: 4 qubits, 6 layers
            - Quantum Gates: AngleEmbedding + StronglyEntanglingLayers
            - Classical Head: 2-layer MLP
            
            **Training Details:**
            - Dataset: Brain tumor MRI classification dataset
            - Classes: 2 (Tumor, No Tumor)
            - Best Accuracy: 94.29%
            - Framework: PyTorch + PennyLane
            
            **Quantum Advantage:**
            - Enhanced feature representation
            - Potential for better generalization
            - Novel approach to medical image analysis
            """)

if __name__ == "__main__":
    main()