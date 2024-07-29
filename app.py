import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import io
import gdown

# Download the models
@st.cache_resource
def download_models():
    gdown.download('https://drive.google.com/uc?id=12TX1C2FlErD44aLEuwQriXWavt0LpA7L', 'Breast-Or-Not-Model.pth', quiet=True)
    gdown.download('https://drive.google.com/uc?id=1f9TQHMSEqr7_5rsPFUH9z3tPLtWc_zpT', 'Breast-Model.pth', quiet=True)

class BreastPipeline:
    def __init__(self, binary_model_path, b_model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the binary classifier model
        self.model = torch.load(binary_model_path, map_location=self.device)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Load the Breast detection model
        self.b_model = torch.load(b_model_path, map_location=self.device)
        self.b_model = self.b_model.to(self.device)
        self.b_model.eval()

        # Define the transformations
        self.transform = transforms.Compose([
            transforms.Resize(size=(128, 128)),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.ToTensor()
        ])
    def preprocess(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return self.transform(image).unsqueeze(0).to(self.device)

    def is_b_image(self, image_tensor):
        with torch.no_grad():
            output = self.model(image_tensor)
            prediction = torch.argmax(output, dim=1).item()
        return prediction == 0  # 0 indicates a breast image

    def detect_amd(self, image_tensor):
        with torch.no_grad():
            output = self.b_model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, prediction].item()
        return prediction, confidence

    def generate_gradcam(self, model, image_tensor, target_layer):
        model.eval()
        gradients = []
        activations = []

        def save_gradient(grad):
            gradients.append(grad)

        def forward_hook(module, input, output):
            output.register_hook(save_gradient)
            activations.append(output)
            return output

        hook = target_layer.register_forward_hook(forward_hook)
        output = model(image_tensor)
        hook.remove()

        target_class = torch.argmax(output, dim=1).item()
        model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()

        gradients = gradients[0].cpu().data.numpy()
        activations = activations[0].cpu().data.numpy()

        weights = np.mean(gradients, axis=(2, 3))
        gradcam = np.zeros(activations.shape[2:], dtype=np.float32)

        for i, w in enumerate(weights[0]):
            gradcam += w * activations[0, i, :, :]

        gradcam = np.maximum(gradcam, 0)
        gradcam = cv2.resize(gradcam, (128, 128))
        gradcam = gradcam - gradcam.min()
        gradcam = gradcam / gradcam.max()

        return gradcam

    def visualize_gradcam(self, image_path):
        image_tensor = self.preprocess(image_path)
        if self.is_b_image(image_tensor):
            gradcam = self.generate_gradcam(self.b_model, image_tensor, self.b_model.features[8])
            image = Image.open(image_path).convert('RGB')
            image = image.resize((128, 128))
            image_np = np.array(image)

            heatmap = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            cam_image = heatmap + np.float32(image_np) / 255
            cam_image = cam_image / np.max(cam_image)

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(image_np)
            axes[0].set_title("Original Image")
            axes[0].axis('off')

            axes[1].imshow(np.uint8(255 * cam_image))
            axes[1].set_title("Grad-CAM")
            axes[1].axis('off')

            return fig
        else:
            return None

    def run(self, image_path):
        image_tensor = self.preprocess(image_path)
        if self.is_b_image(image_tensor):
            amd_result, confidence = self.detect_amd(image_tensor)
            result_text = "Benign" if amd_result == 0 else "Malignant"
            return f"{result_text} with probability {confidence:.4f}", True
        else:
            return "Not a Breast Image", False
       

@st.cache_resource
def load_pipeline():
    download_models()
    return BreastPipeline(binary_model_path='Breast-Or-Not-Model.pth', b_model_path='Breast-Model.pth')

def show_breast_cancer_info():
    st.header("About Breast Cancer")
    st.write("""
    Breast cancer is a type of cancer that forms in the cells of the breasts. It can occur in both women and men, but it's far more common in women.

    Key facts about breast cancer:
    - Breast cancer is the most common cancer in women worldwide.
    - Early detection significantly improves the chances of successful treatment.
    - Regular screenings, including mammograms, are crucial for early detection.
    - Symptoms may include a lump in the breast, changes in breast shape or size, and skin changes.

    Risk factors:
    - Age (risk increases with age)
    - Family history of breast cancer
    - Genetic mutations (BRCA1 and BRCA2)
    - Personal history of breast conditions or cancer
    - Radiation exposure
    - Obesity
    - Alcohol consumption

    Types of breast cancer:
    - Ductal carcinoma in situ (DCIS)
    - Invasive ductal carcinoma
    - Invasive lobular carcinoma
    - Other less common types

    Remember, this app is not a substitute for professional medical advice. If you have concerns about breast cancer, consult a healthcare professional.
    """)

def main():
    st.title("Breast Image Analysis and Cancer Information")

    # Add a navigation menu
    menu = ["Home", "Analyze Image", "Breast Cancer Information"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.write("Welcome to the Breast Image Analysis app. This tool can help analyze breast images for potential abnormalities.")
        st.write("Please note that this tool is for educational purposes only and should not be used as a substitute for professional medical advice.")
        st.write("Use the menu on the left to navigate to the image analysis tool or to learn more about breast cancer.")

    elif choice == "Analyze Image":
        pipeline = load_pipeline()

        st.header("Breast Image Analysis")
        uploaded_file = st.file_uploader("Choose a breast image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            if st.button("Analyze Image"):
                # Save the uploaded file temporarily
                temp_file = "temp_image.jpg"
                with open(temp_file, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Run the pipeline
                result, is_breast_image = pipeline.run(temp_file)
                st.write(result)

                # Generate and display Grad-CAM
                if is_breast_image:
                    fig = pipeline.visualize_gradcam(temp_file)
                    if fig:
                        st.pyplot(fig)
                    else:
                        st.write("Could not generate Grad-CAM visualization.")
                else:
                    st.write("Grad-CAM visualization is not available for non-breast images.")

        st.warning("Remember: This tool is for educational purposes only. Always consult with a healthcare professional for medical advice.")

    elif choice == "Breast Cancer Information":
        show_breast_cancer_info()

if __name__ == "__main__":
    main()
