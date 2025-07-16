# 🌿 Crop Leaf Disease Detection (Real-Time using ViT + Streamlit)

![demo](https://img.shields.io/badge/status-deployed-green)

Predict diseases in crop leaves (corn, potato, rice, wheat) using a Hugging Face Vision Transformer (ViT) model and get instant treatment advice.  
Supports real-time webcam input and file upload.

---

## 📸 App Features

- 📷 **Camera Input** + 📂 **Image Upload**
- 🧠 ViT model from Hugging Face (`wambugu71/crop_leaf_diseases_vit`)
- 💊 Real-world **treatment advice** for each disease
- 🕓 **Prediction history** (CSV logging)
- 📜 Easy-to-use Streamlit UI

---

## 🛠️ How to Run Locally

```bash
git clone https://github.com/RavindraM0/Crop-leaf-disease-detection.git
cd Crop-leaf-disease-detection

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
