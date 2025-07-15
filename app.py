import streamlit as st
from PIL import Image
import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification
import pandas as pd
from datetime import datetime
import os

# ✅ Load Model and Extractor

model_name = "wambugu71/crop_leaf_diseases_vit"
extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)


#💊 Disease Treatment Guide

treatment_guide = {
    "Corn___Common_rust": (
        "1. Apply fungicides early (e.g., azoxystrobin, pyraclostrobin).\n"
        "2. Practice crop rotation and avoid monoculture.\n"
        "3. Use rust-resistant corn hybrids.\n"
        "4. Destroy infected crop residues after harvest."
    ),
    "Corn___Healthy": "✅ No treatment needed. The crop is healthy.",

    "Potato___Early_blight": (
        "1. Apply preventive fungicides (chlorothalonil, mancozeb).\n"
        "2. Ensure proper field drainage to avoid excessive moisture.\n"
        "3. Remove infected leaves to reduce spore spread.\n"
        "4. Use disease-free certified seed potatoes."
    ),
    "Potato___Late_blight": (
        "1. Use systemic fungicides (metalaxyl, cymoxanil).\n"
        "2. Avoid overhead irrigation and dense planting.\n"
        "3. Remove infected plants immediately.\n"
        "4. Use resistant potato cultivars."
    ),

    "Rice___Brown_Spot": (
        "1. Treat seeds with fungicides (carbendazim or thiram).\n"
        "2. Avoid excessive nitrogen fertilization.\n"
        "3. Use resistant rice varieties.\n"
        "4. Ensure proper water drainage and field hygiene."
    ),
    "Rice___Healthy": "✅ No treatment needed. The crop is healthy.",

    "Wheat___Leaf_rust": (
        "1. Apply foliar fungicides (propiconazole, tebuconazole).\n"
        "2. Use resistant wheat varieties.\n"
        "3. Destroy volunteer wheat and weed hosts.\n"
        "4. Monitor early signs and act promptly."
    ),
    "Wheat___Healthy": "✅ No treatment needed. The crop is healthy."
}

# 🖼️ Streamlit UI

st.title("🌿 Crop Disease Detection (Real-Time)")
st.write("Upload or capture a leaf image to detect the disease and get treatment advice.")

# Choose image source
source = st.radio("Choose image source:", ["Upload Image", "Use Camera"])

# Load image
image = None
if source == "Upload Image":
    uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
elif source == "Use Camera":
    camera_image = st.camera_input("Take a picture of the leaf")
    if camera_image:
        image = Image.open(camera_image).convert("RGB")

# Run prediction
if image:
    st.image(image, caption="Leaf Image", use_column_width=True)
    st.write("🔍 Analyzing...")

    inputs = extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        class_name = model.config.id2label[pred_idx]
        confidence = probs[0][pred_idx].item()

    st.success(f"🩺 **Predicted Disease:** {class_name}")
    st.info(f"📊 **Confidence:** {confidence:.2%}")

    # Show treatment advice
    advice = treatment_guide.get(class_name, "⚠️ No treatment advice available.")
    st.warning(f"💊 **Treatment Advice:**\n\n{advice}")

    # Save to prediction history
    history_file = "prediction_history.csv"
    log_entry = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "predicted_class": class_name,
        "confidence (%)": round(confidence * 100, 2)
    }])

    if os.path.exists(history_file):
        log_entry.to_csv(history_file, mode='a', header=False, index=False)
    else:
        log_entry.to_csv(history_file, mode='w', header=True, index=False)

# ----------------------------
# 📜 Sidebar: Show Prediction History
# ----------------------------
st.sidebar.title("📋 Prediction History")
if st.sidebar.button("Show Log"):
    if os.path.exists("prediction_history.csv"):
        df = pd.read_csv("prediction_history.csv")
        st.sidebar.dataframe(df.tail(10))
    else:
        st.sidebar.info("No prediction history found.")
