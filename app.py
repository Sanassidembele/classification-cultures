import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Charger le modèle
model = tf.keras.models.load_model("modele_cultures_agricoles.keras")

# Classes à adapter selon ton dataset
class_names = ['Riz', 'Blé', 'Maïs', 'Tomate', 'Pomme de terre']

st.title("🌱 Classification des Cultures Agricoles")
st.write("Téléversez une image pour identifier la culture agricole.")

uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image chargée", use_column_width=True)

    # Prétraitement
    img = image.resize((128, 128))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    # Prédiction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = round(100 * np.max(prediction), 2)

    st.markdown(f"### ✅ Résultat : **{predicted_class}** ({confidence}%)")