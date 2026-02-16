import streamlit as st
import numpy as np
from tensorflow import keras
from PIL import Image
import os
import io
import base64
import requests


MODEL_PATH = 'Enhanced_EfficientNetB31212.keras'  # Updated model file name to match actual file
DROPBOX_MODEL_URL = 'https://www.dropbox.com/scl/fi/66b6wbqcbndzc5p0f25tu/Enhanced_EfficientNetB31212.keras?rlkey=r3vty8wadpz85vstazbs5fizp&st=257n7o7g&dl=1'

def download_model_if_needed():
    if not os.path.exists(MODEL_PATH):
        st.info('Model file not found locally. Downloading from Dropbox...')
        try:
            import requests
            with requests.get(DROPBOX_MODEL_URL, stream=True) as r:
                r.raise_for_status()
                with open(MODEL_PATH, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            st.success('Model downloaded successfully!')
        except Exception as e:
            st.error(f'Failed to download model: {e}')
            return False
    return True


@st.cache_resource
def load_model():
    if not download_model_if_needed():
        return None
    return keras.models.load_model(MODEL_PATH)

model = load_model()

st.title('Malaria Detection App')
st.write('Upload a cell image to predict malaria presence using the enhanced EfficientNetB3 model.')

uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    img = image.resize((224, 224))  # Update size if your model expects a different input
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # If you have a custom preprocess_image function, apply it here
    # from your_colab_notebook import preprocess_image
    # img_array = preprocess_image(img_array[0])
    # img_array = np.expand_dims(img_array, axis=0)
    if model:
        prediction = model.predict(img_array)
        score = float(prediction[0][0])
        if score > 0.5:
            st.write(f'Prediction: **Uninfected** (Score: {score:.4f})')
        else:
            st.write(f'Prediction: **Parasitized** (Score: {score:.4f})')
        st.progress(score if score > 0.5 else 1 - score)
        st.caption('Score is the model\'s output probability for the "Uninfected" class.')

    else:
        st.warning('Model is not loaded. Please upload the .keras file to the folder.')
