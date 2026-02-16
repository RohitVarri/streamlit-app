# Malaria Detection Web App

This is a Streamlit-based web application for malaria detection using a deep learning model. Users can upload cell images, and the app will predict whether the cell is infected with malaria.

## How to Use

1. Place your Keras model file (e.g., `Enhanced_EfficientNetB3_Final2.keras`) in this folder.
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the app:
   ```sh
   streamlit run malaria_app.py
   ```

## Requirements
- Python 3.8+
- Streamlit
- TensorFlow
- Pillow
- NumPy

---

**Note:** Replace the model file with your own `.keras` file if needed.
