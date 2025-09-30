import tensorflow as tf
from tensorflow.keras.models import load_model
from preprocess_data import IMG_SIZE
from PIL import Image
import numpy as np
import pickle

MODEL_PATH = "models/rnn_invoice_model.h5"
ENCODER_PATH = "models/label_encoder.pkl"

model = load_model(MODEL_PATH)
with open(ENCODER_PATH, "rb") as f:
    le = pickle.load(f)

def predict_image(image_path):
    img = Image.open(image_path).convert("L").resize((IMG_SIZE, IMG_SIZE))
    x = np.array(img)/255.0
    x = x.reshape((1, IMG_SIZE, IMG_SIZE))
    pred = model.predict(x)
    label = le.inverse_transform([pred.argmax()])[0]
    return label

if __name__ == "__main__":
    path = input("Enter path to image: ")
    print("Predicted category:", predict_image(path))
