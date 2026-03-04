import streamlit as st
import pandas as pd
import numpy as np
import kagglehub
import os

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from streamlit_drawable_canvas import st_canvas
from PIL import Image
import cv2

st.title("Character Recognition using MLP (Draw Alphabet)")

# -------------------------------
# Download Dataset
# -------------------------------
@st.cache_data
def load_dataset():

    path = kagglehub.dataset_download(
        "sachinpatel21/az-handwritten-alphabets-in-csv-format"
    )

    files = os.listdir(path)

    for file in files:
        if file.endswith(".csv"):
            csv_path = os.path.join(path, file)

    df = pd.read_csv(csv_path)

    # Reduce dataset size for faster training
    df = df.sample(20000, random_state=42)

    return df


st.write("Loading dataset...")
df = load_dataset()
st.success("Dataset Loaded")

# -------------------------------
# Preprocessing
# -------------------------------
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

X = X / 255.0

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Train MLP Model
# -------------------------------
@st.cache_resource
def train_model(X_train, y_train):

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=20,
        learning_rate_init=0.001,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


with st.spinner("Training Model..."):
    mlp = train_model(X_train, y_train)

st.success("Model Trained")

# -------------------------------
# Accuracy
# -------------------------------
y_pred = mlp.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Accuracy")
st.write(f"{round(accuracy*100,2)} %")

# -------------------------------
# Drawing Canvas
# -------------------------------
st.subheader("Draw a Character")

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# -------------------------------
# Prediction from Drawing
# -------------------------------
if canvas_result.image_data is not None:

    img = canvas_result.image_data

    img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img, (28,28))

    img = img.reshape(1,784)

    img = img / 255.0

    prediction = mlp.predict(img)

    letter = chr(prediction[0] + 65)

    st.subheader("Predicted Character")

    st.success(letter)