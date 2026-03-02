import streamlit as st
import pandas as pd
import os
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# ----------------------------
# Load Dataset
# ----------------------------
path = kagglehub.dataset_download("dhrubangtalukdar/cybersecurity-threat-detection-dataset")
file_path = os.path.join(path, os.listdir(path)[0])

data = pd.read_csv(file_path)

# Clean column names
data.columns = data.columns.str.strip()

# Select features
selected_features = [
    "src_port",
    "dst_port",
    "protocol",
    "bytes_sent",
    "bytes_received",
    "is_internal_traffic"
]

X = data[selected_features]
y = data["label"]

# Encode categorical
X = pd.get_dummies(X, columns=["protocol", "is_internal_traffic"])

# Train model
model = DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42)
model.fit(X, y)

# ----------------------------
# UI
# ----------------------------
# ----------------------------
# UI
# ----------------------------

st.title("Cybersecurity Threat Detection using Decision Tree")

st.markdown("""
This system predicts whether a given network traffic instance 
is **Benign** or **Malicious** using a trained Decision Tree model.
""")

st.subheader("Enter Network Traffic Details")

# Input fields
src_port = st.number_input("Source Port", min_value=0, max_value=65535, value=445)
dst_port = st.number_input("Destination Port", min_value=0, max_value=65535, value=22)
bytes_sent = st.number_input("Bytes Sent", min_value=0, value=5000)
bytes_received = st.number_input("Bytes Received", min_value=0, value=2000)

protocol = st.selectbox("Protocol", ["TCP", "UDP"])
internal = st.selectbox("Is Internal Traffic?", ["TRUE", "FALSE"])

st.write("---")

if st.button("Predict"):

    # Create input dataframe
    input_data = pd.DataFrame({
        "src_port": [src_port],
        "dst_port": [dst_port],
        "bytes_sent": [bytes_sent],
        "bytes_received": [bytes_received],
        "protocol": [protocol],
        "is_internal_traffic": [internal]
    })

    # Encode
    input_data = pd.get_dummies(input_data)

    # Align columns
    input_data = input_data.reindex(columns=X.columns, fill_value=0)

    # Prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Result: Malicious Traffic Detected")
        st.write(f"Probability of Attack: {probability*100:.2f}%")
        st.write("Risk Level: High")
    else:
        st.success("Result: Benign Traffic")
        st.write(f"Probability of Attack: {probability*100:.2f}%")
        st.write("Risk Level: Low")