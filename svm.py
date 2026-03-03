import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import kagglehub
from kagglehub import KaggleDatasetAdapter


st.title("Spam Mail Detection using Support Vector Machine (SVM)")

# -----------------------------
# Load Dataset
# -----------------------------
file_path = "spam.csv"

df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "uciml/sms-spam-collection-dataset",
    file_path,
    pandas_kwargs={"encoding": "latin-1"}
)

# Select needed columns
df = df[['v1','v2']]
df.columns = ['label','message']

# Convert labels
df['label'] = df['label'].map({'ham':0,'spam':1})

st.subheader("Dataset Preview")
st.dataframe(df.head())


# -----------------------------
# Feature Extraction
# -----------------------------
X = df['message']
y = df['label']

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(X)


# -----------------------------
# Train Test Split
# -----------------------------
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)


# -----------------------------
# Train SVM Model
# -----------------------------
model = SVC(kernel='linear')

model.fit(X_train,y_train)


# -----------------------------
# Model Evaluation
# -----------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)

st.subheader("Model Accuracy")

st.success(f"Accuracy: {accuracy:.2f}")


# -----------------------------
# Confusion Matrix
# -----------------------------
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test,y_pred)

fig, ax = plt.subplots()

ax.imshow(cm)

ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")

ax.set_xticks([0,1])
ax.set_yticks([0,1])

ax.set_xticklabels(["Ham","Spam"])
ax.set_yticklabels(["Ham","Spam"])

for i in range(2):
    for j in range(2):
        ax.text(j,i,cm[i,j],ha="center",va="center",color="black")

st.pyplot(fig)


# -----------------------------
# Classification Report
# -----------------------------
st.subheader("Classification Report")

report = classification_report(y_test,y_pred)

st.text(report)


# -----------------------------
# SVM Decision Boundary Visualization
# -----------------------------
st.subheader("SVM Decision Boundary Illustration")

from sklearn.datasets import make_blobs

X_vis, y_vis = make_blobs(n_samples=50, centers=2, random_state=6)

model_vis = SVC(kernel='linear')
model_vis.fit(X_vis, y_vis)

plt.figure()

plt.scatter(X_vis[:,0],X_vis[:,1],c=y_vis,cmap='coolwarm')

ax = plt.gca()

xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0],xlim[1],30)
yy = np.linspace(ylim[0],ylim[1],30)

YY,XX = np.meshgrid(yy,xx)

xy = np.vstack([XX.ravel(),YY.ravel()]).T

Z = model_vis.decision_function(xy).reshape(XX.shape)

ax.contour(
    XX,YY,Z,
    colors='green',
    levels=[-1,0,1],
    linestyles=['--','-','--']
)

st.pyplot(plt)


# -----------------------------
# User Prediction UI
# -----------------------------
st.subheader("Test Your Message")

message = st.text_area("Enter SMS or Email")

if st.button("Predict"):

    if message.strip() == "":
        st.warning("Please enter a message")

    else:
        data = vectorizer.transform([message])

        prediction = model.predict(data)

        if prediction[0] == 1:
            st.error("🚨 This message is SPAM")

        else:
            st.success("✅ This message is NOT Spam")