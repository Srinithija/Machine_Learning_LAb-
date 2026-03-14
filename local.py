import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

st.title("Locally Weighted Regression (LWR) Demo")

# Load dataset
data = pd.read_csv("student_scores_200.csv")

X = data['Hours'].values
y = data['Scores'].values

# Add bias
X_mat = np.c_[np.ones(len(X)), X]

# LWR function
def lwr(query_point, X, y, tau):
    
    m = X.shape[0]
    W = np.eye(m)

    for i in range(m):
        diff = query_point - X[i]
        W[i,i] = np.exp(diff @ diff.T / (-2 * tau**2))

    theta = np.linalg.pinv(X.T @ W @ X) @ X.T @ W @ y

    return query_point @ theta


# Bandwidth slider
tau = st.slider("Bandwidth (Tau)", 0.1, 2.0, 0.5)

# Predictions
y_pred = []

for i in range(len(X)):
    y_pred.append(lwr(np.array([1,X[i]]), X_mat, y, tau))

y_pred = np.array(y_pred)

# Evaluation Metrics
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

st.subheader("Evaluation Metrics")

st.write("MSE:", mse)
st.write("RMSE:", rmse)
st.write("R2 Score:", r2)

# Plot Graph
st.subheader("Regression Graph")

fig, ax = plt.subplots()

ax.scatter(X,y,color="blue",label="Original Data")
ax.plot(X,y_pred,color="red",label="LWR Curve")

ax.set_xlabel("Study Hours")
ax.set_ylabel("Scores")
ax.legend()

st.pyplot(fig)

# Input prediction
st.subheader("Predict Score")

hours = st.number_input("Enter Study Hours",0.0,10.0)

if st.button("Predict"):

    pred = lwr(np.array([1,hours]),X_mat,y,tau)

    st.success(f"Predicted Score: {pred:.2f}")