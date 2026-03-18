import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Title
st.title("K-Means Clustering Visualization App")

st.write("This app performs K-Means clustering on a dataset.")

# --------------------------------------------------
# Dataset (Built inside code)
# --------------------------------------------------
data = {
    'X': [2, 2, 8, 5, 7, 6, 1, 4, 9, 10],
    'Y': [10, 5, 4, 8, 5, 4, 2, 9, 3, 6]
}

df = pd.DataFrame(data)

st.subheader("Dataset")
st.write(df)

# --------------------------------------------------
# User Input
# --------------------------------------------------
k = st.slider("Select number of clusters (K)", 1, 5, 2)

# --------------------------------------------------
# Apply K-Means
# --------------------------------------------------
X = df[['X', 'Y']]

kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)

df['Cluster'] = kmeans.labels_
centroids = kmeans.cluster_centers_

# --------------------------------------------------
# Show Output
# --------------------------------------------------
st.subheader("Clustered Data")
st.write(df)

st.subheader("Centroids")
st.write(centroids)

# --------------------------------------------------
# Visualization
# --------------------------------------------------
fig, ax = plt.subplots()

ax.scatter(df['X'], df['Y'], c=df['Cluster'])
ax.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("K-Means Clustering")

st.pyplot(fig)