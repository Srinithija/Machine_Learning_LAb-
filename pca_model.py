import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def perform_pca(n_components):

    # Load Iris dataset
    iris = load_iris()
    X = iris.data

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Create DataFrame for PCA output
    columns = [f'PC{i+1}' for i in range(n_components)]
    pca_df = pd.DataFrame(X_pca, columns=columns)

    # Get explained variance
    variance = pca.explained_variance_ratio_

    # Scree Plot
    plt.figure()
    plt.bar(range(1, n_components + 1), variance)
    plt.xlabel("Principal Components")
    plt.ylabel("Explained Variance Ratio")
    plt.title("Scree Plot for PCA")
    plt.show()

    # Return dataframe and variance
    return pca_df, variance