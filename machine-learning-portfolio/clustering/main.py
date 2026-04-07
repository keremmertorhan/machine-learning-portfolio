import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

# Load data from file
def load_data(file_path, has_label=False):
    if has_label:
        data = pd.read_csv(file_path, sep='\t', header=None)
        data.columns = ['X', 'Y', 'Label']
    else:
        data = pd.read_csv(file_path, sep='\s+', header=None)
        data.columns = ['X', 'Y']
    return data

# File paths for datasets
files = {
    "s1": "s1.txt",
    "s2": "s2.txt",
    "s3": "s3.txt",
    "s4": "s4.txt",
    "spiral": "spiral.txt"
}

# Load all datasets into a dictionary
datasets = {name: load_data(path, has_label=(name == "spiral")) for name, path in files.items()}

# Function to apply KMeans and plot results
def plot_clusters(data, name, k=4, init_method='k-means++', compare_labels=False):
    model = KMeans(n_clusters=k, init=init_method, n_init=10, random_state=42)
    coords = data[['X', 'Y']]
    model.fit(coords)
    labels = model.labels_
    centroids = model.cluster_centers_

    # Plot clustering result
    plt.figure(figsize=(6, 5))
    plt.scatter(data['X'], data['Y'], c=labels, cmap='tab10', s=10, label='Clustered Points')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=100, label='Centroids')
    plt.title(f"{name} - KMeans Clustering (k={k})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()

    # If original labels are available, compare
    if compare_labels and 'Label' in data.columns:
        plt.figure(figsize=(6, 5))
        plt.scatter(data['X'], data['Y'], c=data['Label'], cmap='tab10', s=10)
        plt.title(f"{name} - Original Labels")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.show()

# Apply clustering and plot for each dataset
plot_clusters(datasets['s1'], "s1.txt", k=4)
plot_clusters(datasets['s2'], "s2.txt", k=4)
plot_clusters(datasets['s3'], "s3.txt", k=4)
plot_clusters(datasets['s4'], "s4.txt", k=4)
plot_clusters(datasets['spiral'], "spiral.txt", k=3, compare_labels=True)
