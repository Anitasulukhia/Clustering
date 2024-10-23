import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score

cars = pd.read_csv('cars.csv')
features = ['Horsepower', 'Year', '0-60 MPH Time (seconds)']
cars = cars.dropna(subset=features)
data = cars[features].copy()

data['Horsepower'] = pd.to_numeric(data['Horsepower'], errors='coerce')
data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
data['0-60 MPH Time (seconds)'] = pd.to_numeric(data['0-60 MPH Time (seconds)'], errors='coerce')
data = data.dropna()

data = ((data - data.min()) / (data.max() - data.min())) * 9 + 1

def random_centroids(data, k):
    return data.sample(n=k, random_state=42).reset_index(drop=True)

def get_labels(data, centroids):
    labels = []
    for point in data.values:
        distances = [np.sqrt(np.sum((point - centroid) ** 2)) for centroid in centroids.values]
        closest_centroid_index = np.argmin(distances)
        labels.append(closest_centroid_index)
    return np.array(labels)

def new_centroids(data, labels, k):
    return data.groupby(labels).mean().reset_index(drop=True)

def k_means_clustering(data, k, max_iterations=100):
    centroids = random_centroids(data, k)
    old_centroids = pd.DataFrame()
    iteration = 1
    labels = None

    while iteration < max_iterations and not centroids.equals(old_centroids):
        old_centroids = centroids.copy()
        labels = get_labels(data, centroids)
        centroids = new_centroids(data, labels, k)
        plot_clusters(data, labels, centroids, iteration)
        iteration += 1

    return labels, centroids

def plot_clusters(data, labels, centroids, iteration):
    plt.title(f'K-Means Clustering - Iteration {iteration}')
    plt.scatter(data['Horsepower'], data['Year'], c=labels, cmap='viridis', marker='o')
    plt.scatter(centroids['Horsepower'], centroids['Year'], c='red', marker='x')
    plt.xlabel('Horsepower (Scaled)')
    plt.ylabel('Year (Scaled)')
    plt.show()

def random_medoids(data, k):
    return data.sample(n=k, random_state=42).reset_index(drop=True)

def calculate_cost(data_point, medoid):
    return np.sum(np.abs(data_point - medoid))

def assign_labels(data, medoids):
    distances = np.array([[calculate_cost(row, medoid) for medoid in medoids.values] for row in data.values])
    return np.argmin(distances, axis=1)

def update_medoids(data, labels, k):
    new_medoids = []
    for i in range(k):
        cluster_points = data[labels == i]
        medoid_costs = [sum([calculate_cost(point, medoid) for point in cluster_points.values]) for medoid in cluster_points.values]
        new_medoids.append(cluster_points.iloc[np.argmin(medoid_costs)].values)
    return pd.DataFrame(new_medoids, columns=data.columns)

def k_medoids_clustering(data, k, max_iterations=100):
    medoids = random_medoids(data, k)
    labels = assign_labels(data, medoids)

    for iteration in range(max_iterations):
        new_medoids = update_medoids(data, labels, k)
        if new_medoids.equals(medoids):
            break
        medoids = new_medoids
        labels = assign_labels(data, medoids)
        plot_medoids_clusters(data, labels, medoids, iteration)

    return labels, medoids

def plot_medoids_clusters(data, labels, medoids, iteration):
    plt.title(f'K-Medoids Clustering - Iteration {iteration}')
    plt.scatter(data['Horsepower'], data['Year'], c=labels, cmap='viridis', marker='o')
    plt.scatter(medoids['Horsepower'], medoids['Year'], c='red', marker='x')
    plt.xlabel('Horsepower (Scaled)')
    plt.ylabel('Year (Scaled)')
    plt.show()

k = 3

labels_kmeans, centroids_kmeans = k_means_clustering(data, k)
labels_kmedoids, medoids_kmedoids = k_medoids_clustering(data, k)

silhouette_kmeans = silhouette_score(data, labels_kmeans)
silhouette_kmedoids = silhouette_score(data, labels_kmedoids)
db_index_kmeans = davies_bouldin_score(data, labels_kmeans)
db_index_kmedoids = davies_bouldin_score(data, labels_kmedoids)

print(f'Silhouette Score - K-Means: {silhouette_kmeans}')
print(f'Silhouette Score - K-Medoids: {silhouette_kmedoids}')
print(f'Davies-Bouldin Index - K-Means: {db_index_kmeans}')
print(f'Davies-Bouldin Index - K-Medoids: {db_index_kmedoids}')
