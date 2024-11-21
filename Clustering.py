import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from time import time
from sklearn.pipeline import make_pipeline

import matplotlib.colors as mcolors
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import json
from matplotlib.lines import Line2D

import numpy as np

class KMeansCustom:
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.labels = None

    def fit(self, X):
        np.random.seed(42)
        # KMeans++ initialization
        initial_centroids = self._kmeans_plus_plus(X)
        self.centroids = initial_centroids

        for i in range(self.max_iter):
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            self.labels = np.argmin(distances, axis=1)

            new_centroids = np.array([X[self.labels == j].mean(axis=0) for j in range(self.n_clusters)])

            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break
            self.centroids = new_centroids

    def _kmeans_plus_plus(self, X):
        # Implement KMeans++ initialization
        centroids = [X[np.random.choice(X.shape[0])]]  # Choose first centroid randomly
        for _ in range(1, self.n_clusters):
            # Compute the distance of each point to the nearest centroid
            dist_sq = np.min(np.linalg.norm(X[:, np.newaxis] - np.array(centroids), axis=2), axis=1)**2
            prob_dist = dist_sq / np.sum(dist_sq)
            new_centroid = X[np.random.choice(X.shape[0], p=prob_dist)]
            centroids.append(new_centroid)
        return np.array(centroids)



def load_json_data(filepath):
    data = []
    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()
            if line:  
                try:
                    parsed_line = json.loads(line)
                    for key, value in parsed_line.items():
                        data.append({"source": key, "content": value})
                except json.JSONDecodeError as e:
                    print(f"Skipping malformed line: {line} - {e}")
    return pd.DataFrame(data)

filepath = 'article_data/articles/part-00000'  
df = load_json_data(filepath)

if 'content' not in df or df['content'].isnull().all():
    raise ValueError("The 'content' column is missing or empty in the data.")

vectorizer = TfidfVectorizer(max_features=1000, stop_words='english') 
X_tfidf = vectorizer.fit_transform(df['content']).toarray()  

true_k = 3

kmeans_custom = KMeansCustom(n_clusters=true_k, max_iter=100)
t0 = time()
kmeans_custom.fit(X_tfidf)
train_time = time() - t0
print(f"KMeans clustering done in {train_time:.2f} seconds.")

df['cluster'] = kmeans_custom.labels

lsa = make_pipeline(TruncatedSVD(n_components=200), Normalizer(copy=False))  
X_lsa = lsa.fit_transform(X_tfidf)
reduced_centroids = lsa.transform(kmeans_custom.centroids)
original_space_centroids = lsa[0].inverse_transform(reduced_centroids)
order_centroids = original_space_centroids.argsort()[:, ::-1]

terms = vectorizer.get_feature_names_out()

def assign_shapes(df):
    unique_sources = df['source'].unique()
    shapes = ['o', 's', '^', 'D', 'P', '*', 'H']  # Add more shapes as necessary
    source_to_shape = {
        source: shapes[i % len(shapes)] for i, source in enumerate(unique_sources)
    }
    return source_to_shape

colors = plt.cm.get_cmap('tab20', true_k)  
plt.figure(figsize=(10, 6))

cluster_labels = []
source_labels = []

for cluster_num in range(true_k):
    cluster_points = df[df['cluster'] == cluster_num]
    shapes = assign_shapes(df)  

    for source in cluster_points['source'].unique():
        shape = shapes[source]
        
        plt.scatter(
            X_lsa[cluster_points[cluster_points['source'] == source].index, 0],  # First LSA component
            X_lsa[cluster_points[cluster_points['source'] == source].index, 1],  # Second LSA component
            c=[colors(cluster_num)],  
            label=f'Cluster {cluster_num}' if cluster_num not in cluster_labels else "",  # Label only for first cluster
            marker=shape,  
            alpha=0.7
        )
        
        if f'{source} ({shape})' not in source_labels:
            source_labels.append(f'{source}')

    if cluster_num == 0:
        cluster_labels.append(f'Cluster {cluster_num}')

cluster_legend = [Line2D([0], [0], marker='o', color='w', label=f'Cluster {i}', 
                         markersize=10, markerfacecolor=colors(i)) for i in range(true_k)]

source_handles = [plt.Line2D([], [], marker=shape, color='black', label=label, markersize=10, linewidth=0)
                 for shape, label in zip(shapes.values(), source_labels)]

# Combine handles and labels
handles = cluster_legend + source_handles
labels = [handle.get_label() for handle in handles]

plt.legend(handles=handles, labels=labels, loc='upper right', fontsize=9)

plt.show()
plt.title('Article Clusters by News Outlet and Cluster')
plt.xlabel('LSA Component 1')
plt.ylabel('LSA Component 2')

plt.savefig('article_clusters_kmeans.png')  # Save the plot as a PNG file
plt.close() 

print(f"Clustering visualization with shapes and colors saved to article_clusters_lsa_shapes_colors_legend_custom_kmeans.png")
