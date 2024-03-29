import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
import scipy.io as io
from scipy.cluster.hierarchy import dendrogram, linkage  #
from scipy.cluster.hierarchy import linkage, fcluster


# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

"""
Part 4.	
Evaluation of Hierarchical Clustering over Diverse Datasets:
In this task, you will explore hierarchical clustering over different datasets. You will also evaluate different ways to merge clusters and good ways to find the cut-off point for breaking the dendrogram.
"""

# Fill these two functions with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_hierarchical_cluster(dataset, linkage_type, n_clusters):

    data, _ = dataset

    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data)

    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_type)

    labels = model.fit_predict(data_standardized)

    return labels

def fit_modified(dataset, linkage_method='ward'):

    data, _ = dataset

    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data)

    # Calculate the linkage matrix
    Z = linkage(data_standardized, method=linkage_method)

    # Find the maximum rate of change in the distance between successive merges
    # The rate of change between successive distances
    distances = Z[:, 2]
    rate_of_change = np.diff(distances)

    # Finding the index of the maximum rate of change
    index_of_max_diff = np.argmax(rate_of_change)
    
    # Estimating cut-off distance as the distance where this change occurs
    cut_off_distance = distances[index_of_max_diff]
    
    # Classifying clusters based on the cut-off distance
    labels = fcluster(Z, cut_off_distance, criterion='distance')

    return labels, cut_off_distance


def compute():
    answers = {}

    """
    A.	Repeat parts 1.A and 1.B with hierarchical clustering. That is, write a function called fit_hierarchical_cluster (or something similar) that takes the dataset, the linkage type and the number of clusters, that trains an AgglomerativeClustering sklearn estimator and returns the label predictions. Apply the same standardization as in part 1.B. Use the default distance metric (euclidean) and the default linkage (ward).
    """

    random_state = 42

    n_samples = 100
    noisy_circles = make_circles(n_samples=n_samples, factor=.5, noise=.05, random_state=random_state)
    noisy_moons = make_moons(n_samples=n_samples, noise=.05, random_state=random_state)
    blobs = make_blobs(n_samples=n_samples, random_state=random_state)
    varied = make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)

    # Anisotropicly distributed data
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(blobs[0], transformation)
    aniso = (X_aniso, blobs[1])

    datasets = {
        "nc": noisy_circles,
        "nm": noisy_moons,
        "bvv": varied,
        "add": aniso,
        "b": blobs
        }
    
    # Scaling all datasets for consistency
    for key in datasets.keys():
        datasets[key] = (StandardScaler().fit_transform(datasets[key][0]), datasets[key][1])

    # Dictionary of 5 datasets. e.g., dct["nc"] = [data, labels]
    # keys: 'nc', 'nm', 'bvv', 'add', 'b' (abbreviated datasets)
    dct = answers["4A: datasets"] = datasets

    # dct value:  the `fit_hierarchical_cluster` function
    dct = answers["4A: fit_hierarchical_cluster"] = fit_hierarchical_cluster

    """
    B.	Apply your function from 4.A and make a plot similar to 1.C with the four linkage types (single, complete, ward, centroid: rows in the figure), and use 2 clusters for all runs. Compare the results to problem 1, specifically, are there any datasets that are now correctly clustered that k-means could not handle?

    Create a pdf of the plots and return in your report. 
    """

    linkage_types = ['single', 'complete', 'ward']

    # Plot settings
    plt.figure(figsize=(15, 10))

    for i, (dataset_label, dataset) in enumerate(datasets.items()):
        for j, linkage in enumerate(linkage_types):
            plt_index = len(linkage_types) * i + j + 1
            plt.subplot(len(datasets), len(linkage_types), plt_index)

            
            labels = fit_hierarchical_cluster(dataset, linkage, n_clusters=2)

            # Scatter plot of the data colored by cluster label
            plt.scatter(dataset[0][:, 0], dataset[0][:, 1], c=labels)
            if j == 0:
                plt.ylabel(dataset_label)
            if i == 0:
                plt.title(linkage)
    
    plt.tight_layout()
    plt.show()

    # dct value: list of dataset abbreviations (see 1.C)
    dct = answers["4B: cluster successes"] = ["nc", "nm"]

    """
    C.	There are essentially two main ways to find the cut-off point for breaking the diagram: specifying the number of clusters and specifying a maximum distance. The latter is challenging to optimize for without knowing and/or directly visualizing the dendrogram, however, sometimes simple heuristics can work well. The main idea is that since the merging of big clusters usually happens when distances increase, we can assume that a large distance change between clusters means that they should stay distinct. Modify the function from part 1.A to calculate a cut-off distance before classification. Specifically, estimate the cut-off distance as the maximum rate of change of the distance between successive cluster merges (you can use the scipy.hierarchy.linkage function to calculate the linkage matrix with distances). Apply this technique to all the datasets and make a plot similar to part 4.B.
    
    Create a pdf of the plots and return in your report. 
    """

    plt.figure(figsize=(15, 10))
    for i, (dataset_label, dataset) in enumerate(datasets.items()):
        for j, linkage in enumerate(linkage_types):
            plt_index = len(linkage_types) * i + j + 1
            plt.subplot(len(datasets), len(linkage_types), plt_index)

            # Use the modified function to calculate labels
            labels, cut_off_distance = fit_modified(dataset, linkage_method=linkage)

            # Scatter plot of the data colored by cluster label
            plt.scatter(dataset[0][:, 0], dataset[0][:, 1], c=labels, cmap=plt.cm.Spectral)

            if j == 0:
                plt.ylabel(dataset_label)
            if i == 0:
                plt.title(linkage)

    plt.tight_layout()
    plt.show()

    # dct is the function described above in 4.C
    dct = answers["4C: modified function"] = fit_modified

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part4.pkl", "wb") as f:
        pickle.dump(answers, f)
