from pprint import pprint

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

import myplots as myplt
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from itertools import cycle, islice
import scipy.io as io
from scipy.cluster.hierarchy import dendrogram, linkage  #

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

# ----------------------------------------------------------------------
"""
Part 2
Comparison of Clustering Evaluation Metrics: 
In this task you will explore different methods to find a good value for k
"""

# Fill this function with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_kmeans(dataset, n_clusters):

    data, _ = dataset

    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data)

    kmeans = KMeans(n_clusters=n_clusters, init='random', random_state=42)
    kmeans.fit(data_standardized)

    sse = 0
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    for i in range(len(data_standardized)):
        # Squared distance between data point and its cluster center
        sse += np.square(np.linalg.norm(data_standardized[i] - cluster_centers[labels[i]]))

    return sse



def compute():
    # ---------------------
    answers = {}

    """
    A.	Call the make_blobs function with following parameters :(center_box=(-20,20), n_samples=20, centers=5, random_state=12).
    """

    data, labels = make_blobs(center_box=(-20, 20), n_samples=20, centers=5, random_state=12)

    # dct: return value from the make_blobs function in sklearn, expressed as a list of three numpy arrays
    dct = answers["2A: blob"] = [data, labels, np.zeros(20)]

    """
    B. Modify the fit_kmeans function to return the SSE (see Equations 8.1 and 8.2 in the book).
    """

    # dct value: the `fit_kmeans` function
    dct = answers["2B: fit_kmeans"] = fit_kmeans

    """
    C.	Plot the SSE as a function of k for k=1,2,….,8, and choose the optimal k based on the elbow method.
    """

    # Calculate SSE for k=1 to k=8
    sse_values = []
    for k in range(1, 9):
        sse = fit_kmeans([data, labels], k)
        sse_values.append([k, sse])

    plt.figure(figsize=(10, 6))
    plt.plot([k for k, sse in sse_values], [sse for k, sse in sse_values], marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Sum of Squared Errors (SSE)')
    plt.title('SSE vs. Number of Clusters')
    plt.grid(True)
    plt.show()

    # dct value: a list of tuples, e.g., [[0, 100.], [1, 200.]]
    # Each tuple is a (k, SSE) pair
    answers["2C: SSE plot"] = [[k, sse] for k, sse in sse_values]

    #dct = answers["2C: SSE plot"] = [[0.0, 100.0]]

    """
    D.	Repeat part 2.C for inertia (note this is an attribute in the kmeans estimator called _inertia). Do the optimal k’s agree?
    """

    # dct value has the same structure as in 2C
    dct = answers["2D: inertia plot"] = [[k, sse] for k, sse in sse_values]

    # dct value should be a string, e.g., "yes" or "no"
    dct = answers["2D: do ks agree?"] = "yes"

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part2.pkl", "wb") as f:
        pickle.dump(answers, f)
