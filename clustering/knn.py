"""
Recursive KNN algorith copied from `PolyChord`
"""

import numpy as np
from scipy.spatial import distance_matrix
from clustering.relabel import relabel


def do_knn_clustering(knn_array):
    """
    Uses knn to determine clusters. (redo)
    """
    num_points = knn_array.shape[0]
    labels = np.arange(num_points)
    for neighbours in knn_array:
        old_labels = labels[neighbours]
        new_label = np.min(old_labels)
        labels[np.isin(labels, old_labels)] = new_label
    return relabel(labels)


def compute_knn(position_matrix, k):
    x_squared = np.sum(position_matrix ** 2, axis=1).reshape(-1, 1)
    distance2_matrix = x_squared + x_squared.T - 2 * position_matrix.dot(
        position_matrix.T)

    # Get the indices of the sorted k-nearest neighbors
    knn_array = np.argsort(distance2_matrix, axis=1)[:, :k]

    return knn_array

def knn(position_matrix):
    """
    Returns cluster labels of position matrix using the
    K-Nearest-Neighbours algorithm.

    Slight concern if two points are the same because sklearn.
    """

    print("KNN clustering", flush=True)

    npoints = position_matrix.shape[0]
    k = min(npoints, 10)
    knn_array = compute_knn(position_matrix, k)
    labels = np.arange(npoints)
    num_clusters = npoints

    labels_old = labels

    for n in np.arange(2, k + 1):
        labels = do_knn_clustering(knn_array[:, : n + 1])
        num_clusters = max(labels) + 1

        if num_clusters <= 0:
            raise ValueError("somehow got <= 0 clusters")
        elif 1 == num_clusters:
            return labels
        elif np.all(labels == labels_old):
            break
        elif n == k:
            # If we need to cluster further, then expand the knn list
            k = min(k * 2, npoints)
            knn_array = compute_knn(position_matrix, k)

        labels_old = labels

    if num_clusters > 1:
        i_cluster = 0
        while i_cluster < num_clusters:
            cluster = knn_array[labels == i_cluster]
            labels[labels == i_cluster] = knn(cluster) + num_clusters
            # if didn't find any clusters, move on to the next one
            num_clusters_new = max(labels) + 1 - num_clusters
            if num_clusters_new == 1:
                i_cluster += 1
            labels = relabel(labels)
            num_clusters = max(labels) + 1
    return labels
