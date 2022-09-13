"""
Recursive KNN algorith copied from `PolyChord`
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from clustering.relabel import relabel

def do_knn_clustering(knn_array):
    """
    Uses knn to determine clusters.
    """

    num_points = knn_array.shape[0]
    labels = np.arange(num_points)
    for iii in range(num_points):
        for ii in range(iii + 1, num_points):
            if labels[ii] != labels[iii]:
                if (ii in knn_array[iii]) or (iii in knn_array[ii]):
                    for i in range(num_points):
                        if labels[i] == labels[ii] or labels[i] == labels[iii]:
                            labels[i] = min([labels[ii], labels[iii]])
    return relabel(labels)


def knn(position_matrix):
    """
    Returns cluster labels of position matrix using the K-Nearest-Neighbours algorithm.

    Slight concern if two points are the same because sklearn.
    """
    npoints = position_matrix.shape[0]
    k = min(npoints, 10)
    nn = NearestNeighbors(n_neighbors=k).fit(position_matrix)
    knn_array = nn.kneighbors(position_matrix, return_distance=False)
    labels = np.arange(npoints)
    num_clusters = npoints

    labels_old = labels
    num_clusters_old = num_clusters

    for n in np.arange(2, k):
        labels = do_knn_clustering(knn_array[:, :n])
        num_clusters = max(labels) + 1

        if num_clusters <= 0:
            raise ValueError("somehow got <= 0 clusters")
        elif 1 == num_clusters:
            return labels
        elif np.all(labels == labels_old):
            break
        elif k == n - 1:
            k = min(k * 2, npoints)
            nn = NearestNeighbors(n_neighbors=k).fit(position_matrix)
            knn_array = nn.kneighbors(position_matrix, return_distance=False)

        labels_old = labels
        num_clusters_old = num_clusters

    if num_clusters > 1:
        i_cluster = 0
        while i_cluster < num_clusters:
            cluster = position_matrix[labels == i_cluster]
            labels[labels == i_cluster] = do_knn_clustering(cluster) + num_clusters
            labels = relabel(labels)
            if num_clusters - 1 == max(labels):
                i_cluster += 1
    return labels
