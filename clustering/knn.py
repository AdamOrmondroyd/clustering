"""
Recursive KNN algorith copied from `PolyChord`
"""

import numpy as np
from scipy.spatial import distance_matrix
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


def compute_knn(position_matrix, k):
    npoints = position_matrix.shape[0]
    distance2_matrix = distance_matrix(position_matrix, position_matrix) ** 2
    knn_array = np.empty((npoints, k))
    for i in range(npoints):
        try:
            print(np.argsort(distance2_matrix[:, i]).shape)
            knn_array[i] = np.argsort(distance2_matrix[:, i])[:k]  # [1:k+1]
        except ValueError as e:
            print(e)
            raise e

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
    # nn = NearestNeighbors(n_neighbors=k).fit(position_matrix)
    # knn_array = nn.kneighbors(position_matrix, return_distance=False)
    knn_array = compute_knn(position_matrix, k)
    labels = np.arange(npoints)
    num_clusters = npoints

    labels_old = labels
    num_clusters_old = num_clusters

    for n in np.arange(2, k + 1):
        labels = do_knn_clustering(knn_array[:, :n+1])
        num_clusters = max(labels) + 1
        print(f"{num_clusters} clusters")

        if num_clusters <= 0:
            raise ValueError("somehow got <= 0 clusters")
        elif 1 == num_clusters:
            print("no clusters found", flush=True)
            return labels
        elif np.all(labels == labels_old):
            break
        elif n == k:
            # If we need to cluster further, then expand the knn list
            k = min(k * 2, npoints)
            print(k)
            # nn = NearestNeighbors(n_neighbors=k).fit(position_matrix)
            # knn_array = nn.kneighbors(position_matrix, return_distance=False)
            knn_array = compute_knn(position_matrix, k)

        labels_old = labels

    print(f"{num_clusters} clusters")
    if num_clusters > 1:
        i_cluster = 0
        while i_cluster < num_clusters:
            print(f"there are {num_clusters} clusters, with "
                  f"sizes{[sum(labels == i) for i in range(num_clusters)]}")
            print(f"gonna frikkin recurse {i_cluster}")
            cluster = knn_array[labels == i_cluster]
            print(cluster.shape)
            labels[labels == i_cluster] = knn(cluster) + num_clusters
            print(labels)
            # labels = relabel(labels)
            print(labels)
            # if didn't find any clusters, move on to the next one
            num_clusters_new = max(labels) + 1 - num_clusters
            if num_clusters_new == 1:
                i_cluster += 1
            labels = relabel(labels)
            num_clusters = max(labels) + 1
            print(labels, flush=True)
    print("got to bottom of knn", flush=True)
    return labels
