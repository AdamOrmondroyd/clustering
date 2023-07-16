import numpy as np
from pyclustering.cluster.xmeans import xmeans as pyclustering_xmeans
from pyclustering.cluster.gmeans import gmeans as pyclustering_gmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from clustering.relabel import relabel


def xmeans(
    position_matrix,
):
    """
    Returns cluster labels of position matrix using the X-means clustering algorithm.

    Initially looks for a maximum of 8 clusters. If it finds
    more than this, then it looks for twice as many, and so
    on.
    """

    print("X-means clustering", flush=True)

    amount_initial_centers = 1
    initial_centers = kmeans_plusplus_initializer(
        position_matrix, amount_initial_centers
    ).initialize()

    max_clusters = 8
    clusters_found = 1
    while True:
        xmeans_instance = pyclustering_xmeans(
            position_matrix, initial_centers, max_clusters, ccore=False
        )
        xmeans_instance.process()
        clusters = xmeans_instance.get_clusters()
        cluster_list = np.zeros(len(position_matrix))
        for i, cluster in enumerate(clusters):
            cluster_list[cluster] = i
        num_clusters = max(cluster_list) + 1

        if num_clusters <= max_clusters:
            return relabel(cluster_list.astype(int))
        max_clusters *= 2


def gmeans(
    position_matrix,
):
    """
    Returns cluster labels of position matrix using G-means clustering.
    """

    print("G-means clustering", flush=True)

    gmeans_instance = pyclustering_gmeans(
        position_matrix, k_init=1, ccore=False
    )
    gmeans_instance.process()
    clusters = gmeans_instance.get_clusters()
    cluster_list = np.zeros(len(position_matrix))
    for i, cluster in enumerate(clusters):
        cluster_list[cluster] = i

    return relabel(cluster_list.astype(int))
