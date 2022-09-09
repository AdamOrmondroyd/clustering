import numpy as np
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer


def custom_cluster(
    position_matrix,
):
    """
    X-means clustering algorithm.

    Initially looks for a maximum of 20 clusters. If it finds
    more than this, then it looks for twice as many, and so
    on.
    """
    print("started clustering", flush=True)
    amount_initial_centers = 1
    initial_centers = kmeans_plusplus_initializer(
        position_matrix, amount_initial_centers
    ).initialize()

    max_clusters = 8
    clusters_found = 1
    while True:
        xmeans_instance = xmeans(position_matrix, initial_centers, max_clusters, ccore=False)
        xmeans_instance.process()
        clusters = xmeans_instance.get_clusters()
        cluster_list = np.zeros(len(position_matrix))
        for i, cluster in enumerate(clusters):
            cluster_list[cluster] = i
        num_clusters = max(cluster_list) + 1
       
        if num_clusters <=  max_clusters:
            print("finished clustering", flush=True)
            return cluster_list
        max_clusters *= 2
