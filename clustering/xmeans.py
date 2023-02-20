import numpy as np
from pyclustering.cluster.xmeans import xmeans as pyclustering_xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from clustering.relabel import relabel
from clustering.bic import bic
import sklearn.cluster as skc


def xmeans(position_matrix):
    """
    Cluster labels of position matrix using X-means clustering.

    Initially looks for a maximum of 8 clusters. If it finds
    more than this, then it looks for twice as many, and so
    on.
    """

    print("PyClustering X-means clustering", flush=True)
    amount_initial_centers = 4
    initial_centers = kmeans_plusplus_initializer(
        position_matrix, amount_initial_centers
    ).initialize()

    max_clusters = 8
    while True:
        print(f"max_clusters: {max_clusters}")
        xmeans_instance = pyclustering_xmeans(
            position_matrix, initial_centers, max_clusters, ccore=False
        )
        xmeans_instance.process()
        clusters = xmeans_instance.get_clusters()
        cluster_list = np.zeros(len(position_matrix), dtype=int)
        for i, cluster in enumerate(clusters):
            cluster_list[cluster] = i
        num_clusters = max(cluster_list) + 1

        if num_clusters <= max_clusters:
            print(f"found {num_clusters} clusters", flush=True)
            return relabel(cluster_list.astype(int))
        max_clusters = min(max_clusters * 2, position_matrix.shape[0])


def sklearn_xmeans(position_matrix):
    print("sklearn X-means clustering", flush=True)
    labelss = []
    bics = []
    max_clusters = 6
    for k in range(1, max_clusters + 1):
        labels, bic = kmeans(position_matrix, k)
        labelss.append(labels)
        bics.append(bic)

    print(f"found {np.max(np.argmax(bics))+1} clusters", flush=True)
    return relabel(labelss[np.argmax(bics)])


def kmeans(position_matrix, k):
    km = skc.KMeans(n_clusters=k, algorithm='lloyd',
                    init='k-means++', n_init=1)
    km.fit(position_matrix)
    return km.labels_, bic(position_matrix, km.labels_, km.cluster_centers_)
