import numpy as np
from pyclustering.cluster.xmeans import xmeans as pyclustering_xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from clustering.relabel import relabel
from clustering.bic import bic
import sklearn.cluster as skc


def xmeans(
    position_matrix,
):
    """
    Cluster labels of position matrix using X-means clustering.

    Initially looks for a maximum of 8 clusters. If it finds
    more than this, then it looks for twice as many, and so
    on.
    """

    print("X-means clustering", flush=True)
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
        print(clusters)
        cluster_list = np.zeros(len(position_matrix), dtype=int)
        for i, cluster in enumerate(clusters):
            cluster_list[cluster] = i
        num_clusters = max(cluster_list) + 1

        if num_clusters <= max_clusters:
            print(f"found {num_clusters}", flush=True)
            print(f"old BIC: {old_bic(position_matrix, xmeans_instance.get_centers(), cluster_list)}")
            print(f"BIC: {bic(position_matrix, cluster_list, xmeans_instance.get_centers())}")
            # print(f"pyclustering? :{xmeans_instance.__bayesian_information_criterion(xmeans_instance.get_clusters(), xmeans_instance.get_centers())}")
            return relabel(cluster_list.astype(int))
        max_clusters = min(max_clusters * 2, position_matrix.shape[0])


def sklearn_xmeans(position_matrix):
    print('hello')
    labelss = []
    bics = []
    max_clusters = 6
    for k in range(1, max_clusters + 1):
        print(f"k={k}")
        labels, bic = kmeans(position_matrix, k)
        labelss.append(labels)
        bics.append(bic)
        print(f"finished k={k}")

    print(f"found {np.max(np.argmin(bics))+1} clusters", flush=True)
    print(bics)

    print(relabel(labelss[np.argmin(bics)]))
    return relabel(labelss[np.argmin(bics)])


def kmeans(position_matrix, k):
    print("gonna frikkin init")
    km = skc.KMeans(n_clusters=k, algorithm='lloyd',
                    init='k-means++', n_init=1)
    print("about to fit")
    print(position_matrix)
    km.fit(position_matrix)
    print("done fit")
    print(f"labels {km.labels_}")
    # return km.labels_, bic(position_matrix, km.cluster_centers_, km.labels_)
    print(f"my BIC: {bic(position_matrix, km.labels_, km.cluster_centers_)}")
    # print(f"copied BIC: {pyclustering_bic([[pos for i, pos in enumerate(position_matrix) if km.labels_[i] == label] for label in np.unique(km.labels_)], km.cluster_centers_)}")
    return km.labels_, bic(position_matrix, km.labels_, km.cluster_centers_)
    # return km.labels_, pyclustering_bic([
        # [pos for i, pos in enumerate(position_matrix) if km.labels_[i] == label]
        # for label in np.unique(km.labels_)], km.cluster_centers_)


def old_bic(position_matrix, cluster_centres, labels):
    cluster_centres = np.array([cluster_centres[i] for i in labels])
    print("cluster centres")
    print(cluster_centres)
    n = position_matrix.shape[0]
    # This is not the kmeans k. This is the number of parameters in the model
    k = (max(labels) + 1) * position_matrix.shape[1]
    var = 1 / n * np.sum(np.square(position_matrix-cluster_centres))
    return n * np.log(var) + k * np.log(n)


def pyclustering_bic(clusters, centers):
    """!
    @brief Calculates splitting criterion for input clusters using bayesian information criterion.

    @param[in] clusters (list): Clusters for which splitting criterion should be calculated.
    @param[in] centers (list): Centers of the clusters.

    @return (double) Splitting criterion in line with bayesian information criterion.
            High value of splitting criterion means that current structure is much better.

    @see __minimum_noiseless_description_length(clusters, centers)

    """

    scores = [float('inf')] * len(clusters)     # splitting criterion
    dimension = len(centers[0])
    print(clusters)
    print(centers)

    # estimation of the noise variance in the data set
    sigma_sqrt = 0.0
    K = len(clusters)
    N = 0.0

    for index_cluster in range(0, len(clusters), 1):
        print("Je suis ici")
        print(index_cluster)
        for index_object in clusters[index_cluster]:
            print(f"index pbject: {index_object}")
            # sigma_sqrt += np.sqrt(np.sum(np.square(index_object - centers[index_cluster])))
            print(centers[index_cluster])
            # sigma_sqrt += np.linalg.norm(index_object - centers[index_cluster])

        N += len(clusters[index_cluster])

    if N - K > 0:
        sigma_sqrt /= (N - K)
        p = (K - 1) + dimension * K + 1

        # in case of the same points, sigma_sqrt can be zero (issue: #407)
        sigma_multiplier = 0.0
        if sigma_sqrt <= 0.0:
            sigma_multiplier = float('-inf')
        else:
            sigma_multiplier = dimension * 0.5 * np.log(sigma_sqrt)

        # splitting criterion
        for index_cluster in range(0, len(clusters), 1):
            n = len(clusters[index_cluster])

            L = n * np.log(n) - n * np.log(N) - n * 0.5 * np.log(2.0 * np.pi) - n * sigma_multiplier - (n - K) * 0.5

            # BIC calculation
            scores[index_cluster] = L - p * 0.5 * np.log(N)

    return sum(scores)
