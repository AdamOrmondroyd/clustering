import numpy as np


def bic(position_matrix, labels, cluster_centres):

    K = len(cluster_centres)
    R = len(position_matrix)
    M = len(position_matrix[0])
    if R <= K:
        return np.inf

    Li = np.full(K, np.inf)
    cluster_sizes = np.array([sum(labels == label)
                              for label in np.unique(labels)])
    sigma2 = np.zeros(K)

    for i, pos in enumerate(position_matrix):
        sigma2[labels[i]] += np.sum(np.square(
            np.array(pos) - np.array(cluster_centres[labels[i]])
            ))

    sigma2 /= (R - K)
    sigma2[sigma2 <= 0] = 0

    p = (K - 1) + M * K + 1

    for i, Rn in enumerate(cluster_sizes):

        Li[i] = (Rn * (np.log(Rn / R) - np.log(2 * np.pi) / 2
                       - M / 2 * np.log(sigma2[i])) - (Rn - K) / 2)

    return sum(Li - p * np.log(R) / 2)
