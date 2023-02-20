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
    p = (K - 1) + M * K + 1

    for ii, Rn in enumerate(cluster_sizes):
        sigma2 = 0

        for i, pos in enumerate(position_matrix):
            sigma2 += np.sum(np.square(
                np.array(pos) - np.array(cluster_centres[ii])
                ))

        sigma2 /= (R - K)

        if sigma2 <= 0:
            sigma_multiplier = -np.inf
        else:
            sigma_multiplier = M * 0.5 * np.log(sigma2)

        Li[ii] = (Rn * (np.log(Rn / R) - np.log(2 * np.pi) / 2
                        - sigma_multiplier) - (Rn - K) / 2)

    return sum(Li) - p * np.log(R) / 2
