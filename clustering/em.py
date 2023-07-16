import numpy as np
from sklearn.mixture import GaussianMixture
from clustering.relabel import relabel


def expectation_maximisation(position_matrix):
    print("EM clustering", flush=True)
    max_clusters = min(len(position_matrix), 6)
    best_score = -np.inf
    best_gmm = None
    for n_clusters in range(1, max_clusters+1):
        gmm = GaussianMixture(n_clusters).fit(position_matrix)
        score = gmm.score(position_matrix)
        print(score)
        if score > best_score:
            best_score = score
            best_gmm = gmm

    print(relabel(best_gmm.predict(position_matrix)))
    return relabel(best_gmm.predict(position_matrix))
