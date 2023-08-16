"""
Clustering algorithm copied from `dynesty`

Note that scipy seems to number clusters from 1.
"""
import numpy as np
from scipy import spatial, cluster
from relabel import relabel

# note: scipy seems to number clusters from 1


def dynesty(points, depth=0):
    """Compute covariance from re-centered clusters."""

    print(f"Dynesty clustering {depth=}", flush=True)
    # compute covariance matrix. Have noticed occasional singularity issues,
    # hence the exception block
    try:
        inv = np.linalg.inv(np.cov(points.T)).T
    except np.linalg.LinAlgError as e:
        print(e.message, flush=True)
        print("assuming single cluster and moving on", flush=True)
        return np.zeros(len(points))

    # Compute pairwise distances.
    distances = spatial.distance.pdist(points,
                                       metric='mahalanobis',
                                       VI=inv,
                                       )

    # Identify conglomerates of points by constructing a linkage matrix.
    linkages = cluster.hierarchy.single(distances)

    # Cut when linkage between clusters exceed the radius.
    labels = cluster.hierarchy.fcluster(linkages,
                                        1.0,
                                        criterion='distance')
    labels = relabel(np.array(labels) - 1)
    if max(labels) > 0:
        labels = combine_labels(labels, *[dynesty_cluster(
            points[labels == label], depth=depth+1)
            for label in np.unique(labels)])
    return labels


def combine_labels(initial_splitting, *labelss):
    """
    Combine labels from recursive clustering into a single list
    """
    sizes = [max(labels) for labels in labelss]

    for i in np.arange(max(initial_splitting-1), -1, -1):
        initial_splitting[initial_splitting == i] = labelss[i] + sum(
            sizes[:i-1])
    print(f"{initial_splitting=}", flush=True)
    return initial_splitting
