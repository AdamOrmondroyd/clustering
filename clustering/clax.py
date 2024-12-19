from functools import partial
import numpy as np
import jax
from jax import numpy as jnp
from matplotlib import pyplot as plt
from timeit import timeit
from clustering.relabel import relabel


@jax.jit
def distance_2(x, centres):
    """
    Calculate the probability of each x being chosen as a new centre.

    Probability is proportional to the distance squared from the nearest centre.

    """
    # subtract every x(num_x, ndims) from every centre(num_centres, ndims)
    distances = x[:, None] - centres[None, :]

    distance_2s = jnp.sum(distances ** 2, axis=2)
    # get the minimum distance2 for each x
    return jnp.min(distance_2s, axis=1)


@jax.jit
def p_distance_2(x, centres):
    # return the normalised distance2
    distance_2s = distance_2(x, centres)
    return distance_2s / jnp.sum(distance_2s, axis=0)


def kmeans_plusplus_initialiser(key, x, k):

    centres = jnp.zeros((k, x.shape[1]))

    key, subkey = jax.random.split(key)
    centres = centres.at[0].set(x[jax.random.choice(subkey, x.shape[0])])

    def update_centres(i, carry):
        key, centres, x = carry
        key, subkey = jax.random.split(key)
        p = p_distance_2(x, centres)
        new_centre = x[jax.random.choice(subkey, x.shape[0], p=p)]
        centres = centres.at[i].set(new_centre)

        return key, centres, x
    key, centres, x = jax.lax.fori_loop(1, k, update_centres, (key, centres, x))

    return centres


@jax.jit
def assign(x, centres):
    # assign each x to the nearest centre
    distances = x[:, None] - centres[None, :]
    distance_2s = jnp.sum(distances ** 2, axis=2)
    assignments = jnp.argmin(distance_2s, axis=1)
    return assignments


def update_centres(x, assignments):
    k = assignments.max() + 1
    cluster_counts = jnp.bincount(assignments, length=k)
    cluster_sums = jax.ops.segment_sum(x, assignments, k)
    centroids = cluster_sums / cluster_counts[:, None]

    centroids = jnp.where(jnp.isnan(centroids), 0, centroids)
    return centroids


def kmeans(key, x, k, kmeans_plusplus_initialiser, assign, update_centres, max_iter=100):
    centres = kmeans_plusplus_initialiser(key, x, k)
    for i in range(max_iter):
        assignments = assign(x, centres)
        # update the centres
        centres = update_centres(x, assignments)
        # assert jnp.allclose(centroids, centres, atol=1e-3)
    return centres, assignments


@jax.jit
def bic(x, labels, centres):
    k = len(centres)
    r, m = x.shape
    if r <= k:
        return jnp.inf

    rn = jnp.bincount(labels, length=k)
    # if jnp.any(rn == 0):
        # return jnp.inf

    # compute single sigma2
    # sigma2 = jnp.sum(distance_2(x, centres)) / (r - k)
    # k-1 class probabilities, m*k means, 1 variance
    # p = (k - 1) + m * k + 1
    # compute individual sigma2 for each cluster
    sigma2 = jax.ops.segment_sum(distance_2(x, centres), labels, k) / (rn - k)
    # k-1 class probabilities, m*k means, k variances
    p = (k - 1) + m * k + k

    logl = (
            rn * (jnp.log(rn / r) - m / 2 * (jnp.log(2 * jnp.pi * sigma2)))
            - (rn - k) / 2
    )

    return -2 * jnp.sum(logl), p * jnp.log(r)
    # return -2 * jnp.sum(logl) + p * jnp.log(r)


def xmeans(key, x, kmeans, ic, max_k=8):
    i = 1
    key, subkey = jax.random.split(key)
    centres_i, assignments_i = kmeans(subkey, x, i)
    ic_i = ic(x, assignments_i, centres_i)

    for ii in range(2, max_k):

        key, subkey = jax.random.split(key)
        centres_ii, assignments_ii = kmeans(subkey, x, ii)
        ic_j = ic(x, assignments_ii, centres_ii)

        if ic_i >= ic_j:
            ic_i = ic_j
            centres_i = centres_ii
            assignments_i = assignments_ii

    # if ii == max_k:
        # return xmeans(key, x, kmeans, ic, max_k * 2)
    # TODO: recursive call on subclusters

    return centres_i, assignments_i


_pc_kmeans = partial(
    kmeans,
    kmeans_plusplus_initialiser=kmeans_plusplus_initialiser,
    assign=assign,
    update_centres=update_centres,
)

_pc_key = jax.random.PRNGKey(0)
_pc_xmeans = partial(
    xmeans,
    key=_pc_key,
    kmeans=_pc_kmeans,
    ic=bic,
)


def pc_xmeans(x):
    print("JAX-means clustering", flush=True)
    assignments = relabel(np.array(_pc_xmeans(x=jnp.array(x))[1]))
    print(assignments, flush=True)
    return assignments


if __name__ == "__main__":
    _pc_kmeans = partial(
        kmeans,
        kmeans_plusplus_initialiser=kmeans_plusplus_initialiser,
        assign=assign,
        update_centres=update_centres,
    )
    key = jax.random.PRNGKey(1)
    x = jnp.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    k = 3
    key, subkey0 = jax.random.split(key)
    key, subkey1 = jax.random.split(key)
    x = jnp.vstack([
        jax.random.normal(subkey0, (100, 2)),
        jax.random.normal(subkey1, (100, 2))*2 + 10
    ])

    bics = []
    gof = []
    pen = []

    for k in range(1, 11):
        key, subkey = jax.random.split(key)
        centres, assignments = _pc_kmeans(subkey, x, k)

        # bics.append(bic(x, assignments, centres))
        _ = bic(x, assignments, centres)
        gof.append(_[0])
        pen.append(_[1])
        bics.append(gof[-1] + pen[-1])

        # colors = [f"C{i}" for i in assignments]
        # plt.scatter(x[:, 0], x[:, 1], color=colors)
        # plt.scatter(centres[:, 0], centres[:, 1], color="k")
        plt.show()

    print(bics)
    k = min(range(1, 11), key=lambda k: bics[k-1])
    key, subkey = jax.random.split(key)
    centres, assignments = _pc_kmeans(subkey, x, k)

    colors = [f"C{i}" for i in assignments]

    fig, ax = plt.subplots(2)
    ax[0].scatter(x[:, 0], x[:, 1], color=colors)
    ax[0].scatter(centres[:, 0], centres[:, 1], color="k")
    ax[1].plot(range(1, 11), bics, label="BIC")
    ax[1].plot(range(1, 11), gof, label="goodness of fit")
    ax[1].plot(range(1, 11), pen, label="penalty")
    ax[1].legend()
    plt.show()

    assignments = pc_xmeans(x)
    fig, ax = plt.subplots()
    colors = [f"C{i}" for i in assignments]
    ax.scatter(x[:, 0], x[:, 1], color=colors)
    plt.show()
