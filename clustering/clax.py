import jax
from jax import numpy as jnp
from matplotlib import pyplot as plt
from timeit import timeit


@jax.jit
def p_distance_2(x, centres):
    """
    Calculate the probability of each x being chosen as a new centre.

    Probability is proportional to the distance squared from the nearest centre.

    """
    # subtract every x(num_x, ndims) from every centre(num_centres, ndims)
    distances = x[:, None] - centres[None, :]

    distance_2s = jnp.sum(distances ** 2, axis=-1)
    # get the minimum distance2 for each x
    distance_2s = jnp.min(distance_2s, axis=-1)
    # return the normalised distance2
    return distance_2s / jnp.sum(distance_2s)


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
    distance_2s = jnp.sum(distances ** 2, axis=-1)
    assignments = jnp.argmin(distance_2s, axis=-1)
    return assignments


def update_centres(x, assignments):
    k = assignments.max() + 1
    cluster_counts = jnp.bincount(assignments, length=k)
    cluster_sums = jax.ops.segment_sum(x, assignments, k)
    centroids = cluster_sums / cluster_counts[:, None]

    centroids = jnp.where(jnp.isnan(centroids), 0, centroids)
    return centroids


def kmeans(key, x, k, max_iter=100):
    centres = kmeans_plusplus_initialiser(key, x, k)
    for i in range(max_iter):
        assignments = assign(x, centres)
        # update the centres
        centres = update_centres(x, assignments)
        # assert jnp.allclose(centroids, centres, atol=1e-3)
    return centres, assignments


if __name__ == "__main__":
    key = jax.random.PRNGKey(1)
    x = jnp.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    k = 3
    key, subkey0 = jax.random.split(key)
    key, subkey1 = jax.random.split(key)
    x = jnp.vstack([
        jax.random.normal(subkey0, (100, 2)),
        jax.random.normal(subkey1, (100, 2)) + 2
    ])
    key, subkey = jax.random.split(key)
    centres, assignments = kmeans(subkey, x, 4)
    # print(timeit("kmeans(key, x, 2)", globals=globals(), number=100))

    colors = [f"C{i}" for i in assignments]

    plt.scatter(x[:, 0], x[:, 1], color=colors)
    plt.show()
