import pytest
import numpy as np
from sklearn import datasets
from clustering import xmeans, knn, dbscan, optics
from clustering.relabel import relabel
from pyclustering.utils import read_sample
from pyclustering.samples.definitions import SIMPLE_SAMPLES

algorithm_list = [xmeans, knn, dbscan, optics]

# @pytest.fixture
# def position_matrix():
pyclustering_samples = np.array(read_sample(SIMPLE_SAMPLES.SAMPLE_SIMPLE3))

moons_samples = datasets.make_moons(n_samples=200)  # , noise=0.05)
circles_samples = datasets.make_circles(n_samples=200)  # , noise=0.05)
blobs_samples = datasets.make_blobs(n_samples=200)  # , noise=0.05)

samples_list = [moons_samples, circles_samples, blobs_samples]


@pytest.mark.parametrize("algorithm", algorithm_list)
@pytest.mark.parametrize("samples", samples_list)
def test_xmeans(algorithm, samples):
    position_matrix = samples[0]
    cluster_list = algorithm(position_matrix)
    assert np.all(cluster_list == relabel(samples[1]))
