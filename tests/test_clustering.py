import pytest
import numpy as np
from clustering import xmeans, knn, expectation_maximisation
from pyclustering.utils import read_sample
from pyclustering.samples.definitions import SIMPLE_SAMPLES


@pytest.fixture
def position_matrix():
    return np.array(read_sample(SIMPLE_SAMPLES.SAMPLE_SIMPLE3))


@pytest.mark.parametrize("algorithm", [xmeans, knn, expectation_maximisation])
def test_xmeans(algorithm, position_matrix):
    cluster_list = algorithm(position_matrix)
    assert max(cluster_list) + 1 == 4
