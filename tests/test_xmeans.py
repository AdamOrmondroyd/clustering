import pytest
import numpy as np
from clustering import xmeans, sklearn_xmeans
from clustering.bic import bic
from pyclustering.utils import read_sample
from pyclustering.samples.definitions import SIMPLE_SAMPLES


@pytest.fixture
def position_matrix():
    return np.array(read_sample(SIMPLE_SAMPLES.SAMPLE_SIMPLE3))


@pytest.mark.parametrize('algorithm', [xmeans, sklearn_xmeans])
def test_xmeans(algorithm, position_matrix):
    cluster_list = algorithm(position_matrix)
    assert max(cluster_list) + 1 == 4


def test_bic():
    assert np.isclose(-np.log(2 * np.pi) - 11 / 2 * np.log(2) - 1 / 2,
                      bic(np.array([[0, 0], [2, 2]]),
                          np.array([0, 0]), np.array([[1, 1]])))

    assert np.isclose(-20 * np.log(2) - 2 * np.log(2 * np.pi),
                      bic(np.array([[0, 0], [2, 2], [10, 10], [8, 8]]),
                          np.array([0, 0, 1, 1]),
                          np.array([[1, 1], [9, 9]]),
                          )
                      )
