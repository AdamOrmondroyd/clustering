import numpy as np
from clustering.knn import distance2_knn
import flexknot
from tqdm import tqdm


class AreaMetric:
    def __init__(self, x_min, x_max, y_min, y_max,
                 adaptive=False, N_min=None, N_max=None):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        if adaptive:
            if N_min is None or N_max is None:
                raise ValueError("Must provide N_min and N_max for adaptive case")
            self.flexknot = flexknot.AdaptiveKnot(x_min, x_max)
            self.prior = flexknot.AdaptiveKnotPrior(x_min, x_max, y_min, y_max, N_min, N_max)
        else:
            self.flexknot = flexknot.FlexKnot(x_min, x_max)
            self.prior = flexknot.FlexKnotPrior(x_min, x_max, y_min, y_max)

    def __call__(self, position_matrix):
        print("Area clustering", flush=True)
        areas = np.zeros((len(position_matrix), len(position_matrix)))
        for ii, cubeii in enumerate(tqdm(position_matrix)):
            thetaii = self.prior(cubeii)
            for i, cubei in enumerate(position_matrix):
                thetai = self.prior(cubei)
                areas[ii, i] = self.flexknot.area(thetai, thetaii)
                areas[ii, i] /= ((self.x_max - self.x_min) * (self.y_max
                                                              - self.y_min))
        return areas


class Metric:
    def __init__(self, metrics, params):
        """
        metrics are callables which return squared distance matrix,
        and params list the indices used for each metric
        """
        if len(metrics) != len(params):
            raise ValueError("Need same number of metrics as parameter lists")
        self.metrics = metrics
        self.params = params

    def __call__(self, position_matrix):
        distance2_matrix = np.zeros((len(position_matrix),
                                     len(position_matrix)))
        for g, p in zip(self.metrics, self.params):
            distance2_matrix += g(position_matrix[:, p])
        return distance2_matrix


class MetricCluster:
    """
    This can be used to combine AreaDistance with standard distance2 KNN.
    """

    def __init__(self, metric):
        """
        metric: Callable. Takes a position matrix and
        returns squared distance matrix
        """
        self.metric = metric

    def __call__(self, position_matrix):
        return distance2_knn(self.metric(position_matrix))
