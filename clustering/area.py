import numpy as np
from clustering.knn import distance2_knn
import flexknot
from tqdm import tqdm


class AreaCluster:
    def __init__(self, x_min, x_max, y_min, y_max, adaptive=False):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        if adaptive:
            self.flexknot = flexknot.AdaptiveKnot(x_min, x_max)
            self.prior = flexknot.AdaptiveKnotPrior(x_min, x_max, y_min, y_max)
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
        return distance2_knn(areas)
