import copy
from dataclasses import dataclass
from typing import Union, List, Tuple, Optional
import math
import numpy as np
import itertools
from dg_commons_dev.utils import BaseParams
from scipy.interpolate import griddata


@dataclass
class ExponentialParams(BaseParams):
    """
    Exponential Distribution Parameters
    """

    lamb: float = 1
    """ lambda > 0 """

    def __post_init__(self):
        assert self.lamb > 0


class Exponential:
    """
    Exponential Distribution

    f(t) = lambda * exp(- lambda * t) if t >= 0
    f(t) = 0 otherwise
    """
    REF_PARAMS: dataclass = ExponentialParams

    def __init__(self, params: ExponentialParams):
        self.params = params

    def cdf(self, t):
        """
        Cumulative distribution function
        @param t: val
        @return: evaluation at t
        """
        assert t >= 0

        return 1 - math.exp(-self.params.lamb * t)

    def pdf(self, t):
        """
        Probability density function
        @param t: val
        @return: evaluation at t
        """
        assert t >= 0

        return self.params.lamb * math.exp(-self.params.lamb * t)


@dataclass
class PoissonParams(BaseParams):
    """
    Poisson Distribution Parameters
    """

    lamb: float = 1
    """ lambda > 0 """

    def __post_init__(self):
        assert self.lamb > 0


class Poisson:
    """
    Poisson Distribution

    P(k Events) = lambda^k * exp(- lambda) / k!
    """
    REF_PARAMS: dataclass = PoissonParams

    def __init__(self, params: PoissonParams):
        self.params = params

    def pmf(self, k: int) -> float:
        """
        Probability mass function
        @param k: number of events
        @return: probability of k events
        """
        assert isinstance(k, int)

        return math.pow(self.params.lamb, k) * math.exp(- self.params.lamb) / math.factorial(k)

    def cdf(self, k: int) -> float:
        """
        Cumulative distribution function
        @param k: number of events
        @return: cumulative evaluation
        """
        k = int(k)

        return_val = 0
        for i in range(k):
            return_val += self.pmf(i)
        return_val += self.pmf(k)

        return return_val


class GridOneD:
    """ Manages one dimensional grid of values"""

    def __init__(self, length: float, n_val: int, values=None):
        self.length: float = length
        self.nodes: List[float] = list(np.linspace(-length/2, length/2, num=n_val))

        if values is not None:
            assert len(values) == n_val
            self._values: List[float] = values
        else:
            self._values: List[float] = len(self.nodes) * [math.nan]

    @property
    def values(self) -> List[float]:
        """
        Values getter
        @return: Values
        """
        return self._values

    def as_numpy(self) -> np.ndarray:
        """
        @return: Values as numpy array
        """
        return np.array(self.values)

    @values.setter
    def values(self, values: List[float]):
        """
        Values setter enforcing correct dimensions
        """
        assert len(values) == len(self.nodes)
        self._values = values

    def index_by_position(self, pos: float) -> Tuple[int, float]:
        """
        Returns index corresponding to current position
        @param pos: Position of interest
        @return: Index of closest node and interpolated index
        """
        assert self.nodes[0] <= pos <= self.nodes[-1]

        closest_idx = min(range(len(self.nodes)), key=lambda i: abs(self.nodes[i]-pos))

        if self.nodes[closest_idx] < pos:
            other_idx = closest_idx + 1
        else:
            other_idx = closest_idx - 1

        idx = (closest_idx * abs(self.nodes[other_idx] - pos) +
               other_idx * abs(self.nodes[closest_idx] - pos)) / \
            abs(self.nodes[other_idx] - self.nodes[closest_idx])

        return closest_idx, idx

    def position_by_index(self, idx: float) -> Tuple[float, float]:
        """
        Returns position corresponding to passed index
        @param idx: Index of interest
        @return: Position of closest node and interpolated position
        """
        assert 0 <= idx < len(self.nodes)

        previous_idx, next_idx, rest = int(idx), int(idx) + 1, round(idx % 1, 4)
        if not next_idx < len(self.nodes):
            next_idx = previous_idx

        if rest < 0.5:
            closest_pos: float = self.nodes[previous_idx]
        else:
            closest_pos: float = self.nodes[next_idx]

        pos = self.nodes[previous_idx] * (1 - rest) + self.nodes[next_idx] * rest
        return closest_pos, pos

    def value_by_index(self, idx: float) -> Tuple[float, float]:
        """
        Returns value corresponding to passed index
        @param idx: Index of interest
        @return: Value at the passed index
        """
        assert all([value is not math.nan for value in self.values])
        assert 0 <= idx < len(self.nodes)

        previous_idx, next_idx, rest = int(idx), int(idx) + 1, round(idx % 1, 4)
        if not next_idx < len(self.nodes):
            next_idx = previous_idx

        if rest < 0.5:
            closest_val: float = self.values[previous_idx]
        else:
            closest_val: float = self.values[next_idx]

        val = self.values[previous_idx] * (1 - rest) + self.values[next_idx] * rest
        return closest_val, val

    def value_by_position(self, pos: float) -> Tuple[float, float]:
        """
        Returns value corresponding to passed position
        @param pos: position of interest
        @return: Value at the passed position
        """

        _, idx = self.index_by_position(pos)
        return self.value_by_index(idx)

    def set(self, idx: int, value: float):
        """
        Set the value of a grid node
        @param idx: Index of the node
        @param value: Value to set
        """
        assert isinstance(idx, int)

        self.values[idx] = value


class GridTwoD:
    """ Manages two-dimensional grid of values"""

    def __init__(self, length: float, n_length: int, width: float, n_width: int, values: np.ndarray = None):
        self.area: float = length * width
        self.length_grid: GridOneD = GridOneD(length, n_length)
        self.width_grid: GridOneD = GridOneD(width, n_width)

        if values is not None:
            assert values.shape == (n_length, n_width)
            self._values: np.ndarray = values
        else:
            self._values = np.zeros((n_length, n_width))

    @property
    def values(self) -> np.ndarray:
        """
        Values getter
        @return: Values
        """
        return self._values

    def as_numpy(self) -> np.ndarray:
        """
        @return: Values as numpy array
        """
        return np.array(self.values)

    @values.setter
    def values(self, values: np.ndarray):
        """
        Values setter enforcing correct dimensions
        """
        assert (values.shape[0], values.shape[1]) == (len(self.length_grid.nodes), len(self.width_grid.nodes))
        self._values = values

    def set(self, idx: Tuple[int, int], value: float):
        """
        Set the value of a grid node
        @param idx: X-Index and Y-Index of the node
        @param value: Value to set
        """
        assert isinstance(idx[0], int)
        assert isinstance(idx[1], int)
        self._values[idx[0], idx[1]] = value

    def index_by_position(self, pos: List[float]) -> Tuple[float, float]:
        """
        Returns x- and y-indices corresponding to current position
        @param pos: Position of interest
        @return: Interpolated indices
        """
        _, idx_x = self.length_grid.index_by_position(pos[0])
        _, idx_y = self.width_grid.index_by_position(pos[1])
        return idx_x, idx_y

    def position_by_index(self, idx: Tuple[float, float]) -> Tuple[float, float]:
        """
        Returns x- and y-indices corresponding to current position
        @param idx: X-index and Y-index of interest
        @return: Interpolated position
        """
        _, pos_x = self.length_grid.position_by_index(idx[0])
        _, pos_y = self.width_grid.position_by_index(idx[1])
        return pos_x, pos_y

    def value_by_index(self, idx: Tuple[float, float]) -> float:
        """
        Returns value corresponding to passed indices
        @param idx: X-index and Y-index of interest
        @return: Value at the passed indices
        """
        top_edge: bool = int(idx[0]) == len(self.length_grid.nodes) - 1
        right_edge: bool = int(idx[1]) == len(self.width_grid.nodes) - 1

        idxs_x = [int(idx[0]), int(idx[0]) + 1]
        idxs_y = [int(idx[1]), int(idx[1]) + 1]
        if top_edge:
            idxs_x = [int(idx[0])-1, int(idx[0])]
        if right_edge:
            idxs_y = [int(idx[1])-1, int(idx[1])]
        indices = itertools.product(idxs_x, idxs_y)

        values, xs, ys = [], [], []
        for idx_pair in indices:
            values.append(self.values[idx_pair[0], idx_pair[1]])
            pos = self.position_by_index(idx_pair)
            xs.append(pos[0])
            ys.append(pos[1])

        assert all([val is not None for val in values])
        pos_of_interest = self.position_by_index(idx)

        points = np.array((np.array(xs).flatten(), np.array(ys).flatten())).T
        values = np.array(values).flatten()

        return float(griddata(points, values, (pos_of_interest[0], pos_of_interest[1])))

    def value_by_position(self, pos: List[float]) -> float:
        """
        Returns value corresponding to passed position
        @param pos: position of interest
        @return: Value at the passed position
        """

        idx = self.index_by_position(pos)
        return self.value_by_index(idx)


class CircularGrid:
    """ Manages circular grid of values"""

    def __init__(self, radius: float, n_point_per_line: int, n_lines: int, d_first_ring: float,
                 values: np.ndarray = None):
        self.radius = radius
        self.n_lines = n_lines
        self.n_points = n_point_per_line
        self.area: float = radius ** 2 * math.pi

        self.degrees = np.linspace(0, 2 * math.pi, n_lines + 1)[:-1]
        self.distances = np.linspace(d_first_ring, radius, n_point_per_line)

        self._pos_x = np.zeros((n_lines, n_point_per_line))
        self._pos_y = np.zeros((n_lines, n_point_per_line))
        for i in range(n_lines):
            for j in range(n_point_per_line):
                self._pos_x[i, j] = math.cos(self.degrees[i]) * self.distances[j]
                self._pos_y[i, j] = math.sin(self.degrees[i]) * self.distances[j]

        if values is not None:
            assert values.shape == (n_lines, n_point_per_line)
            self._values: np.ndarray = values
        else:
            self._values = np.zeros((n_lines, n_point_per_line))

    @property
    def values(self) -> np.ndarray:
        """
        Values getter
        @return: Values
        """
        return self._values

    def as_numpy(self) -> np.ndarray:
        """
        @return: Values as numpy array
        """
        return np.array(self.values)

    @property
    def pos_x(self):
        """
        pos_x getter
        @return: x positions
        """
        return self._pos_x

    @property
    def pos_y(self):
        """
        pos_y getter
        @return: y positions
        """
        return self._pos_y

    @values.setter
    def values(self, values: np.ndarray):
        """
        Values setter enforcing correct dimensions
        """
        assert (values.shape[0], values.shape[1]) == (self.n_lines, self.n_points)
        self._values = values

    def set(self, idx: Tuple[int, int], value: float):
        """
        Set the value of a grid node
        @param idx: Theta-Index and dist-Index of the node
        @param value: Value to set
        """
        assert isinstance(idx[0], int)
        assert isinstance(idx[1], int)
        self._values[idx[0], idx[1]] = value

    def index_by_position(self, pos: List[float]) -> Tuple[float, float]:
        """
        Returns theta- and distance-indices corresponding to current position
        @param pos: Position of interest
        @return: Interpolated indices
        """
        assert self.conservative_in(np.array(pos))

        dist = np.linalg.norm(np.array(pos))
        theta = math.atan2(pos[1], pos[0])
        theta = theta + 2 * math.pi if theta < 0 else theta

        upper_theta = np.argmax(self.degrees >= theta)
        if self.degrees[upper_theta] == theta or upper_theta == 0:
            idx_theta = upper_theta
        else:
            lower_theta = upper_theta - 1
            idx_theta = lower_theta + (theta - self.degrees[lower_theta]) / \
                (self.degrees[upper_theta] - self.degrees[lower_theta])

        upper_d = np.argmax(self.distances >= dist)
        if self.distances[upper_d] == dist or upper_d == 0:
            idx_d = upper_d
        else:
            lower_d = upper_d - 1
            idx_d = lower_d + (dist - self.distances[lower_d]) / \
                (self.distances[upper_d] - self.distances[lower_d])

        return idx_theta, idx_d

    def position_by_index(self, idx: Tuple[float, float]) -> Tuple[float, float]:
        """
        Returns x- and y-indices corresponding to current position
        @param idx: Theta-index and Distance-index of interest
        @return: Interpolated position
        """
        assert 0 <= idx[0] <= self.n_lines-1
        assert 0 <= idx[1] <= self.n_points-1

        lower_theta, upper_theta = math.floor(idx[0]), math.ceil(idx[0])
        lower_d, upper_d = math.floor(idx[1]), math.ceil(idx[1])

        theta = self.degrees[lower_theta]
        if lower_theta != upper_theta:
            delta_theta = idx[0] - lower_theta
            theta += delta_theta * (self.degrees[upper_theta] - self.degrees[lower_theta]) / (upper_theta - lower_theta)

        distance = self.distances[lower_d]
        if lower_d != upper_d:
            delta_d = idx[1] - lower_d
            distance += delta_d * (self.distances[upper_d] - self.distances[lower_d]) / (upper_d - lower_d)

        pos_x = math.cos(theta) * distance
        pos_y = math.sin(theta) * distance
        return pos_x, pos_y

    def value_by_index(self, idx: Tuple[float, float]) -> float:
        """
        Returns value corresponding to passed indices
        @param idx: Theta-index and Distance-index of interest
        @return: Value at the passed indices
        """
        assert 0 <= idx[0] <= self.n_lines-1
        assert 0 <= idx[1] <= self.n_points-1

        lower_theta, upper_theta = int(idx[0]), int(idx[0]) + 1
        lower_d, upper_d = int(idx[1]), int(idx[1]) + 1

        idx_theta = [lower_theta, upper_theta]
        idx_d = [lower_d, upper_d]
        if not upper_theta <= self.n_lines - 1:
            idx_theta = [lower_theta-1, lower_theta]
        if not upper_d <= self.n_points - 1:
            idx_d = [lower_d-1, lower_d]

        indices = itertools.product(idx_theta, idx_d)

        values, xs, ys = [], [], []
        for idx_pair in indices:
            values.append(self.values[idx_pair[0], idx_pair[1]])
            pos = self.position_by_index(idx_pair)
            xs.append(pos[0])
            ys.append(pos[1])

        assert all([val is not None for val in values])
        pos_of_interest = self.position_by_index(idx)

        points = np.array((np.array(xs).flatten(), np.array(ys).flatten())).T
        values = np.array(values).flatten()

        res = float(griddata(points, values, (pos_of_interest[0], pos_of_interest[1])))

        return res

    def value_by_position(self, pos: List[float]) -> float:
        """
        Returns interpolated value corresponding to passed position
        @param pos: position of interest
        @return: Value at the passed position
        """
        assert self.conservative_in(np.array(pos))

        idx = self.index_by_position(pos)
        return self.value_by_index(idx)

    def gen_structure(self):
        """
        Generator yielding elements of the grid structure:

        for each degree:
            for each distance:
                yield corresponding index and x- and y-position
        """
        for i in range(self.n_lines):
            for j in range(self.n_points):
                idx: Tuple[int, int] = (i, j)
                position = np.array([self.pos_x[i, j], self.pos_y[i, j]])
                yield idx, position

    def conservative_in(self, pos: np.ndarray) -> bool:
        """
        @return: whether a passed position is conservately inside the equilateral polygon
        """
        distance = np.linalg.norm(pos)
        return distance < self.radius * math.cos(self.degrees[1]/2)


PDistribution = Union[Exponential]
PDistributionParams = Union[ExponentialParams]

PDistributionDis = Union[Poisson]
PDistributionDisParams = Union[PoissonParams]
