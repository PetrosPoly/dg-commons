from dataclasses import dataclass
from typing import Union, List
import math
from dg_commons_dev.utils import BaseParams


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

    def __init__(self, params: ExponentialParams):
        self.params = params

    def pmf(self, k: int) -> float:
        """
        Probability mass function
        @param k: number of events
        @return: probability of k events
        """
        assert isinstance(k, int)

        return math.pow(self.params.lamb, k) * math.exp(- self.params.lamb) / math.factorial(k)

    def pdf(self, k: int) -> float:
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


PDistribution = Union[Exponential]
PDistributionParams = Union[ExponentialParams]
