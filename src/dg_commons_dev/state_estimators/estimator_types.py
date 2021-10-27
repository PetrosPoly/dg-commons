from abc import ABC, abstractmethod
from dg_commons import U, X
from dataclasses import dataclass
from dg_commons_dev.utils import BaseParams


@dataclass
class EstimatorParams(BaseParams):
    pass


class Estimator(ABC):
    @abstractmethod
    def update_prediction(self, uk: U):
        pass

    @abstractmethod
    def update_measurement(self, mk: X):
        pass


@dataclass
class DroppingTechniquesParams(BaseParams):
    pass


class DroppingTechniques(ABC):
    @abstractmethod
    def drop(self) -> bool:
        pass
