from dataclasses import dataclass, fields
from typing import Optional, List, Callable, TypeVar, Mapping, Dict
import numpy as np
import scipy.linalg
from abc import ABC
import itertools
import copy
import toolz


K = TypeVar("K")
V = TypeVar("V")
W = TypeVar("W")


def valmap(f: Callable[[V], W], d: Mapping[K, V]) -> Dict[K, W]:
    """ Wrapper around `toolz.valmap`. Helps with type inference."""
    return toolz.valmap(f, d)


@dataclass
class SemiDef:
    eig: Optional[List[float]] = None

    matrix: Optional[np.ndarray] = None

    def __post_init__(self):
        defined_eig: bool = self.eig is not None
        defined_mat: bool = self.matrix is not None

        assert defined_mat or defined_eig

        if defined_eig and type(self.eig) is not list:
            self.eig = self.eig.tolist()

        if defined_eig and defined_mat:
            mat_eig: List[float] = SemiDef.eig_from_semidef(self.matrix)
            mat_eig.sort()
            self.eig.sort()

            assert mat_eig == self.eig
        elif defined_eig:
            self.matrix = SemiDef.semidef_from_eigenvalues(self.eig)
        else:
            self.eig = SemiDef.eig_from_semidef(self.matrix)

        assert all(i >= 0 for i in self.eig)

        assert np.allclose(self.matrix, self.matrix.T)

    @staticmethod
    def semidef_from_eigenvalues(eigenvalues):
        n = len(eigenvalues)
        s = np.diag(eigenvalues)
        q, _ = scipy.linalg.qr(np.random.rand(n, n))
        semidef = q.T @ s @ q
        return semidef

    @staticmethod
    def eig_from_semidef(semidef) -> List[float]:
        eig = np.linalg.eigvalsh(semidef).tolist()
        eig = [round(i, 6) for i in eig]
        return eig

    def __eq__(self, other):
        self.eig.sort()
        other.eig.sort()
        return self.eig == other.eig

    def __lt__(self, other):
        self.eig.sort()
        other.eig.sort()

        return all(self.eig[i] < other.eig[i] for i, _ in enumerate(self.eig))

    def __le__(self, other):
        self.eig.sort()
        other.eig.sort()

        return all(self.eig[i] <= other.eig[i] for i, _ in enumerate(self.eig))


def func(values):
    return True


@dataclass
class BaseParams(ABC):
    condition: Callable = func

    def __post_init__(self):
        lists = []
        single_list = []
        for field in fields(self):
            values = getattr(self, field.name)
            res = self.process_mutually_exclusive_values(values)
            lists.append(res)
            single_list.append(len(res) == 1)

        self.xplets = list(itertools.product(*lists))
        self.is_single = all(single_list)
        if self.is_single:
            self.n_total = 1
        else:
            helper = copy.deepcopy(self.xplets)
            counter = 0
            for xplet in self.xplets:
                counter += 1
                try:
                    new_instance = self.__new__(type(self))
                    new_instance.__init__(*xplet)
                except Exception as e:
                    helper.remove(xplet)

            self.xplets = helper
            self.n_total = len(self.xplets)

    def process_mutually_exclusive_values(self, values):
        return_values = []

        def process_single_value(inst):
            if self.is_nested(inst):
                for val in inst.gen():
                    return_values.append(val)
            else:
                return_values.append(inst)

        if isinstance(values, list):
            for value in values:
                process_single_value(value)
        else:
            process_single_value(values)
        return return_values

    @staticmethod
    def is_nested(value):
        try:
            value.get_count()
            return True
        except AttributeError:
            return False

    def get_count(self):
        return self.n_total

    def gen(self):
            for xplet in self.xplets:
                if self.condition(xplet):
                    new_instance = self.__new__(type(self))
                    new_instance.__init__(*xplet)
                    yield new_instance
