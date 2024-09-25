import abc
import numpy as np
from utilities import penalty
from utilities import c1, c2, c3, c4
from typing import Callable, Tuple


class ObjectiveFunction(abc.ABC, Callable[[np.ndarray], np.ndarray]):
    minimization_problem = True

    def __init__(self, d, ind, bid, H, c, ratio=None, target_eval=1e-10, max_eval=1e4):
        """
        :param d:
        :param ind:
        :param bid:
        :param H:
        :param c:
        :param ratio:
        :param target_eval:
        :param max_eval:
        """
        if ratio is None:
            ratio = 1.
        self.eval_count = 0  # Number of function evaluations performed so far
        self.d = d
        self.ind = ind
        self.bid = bid
        self.r = ratio
        self.H = H
        self.c = c
        self.N = self.d//2

    @abc.abstractmethod
    def __call__(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        :param X: candidate solutions
        :type X: array_like, shape=(lam, d), dtype=float
        :return: evaluation values
        :rtype: array_like, shape=(lam), dtype=float
        """
        raise NotImplementedError("__call__ not implemented")


class ConstrainedEllipsoid(ObjectiveFunction, abc.ABC):
    def __init__(self, d, ind, bid, H, c, Hqc, E, ratio=None, target_eval=1e-10, max_eval=1e4):
        """
        :param Hqc:
        :param E:
        """
        super(ConstrainedEllipsoid, self).__init__(d, ind, bid, H, c, ratio, target_eval, max_eval)
        self.Hqc = Hqc
        self.E = E

    def __call__(self, X: np.ndarray, flag: bool = False):

        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        self.eval_count += len(X)
        evals = np.full(len(X), np.nan)
        evals_clean = np.full(len(X), np.nan)
        cnstr = np.full(len(X), np.nan)
        sepF = self.H.shape[0] == self.N
        sepG = self.Hqc.shape[0] == self.N

        for k in range(len(X)):
            y, z = self.vars_build(X[k, :])
            if sepF:
                evals[k] = (np.array(y - c1).dot(self.H).dot(np.array(y - c1)) +
                            np.array(z - c1).dot(self.H).dot(np.array(z - c1))) / self.c
            else:
                xc = np.concatenate((y, z))
                evals[k] = (np.array(xc - c3).dot(self.H).dot(np.array(xc - c3))) / self.c
            if sepG:
                cnstr[k] = (np.array(y - c2).dot(self.Hqc).dot(np.array(y - c2)) +
                            np.array(z - c2).dot(self.Hqc).dot(np.array(z - c2))) / self.c
            else:
                xc = np.concatenate((y, z))
                cnstr[k] = (np.array(xc - c4).dot(self.Hqc).dot(np.array(xc - c4))) / self.c
            evals_clean[k] = evals[k]
            if cnstr[k] > self.E:
                evals[k] += penalty*(cnstr[k]-self.E)**2
        # self._update_best_eval(evals)
        return (evals, evals_clean, cnstr) if flag else evals

    @abc.abstractmethod
    def vars_build(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("vars_build() must be implemented for evaluation")


class MixedVarsConstrainedEllipsoid(ConstrainedEllipsoid):
    """
    Ellipsoid function constrained by another Ellipsoid (mixed-integer {x,z})
    """

    def __init__(self, d, ind, bid, H, c, Hqc, E, ratio=None, target_eval=1e-10, max_eval=1e4):
        super(MixedVarsConstrainedEllipsoid, self).__init__(d, ind, bid, H, c, Hqc, E, ratio, target_eval, max_eval)
        ObjectiveFunction.minimization_problem = True

    def vars_build(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return np.array([X[i] for i in range(0, self.N)]),  np.array([np.round(X[i]) for i in range(self.N, self.d)])


class IntConstrainedEllipsoid(ConstrainedEllipsoid):
    """
    Integer Ellipsoid function constrained by another Integer Ellipsoid
    """

    def __init__(self, d, ind, bid, H, c, Hqc, E, ratio=None, target_eval=1e-10, max_eval=1e4):
        super(IntConstrainedEllipsoid, self).__init__(d, ind, bid, H, c, Hqc, E, ratio, target_eval, max_eval)
        ObjectiveFunction.minimization_problem = True

    def vars_build(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return (np.array([np.round(X[i]) for i in range(0, self.N)]),
                np.array([np.round(X[i]) for i in range(self.N, self.d)]))


