import abc
import cma
import numpy as np
from typing import Callable, Union
from utilities import Population
from ObjectiveFunctions import ObjectiveFunction


class Optimizer(abc.ABC):
    def __init__(self, dim: int, maxEval: int, nI: int, nR: int, lowerBound: float, upperBound: float,
                 verbose: bool, optimalSolution: Union[float, np.nan]):
        """
        :param dim:
        :param maxEval:
        :param nI:
        :param nR:
        :param lowerBound:
        :param upperBound:
        :param verbose:
        :param optimalSolution:
        """
        self.dim, self.maxEval = dim, maxEval
        self.nI, self.nR = nI, nR
        self.lowerBound, self.upperBound = lowerBound, upperBound
        self.verbose = verbose
        self.optimalSolution = optimalSolution
        self.history = {}

    @abc.abstractmethod
    def __call__(self, problem: Callable, **kwargs):
        raise NotImplementedError("__call__ not implemented")


class MIES(Optimizer):
    def __init__(self, bounds, dim: int, maxEval: int, nI: int, nR: int, lowerBound: float, upperBound: float,
                 verbose: bool, optimalSolution: Union[float, np.nan]):
        super(MIES, self).__init__(dim, maxEval, nI, nR, lowerBound, upperBound, verbose, optimalSolution)
        self.bounds = bounds

    def __call__(self, problem: ObjectiveFunction, **kwargs):
        mu, lmbda, comma, select, mutateR, mutateI, init, bottleNeck, pR = MIES.__readVars(**kwargs)
        # Generate first population
        X, s = init(mu, self.nR, self.nI, self.bounds)
        f = problem(X)      # Evaluate the random population generated using 'init' function
        flag = self.__save(X, f, iteration=1, minimization=problem.minimization_problem)

        # ES
        for i in range(mu, self.maxEval + 1, lmbda):
            if flag:
                if self.verbose:
                    print(f'optimum result found on iteration #{i}.')
                break
            indices = select(f, lmbda, problem.minimization_problem, bottleNeck)
            Xne, Sne = X[indices, :], s[indices]

            # recombination
            if np.random.rand() < pR:
                for j in range(lmbda//2):
                    Xne[j, :], Xne[j+1, :], Sne[j, :], Sne[j+1, :] = (
                        Population.recombine(Xne[j, :], Xne[j+1, :], Sne[j, :], Sne[j+1, :]))
            # mutation and sigmas updates
            for j in range(lmbda):
                NcI, NcR = np.random.randn(), np.random.randn()
                Xne[j, :self.nR], Sne[j, :self.nR] = (
                    mutateR(Xne[j, :self.nR], self.bounds[:self.nR], Sne[j, :self.nR], NcR))
                Xne[j, self.nR:], Sne[j, self.nR:] = \
                    mutateI(Xne[j, self.nR:], self.bounds[self.nR:], Sne[j, self.nR:], NcI)
            if comma:
                X, f, s = Xne, problem(Xne), Sne
            else:
                X = np.vstack((X, Xne))
                s = np.vstack((s, Sne))
                f = np.hstack((f, problem(Xne)))
            sorted_indices = np.argsort(f)[:mu] if problem.minimization_problem else np.argsort(-f)[:mu]
            X = X[sorted_indices, :]
            s = s[sorted_indices, :]
            f = f[sorted_indices]

            flag = self.__save(X, f, iteration=i, minimization=problem.minimization_problem)

        return self.history

    def __save(self, X, f, iteration: int, minimization: bool):
        best_f = np.min(f) if minimization else np.max(f)
        best_x = X[np.argmax(f == best_f)]
        best = np.hstack(([iteration, best_f], best_x))
        if iteration == 1 or (minimization and best_f < self.history[1]) or (not minimization and best_f > self.history[1]):
            self.history = best

        if self.verbose and iteration % 1 == 0:
            print(f'iteration {iteration} best value {min(f)} (Global best = {self.history[1]})')

        if np.isclose(best_f, self.optimalSolution, atol=1e-8):
            return True
        return False

    @staticmethod
    def __readVars(**kwargs):
        pR = kwargs.get('pR', 0)
        size = kwargs.get('size', 15)
        lmbda = kwargs.get('lmbda', 100)
        comma = kwargs.get('comma', True)
        bottleNeck = kwargs.get('bottleNeck', 3)

        select = kwargs.get('select_func', Population.select_normalized)
        mutateR = kwargs.get('mutate_real', Population.mutation_normal)
        mutateI = kwargs.get('mutate_integer', Population.mutation_geometric)
        # step = kwargs.get('step', Population.step)  # return for sigma scalar uses.

        init = kwargs.get('init_func', Population.generate_random)
        # return size, lmbda, comma, select, mutateR, mutateI, init, bottleNeck, pR, step
        return size, lmbda, comma, select, mutateR, mutateI, init, bottleNeck, pR


class CMA(Optimizer):

    def __init__(self,  options: dict, dim: int, maxEval: int, nI: int, nR: int, lowerBound: float, upperBound: float,
                 verbose: bool, optimalSolution: Union[float, np.nan]):
        super(CMA, self).__init__(dim, maxEval, nI, nR, lowerBound, upperBound, verbose, optimalSolution)

        # CMA algorithm run options cma.fim2(option...)
        self.options = options

    def __call__(self, problem: Callable, **kwargs):
        LB, UB, iteration = CMA.__readVars(**kwargs)
        x, s = Population.generate_random(1, self.nR, self.nI, [(LB, UB)] * self.dim)

        xx = cma.fmin(problem, x0=x, sigma0=s[0][0], options=self.options, restarts=0)[0]
        if self.nI == self.dim:
            xx = np.round(xx)
            return np.hstack(([iteration], problem(xx), xx))

        xx = np.array([xx[index] if index < self.nI else np.round(xx[index]) for index in range(len(xx))])
        return np.hstack(([iteration, problem(xx)], xx))

    @staticmethod
    def __readVars(**kwargs):
        LB = kwargs.get('lb', -np.inf)
        UB = kwargs.get('ub', np.inf)
        iteration = kwargs.get('iteration', -1)
        return LB, UB, iteration
