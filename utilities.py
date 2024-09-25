from typing import List, Tuple
import numpy as np
from scipy.stats import truncnorm
from scipy.linalg import hadamard

# Useful variables
D = 32
c1, c2, c3, c4 = np.array([1, -1]), np.array([1, -1]), np.array([1, -1]), np.array([1, -1])
penalty = D * D * 10**4


def setC(dim: int,  alpha1: int = 7, alpha2: int = 4, alpha3: int = 7, alpha4: int = 4):
    global c1, c2, c3, c4
    c1 = np.array([alpha1, -alpha1] * (dim//2))
    c2 = np.array([-alpha2, alpha2] * (dim // 2))
    c3 = np.array([alpha3, -alpha3] * dim)
    c4 = np.array([-alpha4, alpha4] * dim)


def generate_centers(dim: int,
                     base0: np.ndarray = np.array([7, -7]),
                     base1: np.ndarray = np.array([-4, 4]),
                     trans: int = 0):
    global c1, c2, c3, c4
    c1 = trans + np.tile(base0, dim//2)
    c2 = trans + np.tile(base1, dim//2)
    c3 = trans + np.tile(base0, dim)
    c4 = trans + np.tile(base1, dim)

    return c1, c2, c3, c4


def getRotation(N, theta):
    v = np.ones((N, 1))
    u = np.ones((N, 1))
    for i in range(N):
        if np.mod(i, 2) == 0:
            v[i] = 0
        else:
            u[i] = 0
    v = v / np.linalg.norm(v)
    u = u / np.linalg.norm(u)
    R = np.eye(N) + np.sin(theta) * (u.dot(np.transpose(v)) - v.dot(np.transpose(u))) + (np.cos(theta) - 1.0) * (
                v.dot(np.transpose(v)) + u.dot(np.transpose(u)))
    return R


# Generate and Mutate population for the Algorithms in Algorithms.py
class Population:
    @staticmethod
    def generate_random(size: int,
                        nR: int,
                        nI: int,
                        bounds: List[Tuple[float, float]],
                        LOWER: float = -10**6,
                        UPPER: float = 10**6,
                        adaptiveStep: str = 'personal') -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates a random population of individuals for optimization algorithms.
        :param size: The number of individuals in the population.
        :param nR:  The number of real-valued variables per individual.
        :param nI:  The number of integer-valued variables per individual.
        :param bounds: A list of tuples, where each tuple defines the lower and upper bounds for a variable.
                       The length of the list must be nR + nI.
        :param LOWER: A global lower bound for real-valued variables. (Default value = -10**6)
        :param UPPER: A global upper bound for real-valued variables. (Default value = +10**6)
        :param adaptiveStep: The type of adaptive step size strategy to use. Can be 'personal' or 'global'.
                            (Default value = 'personal')
        :return: A tuple containing:
                    - A numpy array; shape =  (size, nR+nI), representing the population.
                    - A numpy array; shape = (size, nR+nI) or (size, 2) representing the initial step sizes.
        """
        if adaptiveStep == 'personal' and len(bounds) != nR + nI:
            raise ValueError('Each variable must have its own bounds when using an adaptive step size with a "personal"'
                             ' strategy. Therefore, len(bounds) must be nR + nI.')
        lb = np.array([t[0] if t[0] > -np.inf else LOWER for t in bounds])
        ub = np.array([t[1] if t[1] < np.inf else UPPER for t in bounds])

        X = np.zeros((size, nR+nI))
        X[:, :nR] = np.random.uniform(lb[:nR], ub[:nR], size=(size, nR))
        X[:, nR:] = np.random.randint(lb[nR:], ub[nR:], size=(size, nI))

        if adaptiveStep == 'personal':
            S = np.zeros((size, nR+nI))
            S[:, :nR] = (ub[:nR] - lb[:nR]) / 3
            S[:, nR:] = (ub[nR:] - lb[nR:]) / np.sqrt(nI)
        elif adaptiveStep == 'global':
            print('adaptive step size = "global"; assumes all continuous variables have the same bounds '
                  'and all discrete variables have the same bounds.')
            S = np.zeros((size, 2))
            S[:, 0] = max(LOWER, bounds[0][0]) / 3
            S[:, 1] = (min(UPPER, bounds[-1][1]) - max(LOWER, bounds[-1][0])) / np.sqrt(nI)
        else:
            raise ValueError('adaptiveStep algorithm must be "personal" or "global"')

        return X, S

    @staticmethod
    def select_normalized(f: np.ndarray,
                          size: int,
                          minimization: bool = True,
                          bottleNeck: int = 3) -> np.ndarray:
        """
        Selects indices from an array based on normalized values.
        :param f: The input array to be normalized.
        :param size: The number of indices to select.
        :param minimization: If True, selects indices corresponding to minimum values, otherwise maximum.
        :param bottleNeck: If greater than 0, selects indices from a subset of size bottleNeck.
        :return: A numpy array of selected indices (size indices).
        """
        norm_f = (f - np.min(f)) / (np.max(f) - np.min(f) + 1e-8)
        g = 1 - norm_f if minimization else norm_f
        p = g / np.sum(g)

        if bottleNeck == 0:
            indices = np.random.choice(len(f), p=p, size=size)
        else:
            indices = np.zeros(size, dtype=int)
            for i in range(size):
                index_candidates = np.random.choice(len(f), p=p, size=bottleNeck)
                indices[i] = index_candidates[np.argmin(f[index_candidates])] if minimization \
                    else index_candidates[np.argmax(f[index_candidates])]

        return indices

    @staticmethod
    def mutation_normal(X: np.ndarray, bounds: List[Tuple[float, float]], sigma: np.ndarray, Nc: float,
                        SIGMA_MIN: float = 1e-3, SIGMA_MAX: float = 10**6) -> Tuple[np.ndarray, np.ndarray]:
        """
         Performs normal mutation on an array.
        :param X: The input array to be mutated.
        :param bounds: A list of tuples defining the lower and upper bounds for each element in X.
        :param sigma: The adaptive step size for each element in X.
        :param Nc: A constant used in the sigma update.
        :param SIGMA_MIN: The minimum allowed value for sigma.
        :param SIGMA_MAX: The maximum allowed value for sigma.
        :return: A tuple containing the mutated array and the updated adaptive step size (sigma) values.
        """
        n = len(X)
        if n == 0:
            return np.array([]), np.array([])

        bounds = np.array(bounds)

        tau, tau1 = 1 / np.sqrt(2*n), 1 / np.sqrt(2 * np.sqrt(n))

        sigma_limit = np.minimum(bounds[:, 1], SIGMA_MAX)
        sigma *= np.exp(tau * Nc + tau1 * np.random.randn())
        sigma = np.clip(sigma, SIGMA_MIN, np.abs(sigma_limit*0.5))

        '''
        The scipy.stats.truncnorm function expects the lower and upper bounds in standard deviation units. 
        This means that the values passed to it should represent how many standard deviations away from the mean 
        the bounds are.
        '''
        lower_bound = (bounds[:, 0] - X) / sigma
        upper_bound = (bounds[:, 1] - X) / sigma
        X += sigma * truncnorm.rvs(lower_bound, upper_bound, size=n)
        return X, sigma

    @staticmethod
    def mutation_geometric(X: np.ndarray, bounds: List[Tuple[float, float]], sigma: np.ndarray, Nc: float,
                           SIGMA_MIN: float = 1, SIGMA_MAX: float = 10**6) -> Tuple[np.ndarray, np.ndarray]:

        n = len(X)
        if n == 0:
            return np.array([]), np.array([])
        bounds = np.array(bounds)

        tau, tau2 = 1 / np.sqrt(2 * n), 1 / np.sqrt(2 * np.sqrt(n))
        s_over_n = sigma / n
        p = 1 - (s_over_n / (np.sqrt(1 + pow(s_over_n, 2)) + 1))
        pGeo = np.log(1 - p)

        sigma_limit = np.minimum(bounds[:, 0], SIGMA_MAX)
        sigma *= np.exp(tau * Nc + tau2 * np.random.randn())
        sigma = np.clip(sigma, SIGMA_MIN, np.abs(sigma_limit*0.5))

        RV = lambda prob: np.floor(np.log(1 - np.random.uniform(0, 1)) / prob).astype(int)
        for i in range(n):
            lower, upper = bounds[i]
            value = X[i] + (RV(pGeo[i]) - RV(pGeo[i]))
            while value < lower or value > upper:
                value = X[i] + (RV(pGeo[i]) - RV(pGeo[i]))
            X[i] = value
        return X, sigma

    @staticmethod
    def recombine(A: np.array, B: np.array, sigmaA, sigmaB, pRecomb: float = 0.5):
        for i in range(len(A)):
            if np.random.rand() < pRecomb:
                A[i], B[i] = B[i], A[i]
            sigmaA[i] = (sigmaA[i] + sigmaB[i]) / 2
            sigmaB[i] = sigmaA[i]
        return A, B, sigmaA, sigmaB

    @staticmethod
    def step(cur_step: float, n: int, Nc: float, UB: float, SIGMA_MIN: float, sigmaUB: float):
        if n == 0:
            return 0
        tau = 1 / np.sqrt(2*n)
        tau1 = 1 / np.sqrt(2 * np.sqrt(n))
        sigma = max(SIGMA_MIN, cur_step * np.exp(tau * Nc + tau1 * np.random.randn()))
        ub = min(UB, sigmaUB)
        return min(sigma, np.abs(0.5*ub))


# Hessian matrices for Objective Functions in ObjectiveFunction.py
class Hessian:
    @staticmethod
    def genHSphere(N, c=1):
        return c * np.eye(N)

    @staticmethod
    def genHEllipse(N, c):
        H = np.zeros((N, N))
        for i in range(N):
            H[i, i] = c ** (i / (N - 1))
        return H

    @staticmethod
    def genHEllipse2(N, c):
        H = np.zeros((N, N))
        for i in range(N):
            H[i, i] = 1 + (i * (c - 1) / (N - 1))
        return H

    @staticmethod
    def genHadamardHEllipse(N, c):
        H = Hessian.genHEllipse2(N, c)
        R = hadamard(N) / np.sqrt(N)
        H = R.dot(H).dot(np.transpose(R))
        return H

    @staticmethod
    def genHCigar(N, c):
        H = c * np.eye(N)
        H[0][0] = 1.0
        return H

    @staticmethod
    def genHDiscus(N, c):
        H = np.eye(N)
        H[0][0] = c
        return H

    @staticmethod
    def genRotatedHEllipse(N, c, theta=.25 * np.pi):
        H = Hessian.genHEllipse(N, c)
        R = getRotation(N, theta)
        H = R.dot(H).dot(np.transpose(R))
        return H


if __name__ == '__main__':
    # help(Population.generate_random)
    help(Population.select_normalized)
    # help(Population.mutation_normal)
    # print(Population.random_population(5, 3, 3, [(-10, 0), (1, 10), (11, 20), (-20, -11), (30, 39), (-40, -30)]))
    # print(Population.select_normalized(np.array([5, 7, -1, 3, -1]), 4, True, 2))
