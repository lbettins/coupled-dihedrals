from inspect import signature
from scipy.integrate import quad,nquad
import numpy as np
import rmgpy.constants as constants

def boltzmann_distribution(E, T):
    beta = 1/constants.kB/T * constants.E_h
    return np.exp(-beta*E)

def classical_partition_fn(E, T, allspace=(-np.inf, np.inf)):
    # What is the dimensionality of the integral?
    # Determined by the bounds of integration
    if isinstance(allspace[0], int) or isinstance(allspace[0], float):
        Z = quad(lambda x: boltzmann_distribution(E(x), T), *allspace)
    else: # type(allspace[0]) is list/numpy.ndarray/tuple
        Z = nquad(lambda *x: boltzmann_distribution(E(x),T), allspace)
    return Z[0]

def boltzmann_factor(E1, E2, T):
    DeltaE = E2 - E1
    return boltzmann_distribution(DeltaE, T)
