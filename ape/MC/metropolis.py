from ape.MC import probability
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random

import rmgpy.constants as constants
from scipy.integrate import quad

class Metropolis:
    """
    Classical Hamiltonian for torsions, comprising relevant quantities
    and methods.
    """

    def __init__(self, E=None, Xi=0, T=300, sigma=1, period=None, n_it=10**3):
        """
        Setup metropolis algorithm
        --------------------------
        X = position vector (sampling variable)
        Xi = initial position
        E = function that takes in X (and only X) and returns energy / Hartree
        T = temperature / K
        sigma = standard deviation for normal distribution sampling
        period = period, if X is periodic
        n_it = number of Metropolis-Hastings iterations
        """
        if E is None:
            raise ValueError("Specify an energy function f = f(X).")
        # Vector quantities (potentially)
        self.X = np.array(Xi)
        self.sigma = np.array(sigma)
        self.period = np.array(period)
        # Scalar quantities
        self.E = E
        self.T = T
        self.n_it = n_it
        self.result = []
        
    def step(self):
        Xtrial = np.random.normal(self.X, self.sigma)
        prob_accept = np.random.random()
        if self.period:
            Xtrial = roll(Xtrial, period=self.period)
        if probability.boltzmann_factor(self.E(self.X), self.E(Xtrial), 
                self.T) > prob_accept:
            self.X = Xtrial
            self.result.append(Xtrial)
        self.n_it -= 1

    def run(self):
        """
        Metropolis algorithm returning an array of positions at which
        a particle is found throughout the simulation. Accepts an argument
        corresponding to a guess of initial position, and optional arguments
        corresponding to walk size and number of iterations.
        """
        while self.n_it > 0:
            self.step()

    def plot2d(self):
        self.result = np.array(self.result)
        if self.period:
            x = (np.linspace(-np.pi,np.pi), np.linspace(-np.pi,np.pi))
        X,Y = np.meshgrid(x[0],x[1])
        bounds = ((0,2*np.pi),(0,2*np.pi))
        Z = probability.classical_partition_fn(self.E, self.T,allspace=bounds)
        fn = probability.boltzmann_distribution(self.E((X,Y)), self.T)
        fnx = probability.boltzmann_distribution(self.E((x[0],0)), self.T)
        fny = probability.boltzmann_distribution(self.E((0,x[1])), self.T)
        Zx = np.trapz(fnx, x[0])
        Zy = np.trapz(fny, x[1])

        # PLOTTING
        from matplotlib.ticker import NullFormatter
        import matplotlib
        matplotlib.use('MacOSX')
        fig = plt.figure(3)
        nullfmt = NullFormatter()         # no labels
        # definitions for the axes
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        bottom_h = left_h = left + width + 0.02
        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom_h, width, 0.2]
        rect_histy = [left_h, bottom, 0.2, height]
        axScatter = plt.axes(rect_scatter)
        axHistx = plt.axes(rect_histx)
        axHisty = plt.axes(rect_histy)
        # no labels
        axHistx.xaxis.set_major_formatter(nullfmt)
        axHisty.yaxis.set_major_formatter(nullfmt)
        # the scatter plot:
        axScatter.scatter(self.result[:,0], self.result[:,1], alpha=0.3, marker='.')
        axScatter.contour(X,Y,fn/Z,cmap="coolwarm")
        # now determine nice limits by hand:
        xymax = np.max([np.max(np.fabs(self.result[:,0])), 
            np.max(np.fabs(self.result[:,1]))])
        axScatter.set_xlim((-np.pi, np.pi))
        axScatter.set_ylim((-np.pi, np.pi))
        bins = 20
        axHistx.hist(self.result[:,0], bins=bins, density=True)
        axHistx.plot(x[0], fnx/Zx)
        axHisty.hist(self.result[:,1], bins=bins, orientation='horizontal',density=True)
        axHisty.plot(fny/Zy,x[1])
        axHistx.set_xlim(axScatter.get_xlim())
        axHisty.set_ylim(axScatter.get_ylim())
        print(self.result[:,0],self.result[:,1])
        plt.show()

def roll(x, period=2*np.pi):
    """
    Recursively get the correct domain for periodic x
    """
    return x % period
    if (x >= 0.).all() and (x < period).all():
        return x
    elif (x >= period).any():
        return roll(x - (period % x))

def periodic_run():
    """
    Runs metropolis algorithm. Plots result for given energy funciton
    """
    def E(x):
        return np.sin(x) + 1

    mcjob = Metropolis(E=E, period=2*np.pi)
    mcjob.run()
    x = np.linspace(0, 2*np.pi, 1000)
    Z = probability.classical_partition_fn(E, mcjob.T, (0,2*np.pi)) 
    fn = probability.boltzmann_distribution(E(x), mcjob.T)
    fig = plt.figure(1)
    plt.plot(x, fn/Z)
    plt.hist(mcjob.result, bins=20, density=True)
    integral = np.trapz(fn,x)
    total = np.trapz(fn/Z,x)

def regular_run(): 
    k = 0.3
    T = 500
    beta = 1/constants.kB/T * constants.E_h
    def E_HO(x, k=k):
        return 0.5*k*x**2

    mcjob = Metropolis(E=E_HO, T=T, sigma=np.sqrt(1/k/beta))
    mcjob.run()
    x = np.linspace(-0.4, 0.4, 1000)
    Z = probability.classical_partition_fn(E_HO, mcjob.T)
    fn = probability.boltzmann_distribution(E_HO(x), mcjob.T)
    fig = plt.figure(2)
    plt.plot(x, fn/Z)
    plt.hist(mcjob.result, bins=20, density=True)

def multidimensional_run():
    kx, ky = 0.2, 1.0
    T = 500
    beta = 1/constants.kB/T * constants.E_h
    def E_HO(x, k):
        return 0.5*k*x**2
    # Function should accept ONLY X
    # x,y,z should be unpacked within function
    def E_2D(X):
        x,y = X[0],X[1]
        return E_HO(x,kx) + E_HO(y,ky)

    sigmas = (np.sqrt(1/kx/beta), np.sqrt(1/ky/beta))
    mcjob = Metropolis(E=E_2D, Xi=(0,0), T=T, sigma=sigmas, n_it=10**4)
    mcjob.run()
    x = (np.linspace(-0.4, 0.4, 1000), np.linspace(-0.4, 0.4, 1000))
    X,Y = np.meshgrid(x[0],x[1])
    bounds = ((-np.inf,np.inf),(-np.inf,np.inf))
    Z = probability.classical_partition_fn(E_2D, mcjob.T,allspace=bounds)
    fn = probability.boltzmann_distribution(E_2D((X,Y)), mcjob.T)
    fnx = probability.boltzmann_distribution(E_2D((x[0],0)), mcjob.T)
    Zx = np.trapz(fnx,x[0])
    fny = probability.boltzmann_distribution(E_2D((0,x[1])), mcjob.T)
    Zy = np.trapz(fny,x[1])
    results = np.array(mcjob.result)

    # PLOTTING
    from matplotlib.ticker import NullFormatter
    fig = plt.figure(3)
    nullfmt = NullFormatter()         # no labels
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]
    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)
    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)
    # the scatter plot:
    axScatter.scatter(results[:,0], results[:,1], alpha=0.3, marker='.')
    axScatter.contour(X,Y,fn/Z,cmap="coolwarm")
    # now determine nice limits by hand:
    xymax = np.max([np.max(np.fabs(results[:,0])), 
        np.max(np.fabs(results[:,1]))])
    axScatter.set_xlim((-0.25, 0.25))
    axScatter.set_ylim((-0.15, 0.15))
    bins = 20
    axHistx.hist(results[:,0], bins=bins, density=True)
    axHistx.plot(x[0], fnx/Zx)
    axHisty.hist(results[:,1], bins=bins, orientation='horizontal',density=True)
    axHisty.plot(fny/Zy,x[1])
    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    #fig = plt.figure(4)
    #plt.hexbin(results[:,0],results[:,1], cmap="coolwarm")


if __name__ == "__main__":
    periodic_run()
    regular_run()
    multidimensional_run()
    plt.show()
