# -*- coding: utf-8 -*- 

"""
Visualization of torsional coupling energies.
The inclusion of x_ and S_ as global variables excludes
the possibility of visualizing torsional probabilities.

See another module for this functionality.
"""

import numpy as np
from scipy import integrate
from scipy.interpolate import Rbf, griddata

import rmgpy.constants as constants

from ape.qchem import QChemLog
from ape.basic_units import radians, degrees
from ape.plotting.periodicize import augment_with_periodic_bc

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# GLOBAL VARS
x_ = None
S_ = None
P_ = None

def plot(x,V,T=298,pressure=100000):
    n_dihedrals = len(x)
    if n_dihedrals > 2:
        print("Plots only generated for 2D coupling")
        return
    import matplotlib
    matplotlib.use('MacOSX')

    fig1 = plt.figure(1)
    plot_samples(x,V,fig1,nperiods=2)

    fig2 = plt.figure(2)
    plot_sample_probs(x,V,fig2,T,nperiods=2)

    fig3 = plt.figure(3)
    plot_surface(x,V,fig3)

    fig4 = plt.figure(4)
    plot_projected_contours(x,V,fig4)

    fig5 = plt.figure(5)
    plot_projected_contours_prob(x,V,fig5,T)

    fig6 = plt.figure(6)
    plot_2d_contour(x,V,fig6)

    fig7 = plt.figure(7)
    plot_2d_contour_prob(x,V,fig7,T)

    plt.show()

def plot_sample_probs(x,z,fig,T, nperiods=1):
    """
    Plot the (proportional) probability density assuming a
    Boltzmann distribution of coupled torsional states.
    """
    beta = 1/(constants.kB*T) * constants.E_h
    p = np.exp(-beta*z)
    plot_samples(x,p,fig,color='r',alpha=0.3,
            stype='probability',nperiods=nperiods)

def plot_samples(x,z,fig,color='b',alpha=0.3,
        surface=True, stype='potential', nperiods=1):
    """
    Just plot the samples (on a surface).
    """
    ax = fig.gca(projection='3d')
    X,Y = np.meshgrid(*x)
    ax.scatter3D(X,Y,z,color=color,alpha=alpha,
            xunits=radians,yunits=radians)
    if surface:
        add_surface(x,z,ax,stype=stype)

def plot_surface(x,z,fig,stype='potential'):
    """
    Plot the surface without superimposing sample points.
    """
    ax = fig.gca(projection='3d')
    add_surface(x,z,ax,stype=stype)

def add_surface(x,z,ax,stype='potential'):
    xi,fn = generate_surface(x,z,stype=stype)
    Xi = (np.meshgrid(*xi))
    ax.plot_surface(*Xi,fn(*Xi),alpha=0.4,cmap=cm.coolwarm)
    ax.set_xlim(-np.pi,np.pi)
    ax.set_ylim(-np.pi,np.pi)

def generate_surface(x, z, nperiods=2, stype='potential'):
    """
    Use bivariate splines to generate 3D surface 
    for coupled internal rotors.
    """
    global x_
    global S_
    global P_
    # Saves time rather than re-interpolating every plot
    if x_ is None:
        xi = yi = np.linspace(-np.pi, np.pi,200)
        x_ = (xi,yi)
    if S_ is None and stype == 'potential': 
        # Augmented domains are used to force periodic b.c.
        x,z = augment_with_periodic_bc(x, z, 2*np.pi)
        X,Y = np.meshgrid(*x)
        rbfi = Rbf(X, Y, z,function='cubic')  # radial basis function interpolator instance
        #di = rbfi(xi, yi)   # interpolated values, shape=1000,
        #rbfi = griddata((X,Y),z,(*np.meshgrid(*x_)),method='cubic')
        S_ = rbfi
    elif P_ is None and stype == 'probability':
        # Augmented domains are used to force periodic b.c.
        x,z = augment_with_periodic_bc(x, z, 2*np.pi)
        X,Y = np.meshgrid(*x)
        rbfi = Rbf(X, Y, z,function='cubic')  # radial basis function interpolator instance
        rbfi = Rbf(X,Y,z)
        P_ = rbfi
    if stype == 'potential':
        return x_,S_
    elif stype == 'probability':
        return x_,P_

def project_contours(x,z,ax,exclude_z=True,stype='potential'):
    """
    Project *x and z onto respective axes as contours.
    ---------------------------------------------------
    Plot projections of the contours for each dimension.  By choosing offsets
    that match the appropriate axes limits, the projected contours will sit on
    the 'walls' of the graph
    """
    xi,fn = generate_surface(x, z, stype=stype)
    X,Y = np.meshgrid(*xi)
    Z = fn(X,Y)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    margin = (np.max(Z)-np.min(Z)) / 10.
    ax.set_zlim(-np.min(Z)-margin, np.max(Z)+margin)

    if not exclude_z:
        cset = ax.contourf(X, Y, Z, zdir='z', offset=-np.min(Z)-margin,
                cmap=cm.coolwarm)
    cset = ax.contour(X, Y, Z, zdir='x', offset=-np.pi, cmap=plt.get_cmap('twilight'))
    cset = ax.contour(X, Y, Z, zdir='y', offset=np.pi, cmap=plt.get_cmap('twilight'))
        
    ax.set_xlabel(r'$\phi')
    ax.set_ylabel(r'$\psi')
    ax.set_zlabel('z')

def plot_projected_contours_prob(x,z,fig,T):
    """
    Plot the projected contours for proportional Boltzmann probability.
    """
    beta = 1/(constants.kB*T) * constants.E_h
    p = np.exp(-beta*z)
    plot_projected_contours(x,p,fig,stype='probability')

def plot_projected_contours(x,z,fig,stype='potential'):
    """
    Plot the projected contours without superimposing surface or sample points
    """
    ax = fig.gca(projection='3d')
    project_contours(x,z,ax,exclude_z=False,stype=stype)

def add_2d_contour(x,z,ax,axis='z',stype='potential'):
    xi,fn = generate_surface(x,z,stype=stype)
    Xi = (np.meshgrid(*xi))
    ax.contourf(*Xi,fn(*Xi),zdir=axis,cmap=cm.coolwarm)
    ax.set_xlim(-np.pi,np.pi)
    ax.set_ylim(-np.pi,np.pi)

def plot_2d_contour(x,z,fig,stype='potential'):
    ax = plt.gca()
    add_2d_contour(x,z,ax,stype=stype)

def plot_2d_contour_prob(x,z,fig,T):
    beta = 1/(constants.kB*T) * constants.E_h
    p = np.exp(-beta*z)
    plot_2d_contour(x,p,fig,stype='probability')

if __name__ == '__main__':
    x = np.array([np.linspace(-np.pi, np.pi), np.linspace(0,2*np.pi)])
    z = np.zeros(shape=(len(x[0]), len(x[1])))
    for i,xi in enumerate(x[0]):
        for j,xj in enumerate(x[1]):
            z[i,j] += np.sin(xi*xj)*np.cos(xi*xj)
    plot(x,z)
