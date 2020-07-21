# -*- coding: utf-8 -*-
import numpy as np
from scipy import integrate
from scipy.special import eval_hermite
from math import sqrt
from math import factorial as fact

import rmgpy.constants as constants

from ape.FitPES import from_sampling_result
#from ape.HarmonicBasis import IntXHmHnexp
#from ape.FourierBasis import IntXPhimPhin

def H(n,z):
    """
    Hermite polynomials H_n(x)
    """
    return eval_hermite(n,z)

def HO_psi(n,x,a):
    """
    Harmonic oscillator basis function
    x [=] Angstrom
    a [=] 1/Angstrom
    """
    prefactor = sqrt(a/sqrt(np.pi)/(2**n * fact(n)))
    fn = np.exp(-(a*x)**2/2) * H(n,a*x)
    return prefactor * fn

def psi(n,theta,I):
    L = np.pi*np.sqrt(I)
    q = np.sqrt(I)*theta
    if n == 0:
        return 1.0/np.sqrt(2*L)
    elif n % 2 == 1:
        kn = (n+1)/2*np.pi/L
        return 1.0/np.sqrt(L) * np.cos(kn*q)
    else:
        kn = n/2*np.pi/L
        return 1.0/np.sqrt(L) * np.sin(kn*q)

def Hmn(m, n, nmode):
    hbar1 = constants.hbar / constants.E_h # in hartree*s
    hbar2 = constants.hbar * 10 ** 20 / constants.amu # in amu*angstrom^2/s

    result = 0
    k = nmode.get_k()   # 1/s^2
    if nmode.is_tors():
        # use fourier basis function
        I = nmode.get_I() # in amu*angstrom^2
        step_size = nmode.get_step_size() 
        delta_q = sqrt(I) * step_size # in sqrt(amu)*angstrom
        L = np.pi * sqrt(I) # in sqrt(amu)*angstrom

    else:
        # use harmonic basis functions
        mu = nmode.get_mu()  # in amu
        w = sqrt(k)         # in 1/s
        a = sqrt(mu*w/hbar2) # in 1/angstrom
        var1,var2 = nmode.spline(10)
        V = nmode.pes
        K = mu*k*constants.amu*(10**(-20))/constants.E_h
        result += integrate.quad(lambda x: HO_psi(m,x,a)*(V(x)-.5*K*x**2)*HO_psi(n,x,a),-np.inf,np.inf)[0] 
        if m==n:
            result += hbar1*w*(m+0.5)
    return result

def ImprovedH(nmode, size, N_prev, H_prev):
    H = np.zeros((size, size), np.float64)
    for m in range(size):
        for n in range(m+1):
            if m < N_prev and n < N_prev:
                Hmn_val = H_prev[m][n]
            else:
                Hmn_val = Hmn(m, n, nmode)
            H[m][n] = Hmn_val
            H[n][m] = Hmn_val
    return H
