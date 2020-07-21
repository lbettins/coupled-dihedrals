# -*- coding: utf-8 -*-
import sys
import math
import numpy as np
from numpy import exp
from copy import deepcopy
import scipy.integrate as integrate

import rmgpy.constants as constants

class PGJob(object):
    """
    The class to calculate PG properties, including E S G Cp
    """
    def __init__(self, nmode, T):
        self.nmode = nmode
        if not nmode.is_tors():
            raise ValueError("Mode is not torsional!")
        self.q, self.V = nmode.spline(1000)
        self.I = nmode.get_I()
        self.k = nmode.get_k()
        #self.wells = nmode.get_wells()
        self.T = T
    
    def calcPG(self):
        T = self.T
        beta = 1/(constants.kB*T) #in 1/(J/particle)
        I = self.I * constants.amu*(10**(-20))  # kg m^2
        k = self.k
        Ktors = I*k         # kg m^2 / s^2 AKA J
        print("Ktors is",Ktors)
        print("I is",I)
        prefactor = np.sqrt(I*constants.kB*T/(2*np.pi*constants.hbar**2))
        print("Prefactor is,",prefactor)
        integral = 0
        Q = 0
        dQ = 0
        Qclass = 0
        sigma = self.nmode.get_symmetry_number()
        for i,well in enumerate(self.nmode.get_wells()):
            domain,pes_fn = well.get_pes_fn()
            print(domain,i)
            xe = well.get_xe()
            ### Double check this ###
            #print("ddV of well",i,"is",pes_fn(xe,2)*constants.E_h)
            #print("wi of well",i,"is",np.sqrt(pes_fn(xe,2)*constants.E_h/I))
            ### Double check this ###
            wi = np.sqrt(pes_fn(xe,2)*constants.E_h/I)
            zpe_i = constants.hbar*wi/2.0
            intgrnd = lambda x: np.exp(-beta*pes_fn(x)*constants.E_h)
            if not sigma == 1:
                intgrl = integrate.quad(intgrnd,0,2*np.pi/sigma)[0]
            else:
                intgrl = integrate.quad(intgrnd,domain[0],domain[-1])[0]
            #print("Integral is",intgrl)
            Q_ho_i = np.exp(-beta*zpe_i)/(1-np.exp(-beta*constants.hbar*wi))
            #print("Q_ho is",Q_ho_i,i)
            Q_cho_i = 1/(constants.hbar*wi*beta)
            #print("Q_cho is",Q_cho_i,i)
            Q += Q_ho_i/Q_cho_i * prefactor * intgrl
            Qclass += prefactor * intgrl

            dQ_ho_i = -zpe_i*np.exp(zpe_i*beta)*\
                    (np.exp(constants.hbar*wi*beta)+1)/\
                    (np.exp(constants.hbar*wi*beta)-1)**2
            dQ_cho_i_recip = constants.hbar*wi
            #print("dQ_ho =", dQ_ho_i)
            #print("d(1/Q_cho) =", dQ_cho_i_recip)
            dprefactor = -1/(2*beta)*prefactor
            #print("dprefactor =", dprefactor)
            dintgrl = -integrate.quad(lambda x: pes_fn(x)*constants.E_h*np.exp(-beta*pes_fn(x)*constants.E_h),
                    domain[0],domain[-1])[0] 
            #print("dintgrl =",dintgrl/constants.E_h)
            # Put dQ back into Hartree:
            dQ += dQ_ho_i  /Q_cho_i      *prefactor *intgrl +\
                    Q_ho_i*dQ_cho_i_recip*prefactor *intgrl +\
                    Q_ho_i /Q_cho_i     *dprefactor *intgrl +\
                    Q_ho_i /Q_cho_i      *prefactor*dintgrl
            print("dQ is",dQ)
            if not sigma == 1:
                break
        #Q = integral
        print("Partition function is",Q)
        e0 = 0
        E = -dQ/Q 
        F = -np.log(Q)/beta
        S = (E-F)/T
        Cv = np.inf

        print("------------------------------------")
        print("Frequency (cm-1): %.10f" % wi)
        print("Zero point vibrational energy (hartree): %.10f" % e0)
        print("Energy (hartree): %.10f" % E )
        print("Entropy (hartree/K): %.10f" % S)
        print("Free energy (hartree): %.10f" % F)
        print("Partition function: %.10f" % Q)
        hartree2kcalmol = constants.E_h * constants.Na / 4184
        e0 *= hartree2kcalmol
        E *= hartree2kcalmol
        S *= hartree2kcalmol * 1000
        F *= hartree2kcalmol
        Cv *= hartree2kcalmol * 1000
        print("Q_tors_class is",Qclass)

        return e0, E, S, F, Q, Cv
        
################################################################################

if __name__ == '__main__':
    csv_path = '../examples/propane_UMVT/samping_result.csv'
    freq_file = '../examples/propane_UMVT/propane.q.out'
    from main import APE
    ape = APE(freq_file)
    ape.parse()
    mode_dict, energy_dict = from_sampling_result(csv_path)
    nmode = dict_to_NMode(mode_dict, energy_dict)
    thermo = PGJob(nmode,T=298.15)
    thermo.calcThermoOfEachWell()
