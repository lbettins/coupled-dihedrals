# -*- coding: utf-8 -*-
import sys
import math
import numpy as np
from numpy import exp
from copy import deepcopy
from arkane.ess import QChemLog
import scipy.integrate as integrate

import rmgpy.constants as constants

class MCHOJob(object):
    """
    The class to calculate PG properties, including E S G Cp
    """
    def __init__(self, nmode, T):
        if not nmode.is_tors():
            raise ValueError("Mode is not torsional!")
        self.nmode = nmode
        self.q, self.V = nmode.spline(1000)
        #self.wells = nmode.get_wells()
        self.T = T
    
    def calcMCHO(self):
        T = self.T
        beta = 1/(constants.kB*T) #in J/particle
        Q_mcho = 0
        I = self.nmode.I * constants.amu*(10**(-20))  # kg m^2
        for i,well in enumerate(self.nmode.get_wells()):
            # MCHO and PG needs this
            domain,pes_fn = well.get_pes_fn()
            print(domain,i)
            xe = well.get_xe()
            wi = np.sqrt(pes_fn(xe,2)*constants.E_h/I)
            zpe_i = constants.hbar*wi/2.0
            Qtop = np.exp(-beta*(well.get_rel_u()*constants.E_h+zpe_i)) #in J/particle
            Qbot = 1-np.exp(-beta*(constants.hbar*wi))
            Q_mcho += Qtop/Qbot
            print("Q_mcho so far after calc of this well:",Q_mcho)
        print("MCHO Partition function:",Q_mcho)
        return Q_mcho

if __name__ == '__main__':
    csv_path = '../examples/propane_UMVT/samping_result.csv'
    freq_file = '../examples/propane_UMVT/propane.q.out'
    from main import APE
    ape = APE(freq_file)
    ape.parse()
    conformer = ape.conformer
    mode_dict, energy_dict = from_sampling_result(csv_path)
    polynomial_dict = cubic_spline_interpolations(energy_dict,mode_dict)
    thermo = ThermoJob(conformer, polynomial_dict, mode_dict, energy_dict,T=298.15,P=100000)
    thermo.calcThermo(print_HOhf_result=True, zpe_of_Hohf=ape.zpe)
