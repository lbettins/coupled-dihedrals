# -*- coding: utf-8 -*-
import sys
import math
import numpy as np
from numpy import exp
from copy import deepcopy
from arkane.ess import QChemLog
import scipy.integrate as integrate

import rmgpy.constants as constants

from ape.calcThermo import ThermoJob
from ape.FitPES import cubic_spline_interpolations

class MCUMNJob(object):
    """
    The class to calculate MC-UM-N properties, including E S G Cp
    This calculates the properties for ONE SPECIFIC TORSIONAL MODE that has been
    previously sampled and refined by reoptimization.
    """
    def __init__(self, nmode, T, P=100000):
        if not nmode.is_tors():
            raise ValueError("Mode is not torsional, MC-UM-N cannot be run!")
        self.nmode = nmode
        self.wells = nmode.get_wells()
        self.T = T 
        self.P = P 
    
        def calcThermo(self):
            T = self.T
            P = self.P
            beta = 1/(constants.kB*T) * constants.E_h
            E0_mcumn = 0
            E_int_mcumn = 0
            S_int_mcumn = 0
            #F_int_mcumn = 0
            #Q_int_mcumn = 1
            Cv_int_mcumn = 0
            for well in self.wells:
                print("\n\t********** Well ",well.get_well_n()," **********\n\n")
                m_dict,e_dict,xyz_dict = well.get_UM_dicts()
                poly_dict = cubic_spline_interpolations(e_dict,m_dict)
                well_thermo = ThermoJob(polynomial_dict=poly_dict,mode_dict=m_dict,
                    energy_dict=e_dict,xyz_dict=xyz_dict,T=T,P=P)
                v, e0, E, S, F, Q, Cv = well_thermo.SolvEig(well.get_well_n())
                well.set_well_um_freq_zpe(v, e0)
                E0_mcumn += e0
                E_int_mcumn += E
                S_int_mcumn += S
                #F_int_mcumn += F
                #Q_int_mcumn *= Q
                Cv_int_mcumn += Cv
            return E0_mcumn, E_int_mcumn, S_int_mcumn,\
                    F_int_mcumn, Q_int_mcumn, Cv_int_mcumn

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
