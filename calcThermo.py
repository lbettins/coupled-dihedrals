# -*- coding: utf-8 -*-
import sys
import math
import numpy as np
from numpy import exp
from copy import deepcopy

import rmgpy.constants as constants

from ape.calcPG import PGJob
from ape.calcMCHO import MCHOJob
from ape.FitPES import from_sampling_result, cubic_spline_interpolations
from ape.schrodinger import SetAnharmonicH #, SetAnharmonicH2
from ape.schro import ImprovedH

class ThermoJob(object):
    """
    The class to calculate thermodynamic properties, including E S G Cp
    """
    def __init__(self, ape_obj, polynomial_dict, mode_dict, energy_dict, xyz_dict, T,P=100000, nmodes=None):
        self.ape_obj = ape_obj
        try:
            self.conformer = ape_obj.conformer
        except ValueError:
            pass
        if nmodes is not None:
            self.NModes = nmodes
        else:
            self.NModes = dicts_to_NModes(mode_dict, energy_dict, xyz_dict)
        self.polynomial_dict = polynomial_dict
        self.mode_dict = mode_dict
        self.energy_dict = energy_dict
        self.T = T
        self.P = P
        from ape.main import dicts_to_NModes
    
    def calcThermoOfEachMode(self,eig,N,mode):
        T = self.T
        beta = 1/(constants.kB*T) * constants.E_h
        Q = 0
        E = 0
        dQ = 0
        ddQ = 0
        for i in range(N):
            Ei = eig[i]
            Q += exp(-beta*Ei)
            dQ += Ei*exp(-beta*Ei)*beta/T
            ddQ += -2*Ei*exp(-beta*Ei)*beta/pow(T,2) + pow(Ei,2)*exp(-beta*Ei)*pow(beta,2)/pow(T,2)
            E += Ei*exp(-beta*Ei)
        E /= Q
        is_tors = True if self.mode_dict[mode]['mode'] == 'tors' else False
        if is_tors:
            omega = self.mode_dict[mode]['symmetry_number']
            Q /= omega
            dQ /= omega
            ddQ /= omega

        E0 = eig[0]
        v = (eig[1]-eig[0]) * constants.E_h / constants.h / (constants.c * 100) 
        #print(Q)

        F = -math.log(Q)/beta
        S = (E - F)/T
        Cv = (2/Q*dQ - T*pow(dQ/Q,2) + T/Q*ddQ)/beta

        return v, E0, E, S, F, Q, Cv

    def SolvEig(self,mode,nmode):
        for nm in self.NModes:
            if nm.get_mode_number() != mode:
                continue
            nmode = nm
            break
        Nbasis = 50
        Nbasis_prev = 0
        H_prev = None
        Qold =  np.log(sys.float_info[0])
        vold =  np.log(sys.float_info[0])
        converge = False
        while not converge:
            Nbasis += 1
            H = SetAnharmonicH(self.polynomial_dict, self.mode_dict, self.energy_dict, mode, Nbasis, N_prev=Nbasis_prev, H_prev=H_prev)
            Nbasis_prev = Nbasis
            H_prev = deepcopy(H)
            eig, U =np.linalg.eigh(H)

            #H_improved = ImprovedH(nmode, Nbasis, N_prev=Nbasis_prev, H_prev=H_prev)
            #Nbasis_prev = Nbasis
            #H_prev = deepcopy(H_improved)
            #eig, v =np.linalg.eigh(H_improved)

            v, E0, E, S, F, Q, Cv = self.calcThermoOfEachMode(eig,Nbasis,mode)
            #print(v,E0,Q)

            if Qold == np.log(sys.float_info[0]):
                print("\n\t %d \t\t-\t\t-" % Nbasis) #first run
            else:
                print("\n\t %d \t\t %.10f \t\t %.10f" % (Nbasis,abs(Q-Qold),abs(v-vold)))
            
            if ((abs(Q-Qold)<1e-4) and (abs(v-vold)<1e-2)):
                print(" Convergence criterion met\n","------------------------------------")
                converge = True
                nmode.set_H(H)
                print("Frequency (cm-1): %.10f" % v)
                print("Zero point vibrational energy (hartree): %.10f" % E0)
                print("Energy (hartree): %.10f" % E )
                print("Entropy (hartree/K): %.10f" % S)
                print("Free energy (hartree): %.10f" % F)
                print("Partition function: %.10f" % Q)
                hartree2kcalmol = constants.E_h * constants.Na / 4184
                E0 *= hartree2kcalmol
                E *= hartree2kcalmol
                S *= hartree2kcalmol * 1000
                F *= hartree2kcalmol
                Cv *= hartree2kcalmol * 1000
                '''
                print("Frequency (cm-1): ",v)
                print("Zero point vibrational energy (kcal/mol): ",E0)
                print("Energy (kcal/mol): ",E )
                print("Entropy (cal/mol/K): ",S)
                print("Free energy (kcal/mol): ",F)
                print("Partition function: ",Q)
                '''
            
            Qold = Q
            vold = v
        return v, E0, E, S, F, Q, Cv

    def calcThermo(self, print_HOhf_result=False, zpe_of_Hohf=None):
        T = self.T
        P = self.P
        conformer = self.conformer
        print("Calculate global translation and rotation E, S")
        E_trans = 1.5 * constants.R * T / 4184
        S_trans = conformer.modes[0].get_entropy(T) / 4.184 - constants.R * math.log(P/101325) / 4.184 
        Cv_trans = 1.5 * constants.R / 4184 * 1000  
        #F = conformer.modes[0].get_free_energy(T) / 4.184 
        #Q = conformer.modes[0].get_partition_function

        E_rot = conformer.modes[1].get_enthalpy(T) / 4184
        S_rot = conformer.modes[1].get_entropy(T) / 4.184
        Cv_rot = conformer.modes[1].get_heat_capacity(T) / 4.184
        #F = conformer.modes[1].get_free_energy(T) / 4.184
        #Q = conformer.modes[1].get_partition_function

        print("Calculate internal E, S")
        E0 = 0
        E_int = 0
        S_int = 0
        #F_int = 0
        Q_int = 1
        Cv_int = 0

        E0_pg = 0
        E_int_pg = 0
        S_int_pg = 0
        #F_int_pg = 0
        Q_int_pg = 1
        Cv_int_pg = 0

        E0_mcho = 0
        E_int_mcho = 0
        S_int_mcho = 0
        #F_int_mcho = 0
        Q_int_mcho = 1
        Cv_int_mcho = 0

        E0_mcumn = 0
        E_int_mcumn = 0
        S_int_mcumn = 0
        #F_int_mcumn = 0
        #Q_int_mcumn = 1
        Cv_int_mcumn = 0

        for mode,nmode in zip(sorted(self.mode_dict.keys()), self.NModes):
            ################################################################
            ##### THIS SECTION IS FOR UMN AND UMVT THERMODYNAMICS ONLY #####
            ################################################################
            print("\n\t********** Mode ",mode," **********\n\n")
            v, e0, E, S, F, Q, Cv = self.SolvEig(mode,nmode)
            self.mode_dict[mode]['v_um'] = v    #cm^-1
            self.mode_dict[mode]['zpe_um'] = e0 #kcal/mol
            # Store UMN/UMVT v and zpe in NMode for later use
            nmode.set_um_freq_zpe(v, e0)
            E0 += e0
            E_int += E
            S_int += S
            #F_int += F
            Q_int *= Q
            Cv_int += Cv

            q,V = nmode.spline(1000)
            qmins_true,Vmins_true,geom_mins_true = nmode.get_true_mins(self.ape_obj)
            ################################################################
            ###### THIS SECTION IS FOR MC-UMN THERMO OF TORSIONS ONLY ######
            ################################################################ 
            #print("\n\t********** Mode ",nmode.get_mode_number()," **********\n\n")
            #if nmode.is_tors():
            #    mcumnThermo = MCUMNJob(nmode,self.T,self.P)
            #    e0_, E_, S_, F_, Q_, Cv_ = mcumnThermo.calcMCUMN()
            #    E0_mcumn += e0_
            #    E_int_mcumn += E_
            #    S_int_mcumn += S_
            #    #F_int_mcumn += F_
            #    #Q_int_mcumn *= Q_
            #    Cv_int_mcumn += Cv_
            #else:
            #    E0_mcumn += e0
            #    E_int_mcumn += E
            #    S_int_mcumn += S
            #    #F_int_mcumn += F
            #    #Q_int_mcumn *= Q
            #    Cv_int_mcumn += Cv

            ################################################################
            ######## THIS SECTION IS FOR PG THERMO OF TORSIONS ONLY ########
            ################################################################ 
            print("\n\t********** Mode ",nmode.get_mode_number()," **********\n\n")
            if nmode.is_tors():
                pgThermo = PGJob(nmode,T)
                e0_, E_, S_, F_, Q_, Cv_ = pgThermo.calcPG()
                E0_pg += e0_
                E_int_pg += E_
                S_int_pg += S_
                #F_int_pg += F_
                Q_int_pg *= Q_
                Cv_int_pg += Cv_
            else:
                E0_pg += e0
                E_int_pg += E
                S_int_pg += S
                #F_int_pg += F
                Q_int_pg *= Q
                Cv_int_pg += Cv


            ################################################################
            ####### THIS SECTION IS FOR MCHO THERMO OF TORSIONS ONLY #######
            ################################################################ 
            print("\n\t********** Mode ",nmode.get_mode_number()," **********\n\n")
            if nmode.is_tors():
                mchoThermo = MCHOJob(nmode,self.T)
                Q_int_mcho *= mchoThermo.calcMCHO()
           #     e0_, E_, S_, F_, Q_, Cv_ = mchoThermo.calcMCHO()
           #     E0_mcho += e0_
           #     E_int_mcho += E_
           #     S_int_mcho += S_
           #     #F_int_mcho += F_
           #     #Q_int_mcho *= Q_
           #     Cv_int_mcho += Cv_
           # else:
           #     E0_mcho += e0
           #     E_int_mcho += E
           #     S_int_mcho += S
           #     #F_int_mcho += F
           #     #Q_int_mcho *= Q
           #     Cv_int_mcho += Cv


            print("UMVT:",Q_int,"\nPG:",Q_int_pg,"\nMCHO:",Q_int_mcho)
            
        print("\n\t********** Final results **********\n\n")
        print("Temperature (K): ",T)
        print("Pressure (Pa): ",P)
        print("Zero point vibrational energy (kcal/mol): %.10f" % (E0))
        print("Translational energy (kcal/mol): %.10f" % (E_trans))
        print("Translational entropy (cal/mol/K): %.10f" % (S_trans))
        print("Translational Cv (cal/mol/K): %.10f" % (Cv_trans))
        print("Rotational energy (kcal/mol): %.10f" % (E_rot))
        print("Rotational entropy (cal/mol/K): %.10f" % (S_rot))
        print("Rotational Cv (cal/mol/K): %.10f" % (Cv_rot))
        print("Internal (rot+vib) energy (kcal/mol): %.10f" % (E_int))
        print("Internal (tor+vib) entropy (cal/mol/K): %.10f" % (S_int))
        print("Internal (tor+vib) Cv (cal/mol/K): %.10f" % (Cv_int))
        print("\n")
        print("Total energy (kcal/mol): %.10f" % (E_trans+E_rot+E_int))
        print("Total enthalpy (kcal/mol): %.10f" % (E_trans+E_rot+E_int+constants.kB*T*constants.Na/4184))
        print("Enthalpy H(%f K)-H(0 K) (kcal/mol):  %.10f" % (T, E_trans+E_rot+E_int+constants.kB*T*constants.Na/4184-E0))
        print("Total entropy (cal/mol/K): %.10f" % (S_trans+S_rot+S_int))
        print("Total Cv (cal/mol/K): %.10f" % (Cv_trans+Cv_rot+Cv_int))

    
        print("\n\t********** PG results **********\n\n")
        print("Temperature (K): ",T)
        print("Pressure (Pa): ",P)
        print("Zero point vibrational energy (kcal/mol): %.10f" % (E0_pg))
        #print("Translational energy (kcal/mol): %.10f" % (E_trans))
        #print("Translational entropy (cal/mol/K): %.10f" % (S_trans))
        #print("Translational Cv (cal/mol/K): %.10f" % (Cv_trans))
        #print("Rotational energy (kcal/mol): %.10f" % (E_rot))
        #print("Rotational entropy (cal/mol/K): %.10f" % (S_rot))
        #print("Rotational Cv (cal/mol/K): %.10f" % (Cv_rot))
        print("Internal (rot+vib) energy (kcal/mol): %.10f" % (E_int_pg))
        print("Internal (tor+vib) entropy (cal/mol/K): %.10f" % (S_int_pg))
        print("Internal (tor+vib) Cv (cal/mol/K): %.10f" % (Cv_int_pg))
        print("\n")
        print("Total energy (kcal/mol): %.10f" % (E_trans+E_rot+E_int_pg))
        print("Total enthalpy (kcal/mol): %.10f" % (E_trans+E_rot+E_int_pg+constants.kB*T*constants.Na/4184))
        print("Enthalpy H(%f K)-H(0 K) (kcal/mol):  %.10f" % (T, E_trans+E_rot+E_int_pg+constants.kB*T*constants.Na/4184-E0_pg))
        print("Total entropy (cal/mol/K): %.10f" % (S_trans+S_rot+S_int_pg))
        print("Total Cv (cal/mol/K): %.10f" % (Cv_trans+Cv_rot+Cv_int_pg))

        if print_HOhf_result:
            # compare to HOhf model
            E_vib = (conformer.modes[2].get_enthalpy(T) + zpe_of_Hohf) / 4184
            # E_vib should be calculated by freq...
            S_vib = conformer.modes[2].get_entropy(T) / 4.184
            print("\n")
            print("\n\t********** HOhf results **********\n\n")
            print("Translational energy (kcal/mol): %.10f" % (E_trans))
            print("Rotational energy (kcal/mol): %.10f" % (E_rot))
            print("Vibrational energy (kcal/mol): %.10f" % (E_vib))
            print("gas constant (RT): %.10f" % (constants.R * T / 4184))
            print("Translational entropy (cal/mol/K): %.10f" % (S_trans))
            print("Rotational entropy (cal/mol/K): %.10f" % (S_rot))
            print("Vibrational entropy (cal/mol/K): %.10f" % (S_vib))
            print("\n")
            print("Total energy (kcal/mol): %.10f" % (E_trans+E_rot+E_vib))
            print("Total enthalpy (kcal/mol): %.10f" % (E_trans+E_rot+E_vib+constants.R*T/4184))
            print("Enthalpy H(%f K)-H(0 K) (kcal/mol): %.10f" % (T, conformer.get_enthalpy(T)/4184))
            print("Total entropy (cal/mol/K): %.10f" % (S_trans+S_rot+S_vib))
            print("Total Cv (cal/mol/K): %.10f" % (conformer.get_heat_capacity(T) / 4.184))

################################################################################

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
