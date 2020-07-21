# -*- coding: utf-8 -*- 
# [1] https://doi.org/10.1021/acs.jctc.5b01177 Thermodynamics of Anharmonic Systems: Uncoupled Mode Approximations for Molecules

"""
APE's main module.
To run APE through its API, first, first get a freq output file from Qchem, then call the .execute. For example:

    ape = APE(input_file)
    ape.execute()

"""

import os
import csv
import numpy as np
import subprocess
import copy

import rmgpy.constants as constants

from arkane.common import symbol_by_number
from arkane.statmech import StatMechJob, determine_rotor_symmetry, is_linear
from arkane.ess import QChemLog

from arc.species.species import ARCSpecies

from ape.job import Job, record_script
from ape.torsion import HinderedRotor
from ape.InternalCoordinates import get_RedundantCoords, getXYZ
from ape.FitPES import cubic_spline_interpolations
from ape.calcThermo import ThermoJob
from ape.calcPG import PGJob
from ape.schro import HO_psi,psi
from ape.NMode import NMode, Well
from ape.sampling import torsional_sampling, vibrational_sampling
from ape.coupled_sampling import coupled_torsional_sampling

import matplotlib
from ape.basic_units import radians, degrees
from scipy import integrate

class APE(object):
    """
    The main APE class.
    """

    def __init__(self,input_file, name=None, project_directory=None, 
            protocol=None, multiplicity=None, charge = None,
            external_symmetry=None, level_of_theory=None, 
            basis=None, ncpus=None, 
            which_modes = [], imaginary_bonds=None, just_write=False):
        self.input_file = input_file
        self.name = name
        self.project_directory = project_directory if project_directory is not None\
                else os.path.abspath(os.path.dirname(input_file))
        self.protocol = protocol
        self.multiplicity = multiplicity
        self.charge = charge
        self.external_symmetry = external_symmetry
        self.level_of_theory = level_of_theory
        self.basis = basis
        self.ncpus = ncpus
        self.which_modes = which_modes  # What modes to sample
        self.imaginary_bonds = imaginary_bonds  # For UMVT
        self.just_write = just_write

    def parse(self):
        Log = QChemLog(self.input_file)
        self.log = Log
        self.is_qmmm = Log.is_qmmm()
        if self.is_qmmm:
            try:
                self.qmmm_template = QChemLog(self.input_file[:-3]+'sp.out')
            except:
                print('Template file for QMMM does not exist!')
                print('Include template file as a SP output file in the same directory.')
        else:
            self.qmmm_template = None
        # In Shih Cheng Li's implementation, Hessian is not partial?
        # For QMMM, geometry is for QM region only
        self.hessian = Log.load_force_constant_matrix()
        coordinates, number, mass = Log.load_geometry()
        self.symbols = [symbol_by_number[i] for i in number]
        self.conformer, unscaled_frequencies = Log.load_conformer()
        self.cart_coords = coordinates.reshape(-1,)
        if self.name is None:
            self.name = self.input_file.split('/')[-1].split('.')[0]
        self.natom = Log.get_number_of_atoms()  # Total number of atoms
        #self.natom = len(Log.load_geometry()[0])   # number qm atoms
        self.xyz = getXYZ(self.symbols, self.cart_coords)   # qm geom
        log_charge, log_mult, log_bas, log_lvl = Log.get_vars()
        if not self.level_of_theory:  
            self.level_of_theory = log_lvl
        if not self.basis:
            self.basis = log_bas
        if not self.charge:
            self.charge = log_charge
        if not self.multiplicity:
            self.multiplicity = log_mult ##)
        try:
            self.frags = Log.get_fragment_geometries()
        except KeyError:
            self.frags = None
            if self.protocol == 'UMN':
                pass
            elif self.is_qmmm:
                print("Must define fragments for QMMM UMVT job")
                raise
            else:
                print("Fragments undefined, problems if adsorbed complex!")
                pass
        if not self.frags and (self.protocol == 'UMVT' 
                or self.protocol == 'CMT'):
            self.ARCSpecies = ARCSpecies(label=self.name,xyz=self.xyz,
                    multiplicity=self.multiplicity,charge=self.charge)
        elif not self.frags:     # If UMN, ARCSpec not required
            self.ARCSpecies = None
        else:
            for frag in self.frags.keys():
                if not isinstance(frag,int):
                    continue
                self.ARCSpecies = self.frags[frag]['species']    # ARCSpecies set to first fragment encountered
                break

        self.conformer.coordinates = (coordinates, "angstroms")
        self.conformer.number = number
        self.conformer.mass = (mass, "amu")
        self.linearity = is_linear(self.conformer.coordinates.value)
        self.e_elect = Log.load_energy()
        self.zpe = Log.load_zero_point_energy()
        #e0 = self.e_elect + self.zpe
        #self.conformer.E0 = (e0, "J/mol")
        if self.ncpus is None:
            self.ncpus = 2

        # Determine hindered rotors information
        if self.protocol == 'UMVT' or self.protocol == 'CMT':
            self.rotors_dict = self.get_rotors_dict()
            self.n_rotors = len(self.rotors_dict)
        else:
            self.rotors_dict = []
            self.n_rotors = 0
        
        # Look at arkane.ess for more info / for zeolitic systems, = 3*N_QMATOMS
        self.nmode = len(Log.load_mode_freqs()) 
        self.n_vib = self.nmode - self.n_rotors
        # If which modes to run are undefined, run all of them by default
        if not self.which_modes[0]:
            self.which_modes = list(range(1,self.nmode+1))

        ## STUDY THIS
        # Create RedundantCoords object
        self.internal = get_RedundantCoords(self.symbols, self.cart_coords, 
                self.rotors_dict, self.imaginary_bonds)
    
    def get_rotors_dict(self):
        rotors_dict = {}
        species = self.ARCSpecies
        species.determine_rotors()
        for i in species.rotors_dict:
            rotors_dict[i+1] = {}
            pivots = species.rotors_dict[i]['pivots']
            top = species.rotors_dict[i]['top']
            scan = species.rotors_dict[i]['scan']
            rotors_dict[i+1]['pivots'] = pivots 
            rotors_dict[i+1]['top'] = top
            rotors_dict[i+1]['scan'] = scan
        return rotors_dict

    def sampling(self, thresh=0.05, save_result=True, scan_res=10):
        xyz_dict = {}
        energy_dict = {}
        mode_dict = {}
        if not os.path.exists(self.project_directory):
            os.makedirs(self.project_directory)
        if not self.just_write:
            path = os.path.join(self.project_directory,'output_file')
        else:
            path = os.path.join(self.project_directory,'files_to_run')
        if not os.path.exists(path):
            os.makedirs(path)
        if self.protocol == 'UMVT' or self.protocol == 'CMT' and self.n_rotors != 0:
            n_vib = self.n_vib
            if self.frags and self.is_qmmm: # Added is QMMM..
                # NOTE THIS MIGHT BE FALSE, INVESTIGATE
                # NOTE: changed what happens when self.is_qmmm is used
                n_vib -= 6   # due to frustrated trans and rot
            rotor = HinderedRotor(self.symbols, self.cart_coords, 
                    self.hessian, self.rotors_dict, 
                    self.conformer.mass.value_si,
                    n_vib, self.imaginary_bonds)
            projected_hessian = rotor.projectd_hessian()
            vib_freq, unweighted_v = SolvEig(projected_hessian, 
                    self.conformer.mass.value_si, self.n_vib)
            print('Frequencies(cm-1) from projected Hessian:',vib_freq)

            tors_modes = [] 
            for i in range(self.n_rotors):
                mode = i+1
                tors_modes.append(mode) 
                if self.protocol == 'UMVT':
                    if i+1 not in self.which_modes:
                        continue
                    modal_xyzdict, modal_edict, modal_mdict =\
                            torsional_sampling(self, mode, rotor,
                                    path, scan_res, thresh, 
                                    just_write=self.just_write)
                    xyz_dict[mode] = modal_xyzdict
                    energy_dict[mode] = modal_edict
                    mode_dict[mode] = modal_mdict
            if self.protocol == 'CMT':
                scan_res = 36 
                modal_xyzdict, modal_edict, modal_mdict=\
                        coupled_torsional_sampling(self, tors_modes,
                                rotor, path, scan_res, thresh,
                                just_write=self.just_write)
                for mode in tors_modes:
                    xyz_dict[mode] = modal_xyzdict
                    energy_dict[mode] = modal_edict
                    mode_dict[mode] = modal_mdict
                return xyz_dict, energy_dict, mode_dict

        elif self.protocol == 'UMN':
            hessian = self.hessian
            mass = self.conformer.mass.value_si
            vib_freq, unweighted_v = SolvEig(hessian, mass, self.n_vib)
            print('Vibrational frequencies of normal modes: ',vib_freq)

        for i in range(self.nmode):
            mode = i+1
            if i in range(self.n_rotors):
                continue
            if mode not in self.which_modes: 
                continue
            vector = unweighted_v[i-self.n_rotors]
            freq = vib_freq[i-self.n_rotors]
            magnitude = np.linalg.norm(vector)
            reduced_mass = magnitude ** -2 / constants.amu  # in amu
            step_size = np.sqrt(constants.hbar / (reduced_mass * constants.amu)\
                    / (freq * 2 * np.pi * constants.c * 100)) * 10**10  # in angst
            normalized_vector = vector/magnitude
            qj = np.matmul(self.internal.B, normalized_vector)
            P = np.ones(self.internal.B.shape[0],dtype=int)
            n_rotors = len(self.rotors_dict)
            if n_rotors != 0:
                P[-n_rotors:] = 0
            P = np.diag(P)
            qj = P.dot(qj).reshape(-1,)

            modal_xyzdict, modal_edict, modal_mdict =\
                    vibrational_sampling(self, mode,
                            qj, freq, reduced_mass, step_size,
                            path, thresh, just_write=self.just_write)
            xyz_dict[mode] = modal_xyzdict
            energy_dict[mode] = modal_edict
            mode_dict[mode] = modal_mdict

        if save_result and not self.just_write:
            path = self.project_directory
            self.write_samping_result_to_csv_file(path, mode_dict, energy_dict)

            path = os.path.join(path, 'plot')
            if not os.path.exists(path):
                os.makedirs(path)
            self.write_sampling_displaced_geometries(path, 
                    mode_dict, energy_dict, xyz_dict)

        return xyz_dict, energy_dict, mode_dict

    def write_samping_result_to_csv_file(self, path, mode_dict, energy_dict):
        csv_path = os.path.join(path, 'sampling_result.csv')
        with open(csv_path, 'w') as f:
            writer = csv.writer(f)
            for mode in mode_dict.keys():
                if not mode in self.which_modes:
                    continue
                if mode_dict[mode]['mode'] == 'tors': 
                    name = 'mode_{}_tors'.format(mode)
                    is_tors = True
                else: 
                    name = 'mode_{}_vib'.format(mode)
                    is_tors = False
                writer.writerow([name])
                if is_tors:
                    writer.writerow(['symmetry_number', mode_dict[mode]['symmetry_number']])
                writer.writerow(['M', mode_dict[mode]['M']])
                writer.writerow(['K', mode_dict[mode]['K']])
                writer.writerow(['step_size', mode_dict[mode]['step_size']])
                writer.writerow(['sample', 'total energy(HARTREE)'])
                for sample in sorted(energy_dict[mode].keys()):
                    writer.writerow([sample, energy_dict[mode][sample]])
            f.close()
            #print('Have saved the sampling result in {path}'.format(path=csv_path))
    
    def write_sampling_displaced_geometries(self, path, 
            mode_dict, energy_dict, xyz_dict):
        #creat a format can be read by VMD software
        for mode in energy_dict.keys():
            if not mode in self.which_modes:
                continue
            if mode_dict[mode]['mode'] == 'tors':
                # If is a torsion
                txt_path = os.path.join(path, 'mode_{}_tors.txt'.format(mode))
            else:
                # If is a stretch/bend
                txt_path = os.path.join(path, 'mode_{}_sb.txt'.format(mode))
            with open(txt_path, 'w') as f:
                for sample in sorted(energy_dict[mode].keys()): 
                    content = record_script.format(natom=self.natom, sample=sample, e_elect=energy_dict[mode][sample], xyz=xyz_dict[mode][sample])
                    f.write(content)
                f.close()

    def execute(self):
        """
        Execute APE.
        """
        self.parse()
        xyz_dict, energy_dict, mode_dict = self.sampling()
        if self.just_write:
            return
        #Solve SE of 1-D PES and calculate E S G Cp
        polynomial_dict = cubic_spline_interpolations(energy_dict,mode_dict)
        modes = dicts_to_NModes(mode_dict, energy_dict, xyz_dict, ape_obj=self)
        thermo = ThermoJob(self, polynomial_dict, mode_dict, energy_dict, xyz_dict, T=298.15, P=100000, nmodes=modes)
        thermo.calcThermo(print_HOhf_result=True, zpe_of_Hohf=self.zpe)


        #### PLOTTING TAKES PLACE BELOW ####
        import matplotlib.pyplot as plt
        matplotlib.use('MacOSX')
        E0 = 0
        for mode in modes:
            q,V = mode.spline(1000)
            E0 += mode.zpe_ho
            #qmins_true,Vmins_true,geom_mins_true = mode.get_true_mins(self)

            # GET EIGENVALUES FOR DIRECT EIGENVAL SUMMATION VISUALIZATION
            eig,U = mode.get_eigs()
            #############################################################
            if mode.is_tors():
                fig, axs = plt.subplots(2)
                axs[1].plot(q,V,'--',xunits=radians)
                for i,well in enumerate(mode.get_wells()):
                    #qwell,Vwell = well.get_pes()
                    domain,well_pes = well.get_pes_fn()
                    qwell = np.linspace(domain[0],domain[-1],1000)
                    Vwell = well_pes(qwell)
                    if i == 0:
                        xlim_left = qwell[0]
                    axs[1].plot(qwell,Vwell,'-',xunits=radians)
                    well_nmode = well.get_well_nmode()
                    q_e = well.get_xe()
                    #k_well = well_nmode.get_k()
                    k_well = mode.get_k()
                    #mu_well = well_nmode.get_mu()
                    mu_well = mode.get_I()
                    K_well = mu_well*k_well*constants.amu*(10**(-20))/constants.E_h
                    K_well = well_pes(q_e,2)
                    axs[1].plot([val*radians for val in qwell], well.get_rel_u()+0.5*K_well*(qwell-q_e)**2,'--',xunits=radians)
                    #well_nmode = well.get_well_nmode()
                    #well_nmode.spline(1000)
                    theta_ = np.linspace(-np.pi,3*np.pi,1000)
                    #V_ = well_nmode.pes
                    #axs[2].plot(qwell,Vwell,'-',xunits=radians)
                    #axs[2].plot([val*radians for val in theta_],list(map(V_,theta_-q_e)),'--',xunits=radians) 
                qmins,Vmins,geom_mins = mode.get_sample_mins()
                #plt.scatter(qmins,Vmins,facecolors='none',edgecolors='k',xunits=radians)
                #plt.scatter(qmins_true*radians,Vmins_true,facecolors='none',edgecolors='r',xunits=radians)
                axs[1].set_xlim(xlim_left,qwell[-1])
                margin = (max(V)-min(V))/10
                axs[1].set_ylim(min(V)-margin,max(V)+margin)
                axs[1].set_xlabel(r'$\theta$')
                axs[1].set_ylabel(r'$V(\theta)$' + ' / Hartree')
                #axs[2].set_xlim(xlim_left,qwell[-1])
                margin = (max(V)-min(V))/10
                #axs[2].set_ylim(min(V)-margin,max(V)+margin)
                #axs[2].set_xlabel(r'$\theta$')
                #axs[2].set_ylabel(r'$V(\theta)$' + ' / Hartree')
                theta = np.linspace(-2*np.pi,2*np.pi,1000)
                k_tors = mode.get_k()
                I_tors = mode.get_I()
                K_tors = I_tors*k_tors*constants.amu*(10**(-20))/constants.E_h
                V_tors = mode.get_spline_fn()
                K_tors = V_tors(0,2)
                axs[0].plot([val*radians for val in theta],V_tors(theta),'-k',xunits=radians)
                axs[0].plot([val*radians for val in theta],0.5*K_tors*(theta)**2,'-.',xunits=radians)
                axs[0].set_xlim(-2*np.pi,2*np.pi)
                axs[0].set_ylim(min(V)-margin,max(V)+margin)
                axs[0].set_xlabel(r'$\theta$')
                axs[0].set_ylabel(r'$V(\theta)$' + ' / Hartree')
                fig.tight_layout()

                plt.figure(3)
                plt.plot([val*radians for val in theta],V_tors(theta),'-k',xunits=radians)
                x = mode.get_x_sample()
                y = mode.get_v_sample()
                plt.scatter(x,y,facecolors='none',edgecolors='red')
                plt.xlim(0, 2*np.pi)
                plt.xlabel(r"$\theta$")
                plt.ylabel("E / Hartree")
                
                #### VISUALIZATION OF ENERGY LEVELS FOR TORSION ####
                fig3,(tors3,harm3) = plt.subplots(1,2)
                ####################################################
                theta = np.linspace(-2*np.pi,2*np.pi,1000)
                theta1 = [val*radians for val in theta]
                I = mode.get_I()    # amu * angstrom**2
                k = mode.get_k()    # 1/s^2
                w = np.sqrt(k)
                K = V_tors(0,2)      #Second derivative of origin
                tol = 1E-6
                tors3.plot(theta1,0.5*K*theta**2,'--b',xunits=radians)
                tors3.plot(theta1,V_tors(theta),'-k',xunits=radians)
                harm3.plot(theta1,0.5*K*theta**2,'-b',xunits=radians)
                tors3.set_ylim(min(V)-margin, max(V)+margin)
                harm3.set_ylim(min(V)-margin, max(V)+margin)
                tors3.set_xlim(theta[0],theta[-1])
                harm3.set_xlim(theta[0],theta[-1])
                tors3.set_ylabel("E / Hartree")
                tors3.set_xlabel(r"$\theta$")
                harm3.set_xlabel(r"$\theta$")
                for ax in fig3.get_axes():
                    ax.label_outer()
                for n in range(0,len(eig)):
                    v = U[:,n]
                    Psi = 0.0
                    c2 = 0
                    integral = 0
                    for i,c in enumerate(v):
                        Psi += c*psi(i,theta,I)
                        c2 += c*c
                        #integral += integrate.quad(lambda x:
                        #        c*c*psi(i,x,I)*psi(i,x,I),0,2*np.pi)[0]
                    #print(c2)
                    #print(integral)
                    hbar1 = constants.hbar / constants.E_h
                    hbar2 = constants.hbar * 10 ** 20 / constants.amu
                    en = hbar1*w*(n+0.5)
                    print(eig[n],en)
                    tors3.plot(theta1,eig[n]+Psi**2/4000)
                    a = np.sqrt(I*w/hbar2)
                    harm3.plot(theta1,en+(HO_psi(n,theta,a))**2/4000)

                ### VISUALIZATION OF BOLTZMANN DISTRIBUTION ###
                plt.figure(5)
                ###############################################
                sigma = mode.get_symmetry_number() 
                theta = np.linspace(-np.pi,np.pi,1000)
                theta1 = [val*radians for val in theta]
                T = 300 #K 
                beta = 1/(constants.kB*T) * constants.E_h
                Icopy = I*constants.amu*(10**(-20))  # kg m^2
                prefactor = np.sqrt(Icopy*constants.kB*T/(2*np.pi*constants.hbar**2))
                distribution = np.exp(-beta*V_tors(theta))
                norm_c = integrate.quad(lambda x: 
                        prefactor*np.exp(-beta*V_tors(x)),
                        0, 2*np.pi/sigma)[0]
                plt.plot(theta1,distribution,'k',xunits=radians)
                plt.plot(theta1,prefactor*distribution,'--b')
                plt.plot(theta1,prefactor*distribution/norm_c,'-.r')
                plt.xlim(-np.pi/sigma,np.pi/sigma)
                plt.show()

            else:
                plt.figure(2)
                #plt.plot(q,V,'-')
                V_interp = mode.get_spline_fn()
                #A = 1.0
                #q = np.linspace(A*q[0],-A*q[0],1000)
                #plt.plot(q,V_interp(q),'-.')

                V = mode.pes
                plt.plot(q,[val for val in list(map(V,q))],'-k')

                xs = mode.get_x_sample()
                Vs = mode.get_v_sample()
                plt.scatter(xs,Vs,facecolors="none",edgecolors='r')
                plt.xlabel("Displacement / Angstrom")
                plt.ylabel("E / Hartree")

                                
                ### VISUALIZATION OF ANHARMONIC ENERGY LEVELS ###
                fig2,(ax1,ax2) = plt.subplots(1,2)
                #################################################
                hbar1 = constants.hbar / constants.E_h
                hbar2 = constants.hbar * 10 ** 20 / constants.amu
                mu = mode.get_mu()
                k = mode.get_k()
                w = np.sqrt(k)
                a = np.sqrt(mu*w/hbar2)
                K = mu*k*constants.amu*(10**(-20))/constants.E_h
                tol = 1E-6
                xlim = max(np.abs([q[0],q[-1]]))
                qHO = np.linspace(-xlim,xlim,1000)
                ax1.plot(qHO,0.5*K*qHO**2,'--b')
                ax1.plot(q,V_interp(q),'-k')
                ax2.plot(qHO,0.5*K*qHO**2,'-b')
                ax1.set_xlim(-xlim,xlim)
                ax1.set_ylim(0,max([V_interp(q[0]), V_interp(q[-1])]))
                ax2.set_xlim(-xlim,xlim)
                ax1.set_xlim(q[0],q[-1])
                ax2.set_ylim(0,max([V_interp(q[0]), V_interp(q[-1])]))
                ax1.set_ylabel("E / Hartree")
                ax1.set_xlabel("x / Angstrom")
                ax2.set_xlabel("x / Angstrom")
                for ax in fig2.get_axes():
                    ax.label_outer()
                q = np.linspace(-10,10,10000)
                for n in range(0,len(eig)):
                    v = U[:,n]
                    HOpsi = 0.0
                    c2 = 0
                    integral = 0
                    for i,c in enumerate(v):
                        HOpsi += c*HO_psi(i,q,a)
                        #c2 += c*c
                        #integral += integrate.quad(lambda x:
                        #        c*c*HO_psi(i,x,a)*HO_psi(i,x,a),-np.inf,np.inf)[0]
                    #print(c2)
                    #print(integral)
                    HOpsi *= HOpsi/1000
                    if n < 50:
                        for i,val in enumerate(HOpsi):
                            if not abs(val) > tol:
                                continue
                            leftindex = i
                            break
                        for i,val in enumerate(reversed(HOpsi)):
                            if not abs(val) > tol:
                                continue
                            rightindex = -i-1
                            break
                        qplot = q[leftindex:rightindex]
                        HOpsi = HOpsi[leftindex:rightindex]
                        ax1.plot(qplot,eig[n]+HOpsi,'-')
                        en = hbar1*w*(n+0.5)
                        print(eig[n],en)
                        HO = HO_psi(n,q,a)
                        #print(integrate.quad(lambda x: HO_psi(n,x,a)*HO_psi(n,x,a),-np.inf,np.inf)[0])
                        HOplot = HO_psi(n,qHO,a)**2/1000
                        for i,val in enumerate(HOplot):
                            if not abs(val) > tol:
                                continue
                            leftindex = i
                            break
                        for i,val in enumerate(reversed(HOplot)):
                            if not abs(val) > tol:
                                continue
                            rightindex = -i-1
                            break
                        qplot = qHO[leftindex:rightindex]
                        HOplot = HOplot[leftindex:rightindex]
                        ax2.plot(qplot,en+HOplot,'-')

        print("Total HO ZPE:",E0)
        print("Total mass:",self.conformer.mass.value_si)
        plt.show()
        

###################################################################################
def SolvEig(hessian, mass, n_vib):
    # Generate mass-weighted force constant matrix
    mass_3N_array = np.array([i for i in mass for j in range(3)])
    mass_mat = np.diag(mass_3N_array)
    inv_sq_mass_mat = np.linalg.inv(mass_mat**0.5)
    mass_weighted_hessian = inv_sq_mass_mat.dot(hessian).dot(inv_sq_mass_mat)
    eig, v = np.linalg.eigh(mass_weighted_hessian)
    vib_freq = np.sqrt(eig[-n_vib:]) / (2 * np.pi * constants.c * 100) # in cm^-1
    unweighted_v = np.matmul(inv_sq_mass_mat,v).T[-n_vib:]
    return vib_freq, unweighted_v

def dict_to_NMode(mode, m_dict, e_dict, xyz_dict, 
        rotors_dict=[], ape_obj=None):
    try:
       m_dict[mode]['v_um']
    except KeyError:
        m_dict[mode]['v_um'] = 0
        m_dict[mode]['zpe_um'] = 0
        pass

    samples = sorted(e_dict[mode].keys())
    step = m_dict[mode]['step_size']
    xvals = [sample*step for sample in samples] #in rads
    vvals = [e_dict[mode][sample] for sample in samples] #in Hartree
    if xyz_dict:
        geoms = [xyz_dict[mode][sample] for sample in samples]
    else:
        geoms = None
    try:
        v_ho = m_dict[mode]['v_ho']
        v_um = m_dict[mode]['v_um']
        zpe_um = m_dict[mode]['zpe_um']
        min_elect = m_dict[mode]['min_elect']
    except KeyError:
        v_ho = 0
        v_um = 0
        zpe_um = 0
        min_elect = 0
    is_tors = (True if m_dict[mode]['mode'] == 'tors' else False)
    if is_tors:
        I = m_dict[mode]['M']   # amu * angstrom^2
        k = m_dict[mode]['K']   # 1/s^2
        sigma = m_dict[mode]['symmetry_number']
        ##### FOR NOW #####
        if ape_obj.frags:
            sigma = 1
        ##################
        try:
            scan = rotors_dict[mode]['scan']
            print("Scan is",scan)
        except IndexError:
            scan = 0
            pass
        new_mode = NMode(n=mode,v=vvals,x=xvals,sample_geoms=geoms,
                tors=is_tors,scan=scan,
                v_ho=v_ho,v_um=v_um,zpe_um=zpe_um,
                I=I,k=k,min_elec=min_elect,sigma=sigma)
    else:
        mu = m_dict[mode]['M']  # amu
        k = m_dict[mode]['K']   # 1/s^2
        new_mode = NMode(
                n=mode,v=vvals,x=xvals,sample_geoms=geoms,
                tors=is_tors,v_ho=v_ho,v_um=v_um,zpe_um=zpe_um,
                mu=mu,k=k,min_elec=min_elect)
    return new_mode

def dicts_to_NModes(m_dict, e_dict, xyz_dict, 
        rotors_dict=[], ape_obj=None):
    # Get array of NMode types for easy use in PG, MCHO
    modes = []
    for mode in sorted(m_dict.keys()):
        modes.append(dict_to_NMode(mode,m_dict,e_dict,xyz_dict,
            rotors_dict,ape_obj))
    return modes
