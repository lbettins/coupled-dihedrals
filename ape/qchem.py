#!/usr/bin/env python3

###############################################################################
#                                                                             #
# RMG - Reaction Mechanism Generator                                          #
#                                                                             #
# Copyright (c) 2002-2019 Prof. William H. Green (whgreen@mit.edu),           #
# Prof. Richard H. West (r.west@neu.edu) and the RMG Team (rmg_dev@mit.edu)   #
#                                                                             #
# Permission is hereby granted, free of charge, to any person obtaining a     #
# copy of this software and associated documentation files (the 'Software'),  #
# to deal in the Software without restriction, including without limitation   #
# the rights to use, copy, modify, merge, publish, distribute, sublicense,    #
# and/or sell copies of the Software, and to permit persons to whom the       #
# Software is furnished to do so, subject to the following conditions:        #
#                                                                             #
# The above copyright notice and this permission notice shall be included in  #
# all copies or substantial portions of the Software.                         #
#                                                                             #
# THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,    #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE #
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER      #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING     #
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER         #
# DEALINGS IN THE SOFTWARE.                                                   #
#                                                                             #
###############################################################################

"""
Arkane QChem module
Used to parse QChem output files
Edited by lbettins
"""

import math
import logging
import os.path

import numpy as np

import rmgpy.constants as constants
from rmgpy.statmech import IdealGasTranslation, NonlinearRotor, LinearRotor, HarmonicOscillator, Conformer

from arkane.common import check_conformer_energy, get_element_mass
from arkane.exceptions import LogError
from arkane.ess.adapter import ESSAdapter
from arkane.ess.factory import register_ess_adapter
#from arkane.statmech import is_linear


################################################################################


class QChemLog(ESSAdapter):
    """
    Represent an output file from QChem. The attribute `path` refers to the
    location on disk of the QChem output file of interest. Methods are provided
    to extract a variety of information into Arkane classes and/or NumPy
    arrays. QChemLog is an adapter for the abstract class ESSAdapter.
    """

    def __init__(self, path):
        self.path = path

    def get_user_input(self,pathname=''):
        if not pathname:
            pathname = self.path
        with open(pathname, 'r') as f:
            line = f.readline()
            string = ''
            while line:
                if 'User input:' in line:
                    toggle = True
                    if string:
                        # Ensures only last user input is read
                        # This is the one of interest (freq output)
                        string = ''
                    line = f.readline()
                    line = f.readline()
                    while toggle:
                        string += line
                        line = f.readline()
                        if '--------------------------------------------------------------' in line:
                            string += '$end'
                            #f.close()
                            #return string
                            toggle = False
                line = f.readline()
            f.close()
            if string:
                return string
            raise ValueError("User input could not be retrieved")

    def rewrite_user_input(self,pathname=''):
        if not pathname:
            pathname = self.path
        user_input = self.get_user_input()
        filename = pathname[:-3]+'in'
        with open(filename,'w') as f:
            f.write(user_input)
        f.close()

    def is_qmmm(self, user_input=''):
        if not user_input:
            user_input = self.get_user_input()
        for line in user_input.splitlines():
            if 'qmmm' in line.lower():
                return True
        return False

    def get_range_qm_atoms(self, user_input=''):
        if not self.is_qmmm(user_input):
            return range(0,self.get_number_of_atoms())
        if not user_input:
            user_input = self.get_user_input()
        return_toggle = 0
        for line in user_input.splitlines():
            if return_toggle:
                split = line.partition(':')
                first = int(split[0])-1
                last = int(split[-1])
                return range(first,last)
            if '$qm_atoms' in line:
                return_toggle = 1
                continue
        return False

    def get_number_of_qm_atoms(self, user_input=''):
        if not user_input:
            user_input = self.get_user_input()
        count = 0
        for x in self.get_range_qm_atoms(user_input):
            count += 1
        return count

    def get_isotopes(self, user_input=''):
        if not user_input:
            user_input = self.get_user_input()
        isotopes = {}
        setting_toggle = False
        total_n_toggle = False
        isotope_toggle = False
        for line in user_input.splitlines():
            #if setting_toggle or total_n_toggle or isotope_toggle:
            #    print(line)
            if setting_toggle:
                isotopes['setting'] = line.split()
                setting_toggle = False
                total_n_toggle = True
            elif total_n_toggle:
                isotopes['n_iso'] = int(line)
                isotopes['range'] = range(0, int(line))
                count = 0
                total_n_toggle = False
                isotope_toggle = True
            elif isotope_toggle and '$end' in line:
                isotope_toggle = False
                break
            elif isotope_toggle:
                index = int(line.split()[0])
                mass = float(line.split()[1])
                count += 1
                isotopes[index] = {}
                isotopes[index]['count'] = count
                if mass > 999:
                    isotopes[index]['mass'] = np.inf
                else:
                    isotopes[index]['mass'] = float(mass)
            elif '$isotope' in line:
                setting_toggle = True
        #for x in isotopes['range']:
        #    print(x)
        #    print(isotopes[x+1]['index'],isotopes[x+1]['mass'])
        return isotopes

    def get_fragment_geometries(self,user_input=''):
        if not user_input:
            user_input = self.get_user_input()  
        fragments = {}
        fragment_number = None
        atom_number = 0     # Initialize atom count
        for i,line in enumerate(user_input.splitlines()):
            if '$molecule' in line.lower():
                fragment_number = -1
                continue
            elif fragment_number == -1:
                fragments['total_charge'] = int(line.split()[0])
                fragments['total_multiplicity'] = int(line.split()[1])
                fragment_number += 1
            else:
                if '--' in line:
                    fragment_number += 1
                    fragments[fragment_number] = {}
                    fragments[fragment_number]['name'] = line.split()[1]
                    fragments[fragment_number]['xyz'] = ''
                    continue
                elif '$end' in line:
                    break
                elif len(line.split()) == 2:
                    fragments[fragment_number]['charge'] = int(line.split()[0])
                    fragments[fragment_number]['multiplicity'] = int(line.split()[1])
                    continue
                else:
                    fragments[fragment_number][atom_number] = {}
                    fragments[fragment_number][atom_number]['element']\
                            = line.split()[0]
                    fragments[fragment_number][atom_number]['coords']\
                            = line.split()[1:4]
                    fragments[fragment_number][atom_number]['connectivity']\
                            = line.split()[4:]
                    fragments[fragment_number]['xyz'] += line.split()[0] \
                            + '\t' + '\t'.join(line.split()[1:4]) + '\n'
                    atom_number += 1
        fragments['N'] = fragment_number

        from arc.species.species import ARCSpecies
        for i in range(1,fragments['N']+1):
            try:
                species = ARCSpecies(label=fragments[i]['name'],
                        xyz=fragments[i]['xyz'],
                        multiplicity=fragments[i]['multiplicity'],
                        charge=fragments[i]['charge'])
            except (KeyError, ValueError):
                species = None
                pass
            fragments[i]['species'] = species
        print(fragments)

        ### ADD ISOTOPE FUNCTIONALITY ###
        return fragments


    def get_number_of_atoms(self):
        """
        Return the number of atoms in the molecular configuration used in
        the QChem output file.
        """
        n_atoms = 0

        with open(self.path, 'r') as f:
            line = f.readline()
            while line != '' and n_atoms == 0:
                # Automatically determine the number of atoms
                if 'Standard Nuclear Orientation' in line and n_atoms == 0:
                    for i in range(3):
                        line = f.readline()
                    while '----------------------------------------------------' not in line:
                        n_atoms += 1
                        line = f.readline()
                line = f.readline()

        return n_atoms



    def get_vars(self):
        spin_multiplicity = 0
        lvl_of_theory = None
        basis = None
        charge = None
        with open(self.path, 'r') as f:
            line = f.readline()
            while line != '':
                # Read spin multiplicity if not explicitly given
                if '$molecule' in line.lower() and spin_multiplicity == 0:
                    line = f.readline()
                    if len(line.split()) == 2:
                        spin_multiplicity = int(float(line.split()[1]))
                        charge = int(float(line.split()[0]))
                elif 'basis' in line.lower():
                    if line.lower().split()[0] == 'basis':
                        basis = line.split()[-1]
                elif 'ecp' in line.lower():
                    if line.lower().split()[0] == 'ecp':
                        #basis += "\necp\tdef2ecp"
                        pass
                elif 'method' in line.lower() or 'exchange' in line.lower() and len(line.split()) == 2:
                    lvl_of_theory = line.split()[-1]
                    if 'omega' in lvl_of_theory.lower():
                        lvl_of_theory = lvl_of_theory.replace('omega','w')
                if charge is not None and spin_multiplicity and basis and lvl_of_theory:
                    return charge,spin_multiplicity,basis,lvl_of_theory
                line = f.readline()
            f.close()
            return charge, spin_multiplicity, basis, lvl_of_theory

    def load_force_constant_matrix(self):
        """
        Return the force constant matrix (in Cartesian coordinates) from the
        QChem log file. If multiple such matrices are identified,
        only the last is returned. The units of the returned force constants
        are J/m^2. If no force constant matrix can be found in the log file,
        ``None`` is returned.
        """
        force = None
        if self.is_qmmm():
            n_atoms = len(self.load_geometry()[0])
        else:
            n_atoms = self.get_number_of_atoms()
        n_rows = n_atoms * 3
        with open(self.path, 'r') as f:
            line = f.readline()
            while line != '':
                # Read force constant matrix
                if 'Final Hessian.' in line or 'Hessian of the SCF Energy' in line:
                    force = np.zeros((n_rows, n_rows), np.float64)
                    for i in range(int(math.ceil(n_rows / 6.0))):
                        # Header row
                        line = f.readline()
                        # Matrix element rows
                        for j in range(n_rows):  # for j in range(i*6, Nrows):
                            data = f.readline().split()
                            for k in range(len(data) - 1):
                                force[j, i * 6 + k] = float(data[k + 1])
                                # F[i*5+k,j] = F[j,i*5+k]
                    # Convert from atomic units (Hartree/Bohr_radius^2) to J/m^2
                    force *= 4.35974417e-18 / 5.291772108e-11 ** 2
                line = f.readline()
        return force

    def load_geometry(self,user_input=''):
        """
        Return the optimum geometry of the molecular configuration from the
        QChem log file. If multiple such geometries are identified, only the
        last is returned.
        """
        if not user_input:
            user_input = self.get_user_input()
        is_qmmm = self.is_qmmm(user_input)
        atom, coord, number, mass = [], [], [], []
        
        with open(self.path) as f:
            log = f.readlines()

        # First check that the QChem job file (not necessarily a geometry optimization)
        # has successfully completed, if not an error is thrown
        completed_job = False
        for line in reversed(log):
            if 'Total job time:' in line:
                logging.debug('Found a successfully completed QChem Job')
                completed_job = True
                break

        if not completed_job:
            raise LogError('Could not find a successfully completed QChem job '
                           'in QChem output file {0}'.format(self.path))

        # Now look for the geometry.
        # Will return the final geometry in the file under Standard Nuclear Orientation.
        geometry_flag = False
        
        # If the job is QM/MM, the file will be a freq output file, this returns the geometry associated with the partial Hessian
        if is_qmmm:
            for i,line in enumerate(log):
                if 'Standard Nuclear Orientation' in line:
                    if not geometry_flag:
                        geometry_flag = True
                        continue
                    if geometry_flag:
                        for line in log[(i+3):]:
                            if '------------' not in line:
                                data = line.split()
                                atom.append(data[1])
                                coord.append([float(c) for c in data[2:]])
                            else:
                                break
                        break
        else:
            for i in reversed(range(len(log))):
                line = log[i]
                if 'Standard Nuclear Orientation' in line:
                    for line in log[(i + 3):]:
                        if '------------' not in line:
                            data = line.split()
                            atom.append(data[1])
                            coord.append([float(c) for c in data[2:]])
                            geometry_flag = True
                        else:
                            break
                    if geometry_flag:
                        break

        # Assign appropriate mass to each atom in the molecule
        isotopes = self.get_isotopes(user_input)
        for i,atom1 in enumerate(atom):
            mass1, num1 = get_element_mass(atom1)
            number.append(num1)
            if not i+1 in isotopes.keys():
                mass.append(mass1)
            else:
                mass.append(isotopes[i+1]['mass'])

        coord = np.array(coord, np.float64)
        number = np.array(number, np.int)
        mass = np.array(mass, np.float64)
        if len(number) == 0 or len(coord) == 0 or len(mass) == 0:
            raise LogError('Unable to read atoms from QChem geometry output file {0}.'.format(self.path))

        return coord, number, mass

    def load_conformer(self, symmetry=None, spin_multiplicity=0, optical_isomers=None, label='', user_input=''):
        """
        Load the molecular degree of freedom data from an output file created as the result of a
        QChem "Freq" calculation. As QChem's guess of the external symmetry number is not always correct,
        you can use the `symmetry` parameter to substitute your own value;
        if not provided, the value in the QChem output file will be adopted.
        """
        modes = []
        freq = []
        mmass = []
        rot = []
        inertia = []
        unscaled_frequencies = []
        e0 = 0.0

        if user_input:
            is_qmmm = self.is_qmmm(user_input)
        else:
            is_qmmm = self.is_qmmm(self.get_user_input())

        if optical_isomers is None or symmetry is None:
            _optical_isomers, _symmetry, _ = self.get_symmetry_properties()
            if optical_isomers is None:
                optical_isomers = _optical_isomers
            if symmetry is None:
                symmetry = _symmetry
        with open(self.path, 'r') as f:
            line = f.readline()
            while line != '':
                # Read spin multiplicity if not explicitly given
                if '$molecule' in line.lower() and spin_multiplicity == 0:
                    line = f.readline()
                    if len(line.split()) == 2:
                        spin_multiplicity = int(float(line.split()[1]))
                        charge = int(float(line.split()[0]))
                        #print(charge)
                        #print(spin_multiplicity)
                        logging.debug(
                            'Conformer {0} is assigned a spin multiplicity of {1}'.format(label, spin_multiplicity))
                # The rest of the data we want is in the Thermochemistry section of the output
                elif 'VIBRATIONAL ANALYSIS' in line:
                    modes = []
                    line = f.readline()
                    while line != '':

                        # This marks the end of the thermochemistry section
                        if 'Thank you very much for using Q-Chem.' in line:
                            break

                        # Read vibrational modes
                        elif 'VIBRATIONAL FREQUENCIES (CM**-1)' in line:
                            frequencies = []
                            while 'STANDARD THERMODYNAMIC QUANTITIES AT' not in line:
                                if ' Frequency:' in line:
                                    if len(line.split()) == 4:
                                        frequencies.extend([float(d) for d in line.split()[-3:]])
                                    elif len(line.split()) == 3:
                                        frequencies.extend([float(d) for d in line.split()[-2:]])
                                    elif len(line.split()) == 2:
                                        frequencies.extend([float(d) for d in line.split()[-1:]])
                                line = f.readline()
                            line = f.readline()
                            # If there is an imaginary frequency, remove it
                            if frequencies[0] < 0.0:
                                frequencies = frequencies[1:]

                            unscaled_frequencies = frequencies
                            vibration = HarmonicOscillator(frequencies=(frequencies, "cm^-1"))
                            # modes.append(vibration)
                            freq.append(vibration)
                        # Read molecular mass for external translational modes
                        elif 'Molecular Mass:' in line:
                            try:
                                mass = float(line.split()[2])
                            except (ValueError, TypeError) as e:
                                #if not is_qmmm:
                                #    raise e
                                print("***** as mass: continuing with infinite mass")
                                mass = np.inf 

                            translation = IdealGasTranslation(mass=(mass, "amu"))
                            # modes.append(translation)
                            mmass.append(translation)

                        # Read moments of inertia for external rotational modes, given in atomic units
                        elif 'Eigenvalues --' in line:
                            try:
                                inertia = [float(d) for d in line.split()[-3:]]
                            except (ValueError, TypeError) as e:
                                #if not is_qmmm:
                                #    raise e
                                print("***** as inertia: continuing with infinite mass")
                                inertia = [mass for d in range(3)]
                        # Read the next line in the file
                        line = f.readline()

                # Read the next line in the file
                line = f.readline()

                if len(inertia):
                    if inertia[0] == 0.0:
                        # If the first eigenvalue is 0, the rotor is linear
                        inertia.remove(0.0)
                        logging.debug('inertia is {}'.format(str(inertia)))
                        for i in range(2):
                            inertia[i] *= (constants.a0 / 1e-10) ** 2
                        inertia = np.sqrt(inertia[0] * inertia[1])
                        rotation = LinearRotor(inertia=(inertia, "amu*angstrom^2"), symmetry=symmetry)
                        rot.append(rotation)
                    else:
                        for i in range(3):
                            inertia[i] *= (constants.a0 / 1e-10) ** 2
                            rotation = NonlinearRotor(inertia=(inertia, "amu*angstrom^2"), symmetry=symmetry)
                            # modes.append(rotation)
                        rot.append(rotation)

                    inertia = []

        modes = mmass + rot + freq
        return Conformer(E0=(e0 * 0.001, "kJ/mol"), modes=modes, spin_multiplicity=spin_multiplicity, 
                         optical_isomers=optical_isomers), unscaled_frequencies

    def load_energy(self, zpe_scale_factor=1.):
        """
        Load the energy in J/mol from a QChem log file. Only the last energy
        in the file is returned. The zero-point energy is *not* included in
        the returned value.
        """
        e_elect = None
        with open(self.path, 'r') as f:
            a = b = 0
            for line in f:
                if 'Final energy is' in line:
                    a = float(line.split()[3]) * constants.E_h * constants.Na
                if 'Total energy in the final basis set' in line:
                    b = float(line.split()[8]) * constants.E_h * constants.Na
                e_elect = a or b
        if e_elect is None:
            raise LogError('Unable to find energy in QChem output file {0}.'.format(self.path))
        return e_elect

    # Lance Bettinson wrote this --
    def load_mode_freqs(self):
        """
        Load the freqs in cm^-1 for all modes of a given conformer
        """
        from ape.main import SolvEig
        from arkane.statmech import is_linear
        user_input = self.get_user_input()
        coordinates, number, mass = self.load_geometry(user_input=user_input)
        conformer, unscaled_freqs = self.load_conformer(user_input=user_input)
        conformer.coordinates = (coordinates, "angstroms")
        conformer.number = number
        conformer.mass = (mass, "amu")
        hessian = self.load_force_constant_matrix()
        mass = conformer.mass.value_si
        linearity = is_linear(conformer.coordinates.value)
        N = self.get_number_of_atoms()
        n_vib = 3*N-(5 if linearity else 6)
        vib_freq, unweighted_v = SolvEig(hessian, mass, n_vib)
        #return vib_freq
        return vib_freq[vib_freq != 0.]

    def load_zero_point_energy(self):
        """
        Load the unscaled zero-point energy in J/mol from a QChem output file.
        """
        zpe = None
        with open(self.path, 'r') as f:
            for line in f:
                if 'Zero point vibrational energy' in line:
                    zpe = float(line.split()[4]) * 4184  # QChem's ZPE is in kcal/mol, convert to J/mol
                    logging.debug('ZPE is {}'.format(str(zpe)))
        if zpe is not None:
            return zpe
        else:
            raise LogError('Unable to find zero-point energy in QChem output file {0}.'.format(self.path))

    def load_scan_energies(self):
        """
        Extract the optimized energies in J/mol from a QChem log file, e.g. the
        result of a QChem "PES Scan" quantum chemistry calculation.
        """
        v_list = []
        angle = []
        read = False
        with open(self.path, 'r') as f:
            for line in f:
                if '-----------------' in line:
                    read = False
                if read:
                    values = [float(item) for item in line.split()]
                    angle.append(values[0])
                    v_list.append(values[1])
                if 'Summary of potential scan:' in line:
                    logging.info('found a successfully completed QChem Job')
                    read = True
                elif 'SCF failed to converge' in line:
                    raise LogError('QChem Job did not successfully complete: '
                                   'SCF failed to converge in file {0}.'.format(self.path))
        logging.info('   Assuming {0} is the output from a QChem PES scan...'.format(os.path.basename(self.path)))

        v_list = np.array(v_list, np.float64)
        # check to see if the scanlog indicates that one of your reacting species may not be the lowest energy conformer
        check_conformer_energy(v_list, self.path)

        # Adjust energies to be relative to minimum energy conformer
        # Also convert units from Hartree/particle to J/mol
        v_list -= np.min(v_list)
        v_list *= constants.E_h * constants.Na
        angle = np.arange(0.0, 2 * math.pi + 0.00001, 2 * math.pi / (len(v_list) - 1), np.float64)
        return v_list, angle

    def load_negative_frequency(self):
        """
        Return the imaginary frequency from a transition state frequency
        calculation in cm^-1.
        """
        frequency = 0
        with open(self.path, 'r') as f:
            for line in f:
                # Read imaginary frequency
                if ' Frequency:' in line:
                    frequency = float((line.split()[1]))
                    break
        # Make sure the frequency is imaginary:
        if frequency < 0:
            return frequency
        else:
            raise LogError('Unable to find imaginary frequency in QChem output file {0}.'.format(self.path))

    def load_scan_pivot_atoms(self):
        """Not implemented for QChem"""
        raise NotImplementedError('The load_scan_pivot_atoms method is not implemented for QChem Logs')

    def load_scan_frozen_atoms(self):
        """Not implemented for QChem"""
        raise NotImplementedError('The load_scan_frozen_atoms method is not implemented for QChem Logs')

register_ess_adapter("QChemLog", QChemLog)
