# -*- coding: utf-8 -*-

"""
A module to sample the geometries along given direction
"""

import os
import copy
import numpy as np
import rmgpy.constants as constants
from arkane.statmech import determine_rotor_symmetry
from ape.qchem import QChemLog
from arkane.common import symbol_by_number
from ape.job import Job
from ape.InternalCoordinates import get_RedundantCoords, getXYZ

def get_e_elect(ape_obj, xyz, path, file_name, cpus):
    job = write_e_elect(ape_obj, xyz, path, file_name, cpus)
    job.submit()
    output_file_path = os.path.join(path, '{}.q.out'.format(file_name))
    e_elect = QChemLog(output_file_path).load_energy()\
            / (constants.E_h * constants.Na) # in Hartree/particle
    return e_elect

def torsional_sampling(ape_obj, mode, 
        rotor, path, scan_res=10, thresh=0.05, just_write=False):
    xyz_dict = {}
    energy_dict = {}
    mode_dict = {}

    pivots = ape_obj.rotors_dict[mode]['pivots']
    top = ape_obj.rotors_dict[mode]['top']
    scan = ape_obj.rotors_dict[mode]['scan']
    step_size = np.pi / (180/scan_res)
    projected_freq, reduced_mass = rotor.get_projected_out_freq(scan)

    mode_dict['mode'] = 'tors'
    mode_dict['M'] = ape_obj.conformer.get_internal_reduced_moment_of_inertia(
            pivots,top) * constants.Na * 1e23 # in amu*angstrom^2
    mode_dict['K'] = (projected_freq * \
            (2 * np.pi * constants.c * 100)) ** 2 # in 1/s^2
    mode_dict['v_ho'] = projected_freq #cm^-1
    mode_dict['step_size'] = step_size # in radian

    n_rotors = len(ape_obj.rotors_dict)
    internal = copy.deepcopy(ape_obj.internal)
    scan_indices = internal.B_indices[-n_rotors:]
    torsion_ind = len(internal.B_indices) - n_rotors +\
            scan_indices.index([ind-1 for ind in scan])
    print(torsion_ind)

    B = internal.B
    Bt_inv = np.linalg.pinv(B.dot(B.T)).dot(B)
    nrow = B.shape[0]
    qk = np.zeros(nrow, dtype=int)
    qk[torsion_ind] = 1
    nsample = int(360/scan_res) + 1

    initial_geometry = ape_obj.cart_coords
    cart_coords = initial_geometry.copy()
    Fail_in_torsion_sampling = False
    for sample in range(nsample):
        xyz = getXYZ(ape_obj.symbols, cart_coords)
        file_name = 'tors_{}_{}'.format(mode,sample)
        xyz_dict[sample] = xyz
        if not just_write:  # If write-geom-only, do not calculate energies
            e_elec = get_e_elect(ape_obj, xyz, path, file_name, ape_obj.ncpus)
            if sample == 0:
                energy_dict[sample] = 0
                min_elect = e_elec
                mode_dict['min_elect'] = min_elect
            else:
                energy_dict[sample] = e_elec - min_elect
            if e_elec - min_elect > thresh:
                Fail_in_torsion_sampling = True
                print('The torsional barrier of mode {} is higher than {} hartree,'\
                        .format(mode,thresh))
                print('this mode will use harmonic basis in its H matrix.')
                step_size = np.sqrt(constants.hbar / (reduced_mass * constants.amu)\
                        / (projected_freq * 2 * np.pi * constants.c * 100))\
                        * 10 ** 10      # in angstrom
                xyz_dict, energy_dict, mode_dict = vibrational_sampling(ape_obj, mode,
                        qk, projected_freq, reduced_mass, step_size, 
                        path, thresh)
                break
        # TAKE FINITE DISPLACEMENT IN TORSIONAL DIRECTION: ALWAYS
        print(qk,step_size, qk*step_size)
        #print(internal.transform_int_step((qk*step_size).reshape(-1,)))
        print(xyz)
        cart_coords += internal.transform_int_step((qk*step_size).reshape(-1,))


    if not Fail_in_torsion_sampling and not just_write:
        v_list = [i * (constants.E_h * constants.Na) \
                for i in energy_dict.values()] # in J/mol
        name = 'tors_{}'.format(mode)
        symmetry_number = determine_rotor_symmetry(v_list, name, scan)
        mode_dict['symmetry_number'] = symmetry_number
        print(symmetry_number)
    elif just_write:    # Save sampling geometries only
        path_to_geoms = os.path.join(ape_obj.project_directory,'sampling_geoms')
        write_displaced_geoms(mode, path_to_geoms, ape_obj.natom, 
                mode_dict, xyz_dict)
    return xyz_dict, energy_dict, mode_dict 

def vibrational_sampling(ape_obj, mode, 
        internal_vector, freq, reduced_mass, step_size, 
        path, thresh=0.05, max_nloop=20, just_write=False):
    XyzDictOfEachMode = {}
    EnergyDictOfEachMode = {}
    ModeDictOfEachMode = {}

    print("Sampling Mode:",mode)
    ModeDictOfEachMode['mode'] = 'vib'
    ModeDictOfEachMode['M'] = reduced_mass # in amu
    ModeDictOfEachMode['K'] = (freq * (2 * np.pi * constants.c * 100)) ** 2 # in 1/s^2
    ModeDictOfEachMode['v_ho'] = freq
    ModeDictOfEachMode['step_size'] = step_size # in angstrom
    
    initial_geometry = ape_obj.cart_coords
    cart_coords = initial_geometry.copy()
    internal = copy.deepcopy(ape_obj.internal)
    qj = internal_vector

    sample = 0
    while True:
        """
        Sample in positive direction.
        """
        xyz = getXYZ(ape_obj.symbols, cart_coords)
        file_name = 'vib_{}_{}'.format(mode,sample)
        XyzDictOfEachMode[sample] = xyz
        if not just_write:  # if not just writing sampling geoms
            e_elec = get_e_elect(ape_obj, xyz, path, file_name, ape_obj.ncpus)
            if ape_obj.is_qmmm:
                full_n, full_xyz = get_full_geometry(path, file_name)
                XyzDictOfEachMode[sample] = full_xyz

            if sample == 0:
                EnergyDictOfEachMode[sample] = 0
                min_elect = e_elec
                ModeDictOfEachMode['min_elect'] = min_elect
            else:
                EnergyDictOfEachMode[sample] = e_elec - min_elect
            if e_elec - min_elect > thresh:
                break
        sample += 1
        if sample > max_nloop:
            print('The energy of the end point is not above the cutoff value {thresh} hartree. Please increase the max_nloop value or increase step_size'.format(thresh=thresh))
            break
        print(sample)
        try:
            cart_coords += internal.transform_int_step((qj*step_size).reshape(-1,))
        except:
            if just_write:
                break
    
    cart_coords = initial_geometry.copy()
    internal = copy.deepcopy(ape_obj.internal)  
    cart_coords += internal.transform_int_step((-qj*step_size).reshape(-1,))
    sample = -1
    while True:
        """
        Sample in negative direction.
        """
        xyz = getXYZ(ape_obj.symbols, cart_coords)
        file_name = 'vib_{}_{}'.format(mode,sample)
        XyzDictOfEachMode[sample] = xyz
        if not just_write:  # if not just writing sampling geoms
            e_elec = get_e_elect(ape_obj, xyz, path, file_name, ape_obj.ncpus)
            if ape_obj.is_qmmm:
                full_n, full_xyz = get_full_geometry(path, file_name)
                XyzDictOfEachMode[sample] = full_xyz
            EnergyDictOfEachMode[sample] = e_elec - min_elect

            if e_elec - min_elect > thresh:
                break
        sample -= 1
        if sample < -max_nloop:
            print('The energy of the end point is not above the cutoff value {thresh} hartree. Please increase the max_nloop value or increase step_size'.format(thresh=thresh))
            break
        try:
            cart_coords += internal.transform_int_step((-qj*step_size).reshape(-1,))
        except:
            if just_write:
                break

    if just_write:    # Save sampling geometries only
        path_to_geoms = os.path.join(ape_obj.project_directory,'sampling_geoms')
        write_displaced_geoms(mode, path_to_geoms, ape_obj.natom,
                ModeDictOfEachMode, XyzDictOfEachMode)
    return XyzDictOfEachMode, EnergyDictOfEachMode, ModeDictOfEachMode

def get_full_geometry(path, file_name):
    output_file_path = os.path.join(path, '{}.q.out'.format(file_name))
    coordinates, number, mass = QChemLog(output_file_path).load_geometry(' ')
    symbols = [symbol_by_number[i] for i in number]
    cart_coords = coordinates.reshape(-1,)
    natom = len(coordinates)
    xyz = getXYZ(symbols, cart_coords)
    return natom, xyz

def write_displaced_geoms(mode, path, natom, mode_dict, xyz_dict):
    # This is already on a mode-to-mode basis
    #creat a format can be read by VMD software
    record_script = '''{natom}
    # Point {sample} Energy = {e_elect}
    {xyz}
    '''
    if not os.path.exists(path):
        os.makedirs(path)
    if mode_dict['mode'] == 'tors':
        txt_path = os.path.join(path, 'mode_{}_tors_geoms.txt'.format(mode))
    else:
        txt_path = os.path.join(path, 'mode_{}_vib_geoms.txt'.format(mode))
    if os.path.exists(txt_path):
        # Remove if file already exists
        os.system('rm {}'.format(txt_path))
        pass
    with open(txt_path, 'a') as f:
        for sample in xyz_dict.keys():
            content = record_script.format(natom=natom, sample=sample, 
                    e_elect=0, xyz=xyz_dict[sample])
            f.write(content)
        f.close()

def write_e_elect(ape_obj, xyz, path, file_name, cpus, just_write=False):
    job = Job(xyz, path, file_name,jobtype='sp', cpus=cpus,
            charge=ape_obj.charge, multiplicity=ape_obj.multiplicity,
            level_of_theory=ape_obj.level_of_theory, basis=ape_obj.basis,
            qmmm_template=ape_obj.qmmm_template, just_write=just_write)
    job.write_input_file() 
    return job

def write_specs(ape_obj, count, path):
    filename=os.path.join(path,'specs')
    with open(filename,'w') as f:
        f.write(ape_obj.name+'\n')
        f.write('-array\t{start}-{end}\n'.format(start=0,end=count))
        f.write('-nt\t{nt}'.format(nt=ape_obj.ncpus))
        f.close()
