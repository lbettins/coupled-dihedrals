import os
import copy
import numpy as np
import rmgpy.constants as constants
import itertools as it
from arkane.statmech import determine_rotor_symmetry
from ape.qchem import QChemLog
from arkane.common import symbol_by_number
from ape.job import Job
from ape.InternalCoordinates import get_RedundantCoords, getXYZ
from ape.sampling import torsional_sampling, get_e_elect,\
        write_e_elect, write_specs

def coupled_torsional_sampling(ape_obj, tors_modes, 
        rotor, path, scan_res=36, thresh=0.05, just_write=False):
    xyz_dict = {}
    energy_dict = {}
    mode_dict = {}
    
    scan_res = [scan_res for i,mode in enumerate(tors_modes)]
    # do not sample endpts (periodic)
    nsamples = [int(360/scan_res[i]) for i,mode in enumerate(tors_modes)]

    pivots = [ape_obj.rotors_dict[mode]['pivots']\
            for mode in tors_modes]
    tops = [ape_obj.rotors_dict[mode]['top']\
            for mode in tors_modes]
    scans = [ape_obj.rotors_dict[mode]['scan']\
            for mode in tors_modes]
    step_sizes = [2*np.pi / nsamples[i]\
            for i,mode in enumerate(tors_modes)]#Rads
    projected_freqs = [rotor.get_projected_out_freq(scan)\
            for scan in scans]

    mode_dict['mode'] = 'tors'
    mode_dict['M'] = [ape_obj.conformer.\
            get_internal_reduced_moment_of_inertia(pivot,top) *\
            constants.Na * 1e23 for pivot,top in zip(pivots,tops)]
    mode_dict['K'] = [projected_freqs[i][0] * \
            (2*np.pi * constants.c * 100)**2 \
            for i,mode in enumerate(tors_modes)]   # in 1/s^2
    mode_dict['v_ho'] = [projected_freqs[i][0] \
            for i,mode in enumerate(tors_modes)]
    mode_dict['step_size'] = step_sizes
    mode_dict['nsamples'] = nsamples

    n_rotors = len(ape_obj.rotors_dict)

    # Create for each mode a copy of the internal object
    # For speed in iterating over dihedral space
    internals = [copy.deepcopy(ape_obj.internal)\
            for mode in tors_modes]

    # Only need to call one internal object
    # Rn they are all identical in the array
    scan_indices = internals[0].B_indices[-n_rotors:]
    torsion_inds = [len(internals[0].B_indices) - n_rotors +\
            scan_indices.index([ind-1 for ind in scan]) \
            for scan in scans]
    B = internals[0].B
    Bt_inv = np.linalg.pinv(B.dot(B.T)).dot(B)
    nrow = B.shape[0]

    # Finite displacement identity vector for each torsion
    qks = [np.zeros(nrow, dtype=int) for mode in tors_modes]
    for i,ind in enumerate(torsion_inds):
        qks[i][ind] = 1
    
    # We modify the geometries alongside the internal objects
    # Create them as a parallel structure for each dihedral
    geom_coords = [copy.deepcopy(ape_obj.cart_coords) \
            for tors in tors_modes]
    Fail_in_torsion_sampling = [False for mode in tors_modes]

    sample_list = []
    xyz_list = []
    count = 0
    equilibrium = 1
    rev_ind = 1 # Index for calling last relevant geometry

    # Iteration variables
    energy_grid = np.zeros(shape=(nsamples),dtype=float)
    sample_grid = np.array([np.array(range(0,nsamples[i])) \
            for i,mode in enumerate(tors_modes)])

    # Function for returning step displacement
    def finite_step(x, internal, qks, dxs, i, cart_coords):
        """
        For a list of steadily increasing coordinates in ND
        internal = internal object
        x = current position in dihedral space
        qk = displacement vector in internal coordinates
        dx = finite displacement size (dx)
        i = coordinate i.e. ith dihedral angle to update
        """
        if x[i] is x[0] and x[i] == 0: # Init condition
            return cart_coords[i]
        elif x[i] == 0: # Take a finite step along prev geometric coordinate
            cart_coords[i-1] = finite_step(x, internal, qks, dxs, i-1, cart_coords)
            internal[i] = copy.copy(internal[i-1])
            return cart_coords[i-1]
        else:   # Exit condition -- take a finite step 
            cart_coords[i] = cart_coords[i] + internal[i].transform_int_step(
                    (qks[i]*dxs[i]).reshape(-1,))
            return cart_coords[i]

    # Now iterate
    for x in it.product(*sample_grid):
        """
        Iterate over all samples in dihedral space.
        Samples are predefined, and steps are taken based on step_sizes
        x is a vector of dihedral N angles {x0, x1, x2, ..., x(N-1)}
        """
        print(count,*x)
        file_name = 'tors'
        geom_coords[-1] = finite_step(x, internals, qks, step_sizes, -1, geom_coords)
        xyz = getXYZ(ape_obj.symbols, geom_coords[-1])
        xyz_dict[tuple(x)] = xyz

        file_name = 'tors_{}'.format(count)
        if not just_write:
            e_elec = get_e_elect(ape_obj, xyz, path, file_name, ape_obj.ncpus)
            if equilibrium:
                energy_dict[tuple(x)] = 0 
                min_elect = e_elec
                mode_dict['min_elect'] = min_elect
                equilibrium *= 0
            else:
                energy_dict[tuple(x)] = e_elec - min_elect
            # Add energy to grid
            energy_grid[x] += e_elec - min_elect
            #if e_elec - min_elect > thresh:
            #    Fail_in_torsion_sampling = True
            #    continue
        else:
            # Lists for writing geometries for VMD software
            sample_list.append(x)
            xyz_list.append(xyz)
            job = write_e_elect(ape_obj, xyz, 
                    path, file_name, ape_obj.ncpus, just_write)
        
        count += 1
        
    # Clean it up and prepare to return
    for mode in tors_modes:
        xyz_dict[mode] = xyz_list
        energy_dict[mode] = energy_grid
    if not just_write:
        name = 'all_tors'
        mode_dict['symmetry_number'] = 1    # TODO change later
    elif just_write:    # Save sampling geometries only
        write_specs(ape_obj, count, path)
        path_to_geoms = os.path.join(ape_obj.project_directory,
                'sampling_geoms')
        write_displaced_geoms(sample_list, path_to_geoms, ape_obj.natom,
                xyz_list)
    return xyz_dict, energy_dict, mode_dict

def write_displaced_geoms(samples, path, natom, xyz_list):
    record_script = '''{natom}
    # Point {sample} Energy = {e_elect}
    {xyz}
    '''
    if not os.path.exists(path):
        os.makedirs(path)
    txt_path = os.path.join(path, 'all_tors_geoms.txt')
    if os.path.exists(txt_path):
        # Remove if file already exists
        os.system('rm {}'.format(txt_path))
        pass
    with open(txt_path, 'a') as f:
        for i,sample in enumerate(samples):
            content = record_script.format(natom=natom, sample=i,
                e_elect=0, xyz=xyz_list[i])
            f.write(content)
        f.close()
