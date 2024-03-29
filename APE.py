# -*- coding: utf-8 -*-

"""
APE - Automated Property Estimator
"""

import argparse
import os

from ape.main import APE

def parse_command_line_arguments(command_line_args=None):
    
    parser = argparse.ArgumentParser(description='Automated Property Estimator (APE)')
    parser.add_argument('file', metavar='FILE', type=str, nargs=1,
                        help='a file describing the job to execute')
    parser.add_argument('-n', type=int, help='number of CPUs to run quantum calculation')
    parser.add_argument('-p', type=str, help='the sampling protocol(UMN or UMVT) chossen (default: UMVT)')
    parser.add_argument('-i', type=str, help='the imaginary bonds for QMMM calculation')
    parser.add_argument('-mode', type=int, help='which modes (in a list) the sampling protocol will treat (default: all)')
    parser.add_argument('-jw', type=int, help='just write the displaced geometries - do not calculate energies through QChem (default: false)')

    args = parser.parse_args(command_line_args)
    args.file = args.file[0]

    return args

def main():
    """ The main APE executable function"""
    args = parse_command_line_arguments()
    input_file = args.file
    ncpus = args.n
    protocol = args.p
    which_modes = args.mode
    just_write = args.jw
    project_directory = os.path.abspath(os.path.dirname(args.file))
    if not protocol:
        protocol = 'UMVT'
        print('This calculation will use UMVT as sampling protocol')
    elif protocol == 'UMN' or protocol == 'UMVT' or protocol == 'CMT':
        print('This calculation will use {} as sampling protocol'.format(protocol))

    # imaginary bonds for QMMM calculation
    # atom indices starts from 1
    imaginary_bonds = args.i
    if args.i is not None:
        imaginary_bonds_string = imaginary_bonds.strip('[').strip(']')
        imaginary_bonds = []
        for bond in imaginary_bonds_string.split(','):
            atom1, atom2 = bond.split('-')
            imaginary_bonds.append([int(atom1), int(atom2)])

    ape_object = APE(input_file = input_file, ncpus=ncpus, 
            project_directory=project_directory, protocol=protocol, 
            which_modes=[which_modes], imaginary_bonds=imaginary_bonds,
            just_write=just_write)
    ape_object.execute()

################################################################################

if __name__ == '__main__':
    main()
