# -*- coding: utf-8 -*-
import os
import subprocess

from ape.qchem import QChemLog

class Job(object):
    def __init__(self, xyz, path, file_name, jobtype, cpus, charge=None, multiplicity=None, level_of_theory=None, basis=None, qmmm_template=None, just_write=False):
        self.xyz = xyz
        self.path = path
        self.jobtype = jobtype
        self.cpus = cpus
        self.charge = charge
        self.multiplicity = multiplicity
        self.level_of_theory = level_of_theory
        self.basis = basis

        # Base the calculation off of previous sp input
        # needed for QM/MM
        self.qmmm_template = qmmm_template

        if self.cpus > 8:
            self.cpus = 8
        if self.charge is None:
            self.charge = 0
        if self.multiplicity is None:
            self.multiplicity = 1
        if self.level_of_theory is None:
            self.level_of_theory = 'omegaB97X-D'
        if self.basis is None:
            self.basis = '6-311+G(2df,2pd)'
        if not just_write:
            self.input_path = os.path.join(self.path, 'input.qcin')
        else:
            self.input_path = os.path.join(self.path, '{}.qcin'.format(file_name))
        self.output_path = os.path.join(self.path, '{}.q.out'.format(file_name))

    def write_input_file(self, filename = ''):
        """
        Write a software-specific, job-specific input file.
        Save the file locally and also upload it to the server.
        """
            # Get user input. This var will exist as an indicator
            # for QM/MM
        if self.jobtype in {'opt', 'ts', 'sp', 'freq'}:
            if self.qmmm_template:
                script = get_qmmm_script(template=self.qmmm_template,jobtype='sp',xyz=self.xyz)
            else:
                script = input_script.format(jobtype=self.jobtype, level_of_theory=self.level_of_theory, basis=self.basis,\
                fine=fine, charge=self.charge, multiplicity=self.multiplicity, xyz=self.xyz)
        f = open(self.input_path, 'w')
        f.write(script)
        f.close()
        # Delete basis2 if this is a freq job
        # basis2 won't exist for a QM/MM job
        if self.jobtype == 'freq':
            os.system("sed -i '' '/basis2/d' {input_path}".format(input_path=self.input_path))
    
    def submit(self):
        if os.path.exists(self.output_path):
            print('{} exists, so this calculation is passed !'.format(self.output_path))
            pass
        else:
            proc = subprocess.Popen(['qchem -nt {cpus} {input_path} {output_path}'.format(cpus=self.cpus,input_path=self.input_path,output_path=self.output_path)],shell=True)
            proc.wait()


###################################################################################
fine = """\n   max_scf_cycles   250
   geom_opt_max_cycles   1500
   basis2 6-31G*
   scf_algorithm rca_diis
   SYM_IGNORE  TRUE
   print_input   true
   geom_opt_dmax   80
   pop_mulliken false
   XC_GRID 000075000302"""

input_script = """$rem
   JOBTYPE  {jobtype}
   EXCHANGE   {level_of_theory}
   BASIS   {basis}{fine}
$end

$molecule
{charge} {multiplicity}
{xyz}
$end
"""

def get_qmmm_script(template, jobtype, xyz, level_of_theory=None, basis=None, charge=None, multiplicity=None):
    user_input = template.get_user_input()
    c,m,b,l = template.get_vars()
    if not charge:
        charge = c
    if not multiplicity:
        multiplicity = m
    if not basis:
        basis = b
    if not level_of_theory:
        level_of_theory = l
    new_script = ''
    charge_mult_toggle = False
    xyz_toggle = False
    line_by_line = user_input.splitlines()
    for i,line in enumerate(line_by_line):
        if '$molecule' in line.lower():
            new_script += line+'\n'
            charge_mult_toggle = True
        elif charge_mult_toggle:
            new_script += str(charge) + ' ' + str(multiplicity)+'\n'
            charge_mult_toggle = False
            xyz_toggle = True
            count = 0
            n_qm_atoms = template.get_number_of_qm_atoms(user_input)
            xyz_line_by_line = xyz.splitlines()
        elif xyz_toggle:
            if count < n_qm_atoms: 
                new_coords = xyz_line_by_line[count].split()[:4]
                old_connect = line.split()[4:]
                line = '\t'.join(new_coords) + '\t'+ '\t'.join(old_connect)
                count += 1
            new_script += line + '\n'
            if '$end' in line:
                xyz_toggle = False
                continue
        elif 'jobtype' in line.lower():
            new_script += 'jobtype\t'+jobtype+'\n' 
        elif 'method' in line.lower() or 'exchange' in line.lower():
            new_script += 'method\t' +level_of_theory+'\n'
        elif 'basis' in line.lower():
            new_script += 'basis\t' +basis+'\n'
        elif 'scf_guess' in line.lower():
            continue
        else:
            new_script += line + '\n'
    return new_script


#creat a format can be read by VMD software
record_script ='''{natom}
# Point {sample} Energy = {e_elect}
{xyz}
'''
