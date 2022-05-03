# Copyright 2022 Hyun-Yong Lee

import numpy as np
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
from tenpy.algorithms import tebd
import os
import os.path
import sys
import matplotlib.pyplot as plt
import pickle
import model

def ensure_dir(f):
    d=os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)
    return d;

import logging.config
conf = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {'custom': {'format': '%(levelname)-8s: %(message)s'}},
    'handlers': {'to_file': {'class': 'logging.FileHandler',
                             'filename': 'log',
                             'formatter': 'custom',
                             'level': 'INFO',
                             'mode': 'a'},
                'to_stdout': {'class': 'logging.StreamHandler',
                              'formatter': 'custom',
                              'level': 'INFO',
                              'stream': 'ext://sys.stdout'}},
    'root': {'handlers': ['to_stdout', 'to_file'], 'level': 'DEBUG'},
}
logging.config.dictConfig(conf)

os.environ["OMP_NUM_THREADS"] = "68"

Lx = int(sys.argv[1])
Ly = int(sys.argv[2])
phi = float(sys.argv[3])
CHI = int(sys.argv[4])
QN = sys.argv[5]
IS = sys.argv[6]
RM = sys.argv[7]
PATH = sys.argv[8]

model_params = {
    "Lx": Lx,
    "Ly": Ly,
    # "J": J,
    # "eta": eta
    "phi": phi,
    "qn": QN,
    'bc': 'periodic'
}

print("\n\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
M = model.RING_EXCHANGE(model_params)

if IS == 'all_up':
    product_state = ["up"] * int(M.lat.N_sites)
elif IS == 'neel':
    product_state = ( ["up", "down"] * int(Ly/2) + ["down", "up"] * int(Ly/2) ) * int(Lx/2)
    # product_state = ["up", "down"] * int(M.lat.N_sites/2)
elif IS == 'plateau':
    product_state = ( ["up", "down"] * int(Ly/2) + ["up"] * int(Ly) ) * int(Lx/2)
    

psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)

if RM == 'randomize':
    TEBD_params = {'N_steps': 10, 'trunc_params':{'chi_max': 20}, 'verbose': 0}
    eng = tebd.RandomUnitaryEvolution(psi, TEBD_params)
    eng.run()
    psi.canonical_form() 

dchi = int(CHI/2)
chi_list = {}
for i in range(2):
    chi_list[i*10] = (i+1)*dchi

dmrg_params = {
    'mixer': True,  # setting this to True helps to escape local minima
    'mixer_params': {
        'amplitude': 1.e-5,
        'decay': 1.1,
        'disable_after': 50
    },
    'trunc_params': {
        'chi_max': CHI,
        'svd_min': 1.e-9
    },
    'lanczos_params': {
            'N_min': 5,
            'N_max': 20
    },
    'chi_list': chi_list,
    'max_E_err': 1.0e-8,
    'max_S_err': 1.0e-4,
    'max_sweeps': 200
}

eng = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)
E, psi = eng.run()  # equivalent to dmrg.run() up to the return parameters.

mag_p = psi.expectation_value("Sp")
mag_m = psi.expectation_value("Sm")
mag_z = psi.expectation_value("Sz")
EE = psi.entanglement_entropy()
ES = psi.entanglement_spectrum()

corr_ver_pm =[]
corr_ver_mp =[]
corr_ver_z =[]

corr_hor_pm =[]
corr_hor_mp =[]
corr_hor_z =[]

# measuring NN spin correlation
for i in range(0,Lx): 
    for j in range(0, Ly):

        I = i*Ly + j
        J = I + 1
        if j==(Ly-1):
            J = J - Ly
        
        corr_ver_pm.append( psi.expectation_value_term([('Sp',I),('Sm',J)]) )
        corr_ver_mp.append( psi.expectation_value_term([('Sm',I),('Sp',J)]) )
        corr_ver_z.append( psi.expectation_value_term([('Sz',I),('Sz',J)]) )
        
        J = I + Ly
        corr_hor_pm.append( psi.expectation_value_term([('Sp',I),('Sm',J)]) )
        corr_hor_mp.append( psi.expectation_value_term([('Sm',I),('Sp',J)]) )
        corr_hor_z.append( psi.expectation_value_term([('Sz',I),('Sz',J)]) )

ensure_dir(PATH + "observables/")
ensure_dir(PATH + "entanglement/")
ensure_dir(PATH + "logs/")
ensure_dir(PATH + "mps/")

file1 = open( PATH + "observables/energy.txt","a")
file1.write(repr(phi) + " " + repr(E) + " " + repr(psi.correlation_length()) + " " + "\n")

file2 = open( PATH + "observables/sx.txt","a")
file2.write(repr(phi) + " " + "  ".join(map(str, np.real(mag_p+mag_m)/2.)) + " " + "\n")

file3 = open( PATH + "observables/sy.txt","a")
file3.write( repr(phi) + " " + "  ".join(map(str, np.imag(mag_p-mag_m)/2.)) + " " + "\n")

file4 = open( PATH + "observables/sz.txt","a")
file4.write( repr(phi) + " " + "  ".join(map(str, mag_z)) + " " + "\n")  


file_Energy = open( PATH + "observables/energy_phi%.5f.txt" % phi,"a")
file_Energy.write(repr(E) + " " + repr(psi.correlation_length()) + " " + "\n")

file_Ss = open( PATH + "observables/magnetization_phi%.5f.txt" % phi,"a")
file_Ss.write("  ".join(map(str, np.real(mag_p+mag_m)/2.)) + " " + "\n")
file_Ss.write("  ".join(map(str, np.imag(mag_p-mag_m)/2.)) + " " + "\n")
file_Ss.write("  ".join(map(str, mag_z)) + " " + "\n")  

file_CORR = open( PATH + "observables/nn_corr_phi%.5f.txt" % phi ,"a")
file_CORR.write("  ".join(map(str, np.array(corr_ver_pm)/2. + np.array(corr_ver_mp)/2. + np.array(corr_ver_z))) + " " + "\n")
file_CORR.write("  ".join(map(str, np.array(corr_hor_pm)/2. + np.array(corr_hor_mp)/2. + np.array(corr_hor_z))) + " " + "\n")


# file_Energy.write(repr(E) + " " + "\n")
file_ES = open( PATH + "entanglement/es_phi%.5f.txt" % phi,"a")
for i in range(0,Lx*Ly):
    file_ES.write("  ".join(map(str, ES[i])) + " " + "\n")
file_EE = open( PATH + "entanglement/ee_phi%.5f.txt" % phi,"a")
file_EE.write("  ".join(map(str, EE)) + " " + "\n")

file_STAT = open( PATH + "logs/stat_phi%.5f.txt" % phi,"a")
file_STAT.write("  ".join(map(str,eng.sweep_stats['E'])) + " " + "\n")
file_STAT.write("  ".join(map(str,eng.sweep_stats['S'])) + " " + "\n")
file_STAT.write("  ".join(map(str,eng.sweep_stats['max_trunc_err'])) + " " + "\n")
file_STAT.write("  ".join(map(str,eng.sweep_stats['norm_err'])) + " " + "\n")

with open( PATH + 'mps/psi_phi%.5f.pkl' % phi, 'wb') as f:
    pickle.dump(psi, f)

print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n\n")
