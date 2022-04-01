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
PATH = sys.argv[5]

model_params = {
    "Lx": Lx,
    "Ly": Ly,
    # "J": J,
    # "eta": eta
    "phi": phi
}

print("\n\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
M = model.RING_EXCHANGE(model_params)

product_state = ["up"] * M.lat.N_sites
psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)

TEBD_params = {'N_steps': 10, 'trunc_params':{'chi_max': 20}, 'verbose': 0}
eng = tebd.RandomUnitaryEvolution(psi, TEBD_params)
eng.run()
psi.canonical_form() 

dmrg_params = {
    'mixer': True,  # setting this to True helps to escape local minima
    'trunc_params': {
        'chi_max': CHI,
        'svd_min': 1.e-10
    },
    'max_E_err': 1.e-10,
    'max_sweeps': 100
}

eng = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)
E, psi = eng.run()  # equivalent to dmrg.run() up to the return parameters.

mag_x = psi.expectation_value("Sx")
mag_y = psi.expectation_value("Sy")
mag_z = psi.expectation_value("Sz")
EE = psi.entanglement_entropy()
ES = psi.entanglement_spectrum()

corr_ver_x =[]
corr_ver_y =[]
corr_ver_z =[]

corr_hor_x =[]
corr_hor_y =[]
corr_hor_z =[]

# measuring NN spin correlation
for i in range(0,Lx):
    for j in range(0, Ly):

        I = i*Ly + j
        J = I + 1
        if j==(Ly-1):
            J = J - Ly
        
        corr_ver_x.append( psi.expectation_value_term([('Sx',I),('Sx',J)]) )
        corr_ver_y.append( psi.expectation_value_term([('Sy',I),('Sy',J)]) )
        corr_ver_z.append( psi.expectation_value_term([('Sz',I),('Sz',J)]) )
        
        J = I + Ly
        corr_hor_x.append( psi.expectation_value_term([('Sx',I),('Sx',J)]) )
        corr_hor_y.append( psi.expectation_value_term([('Sy',I),('Sy',J)]) )
        corr_hor_z.append( psi.expectation_value_term([('Sz',I),('Sz',J)]) )

ensure_dir(PATH + "observables/")
ensure_dir(PATH + "entanglement/")
ensure_dir(PATH + "logs/")
ensure_dir(PATH + "mps/")

file_Energy = open( PATH + "observables/energy_phi%.2f.txt" % phi,"a")
file_Energy.write(repr(E) + " " + repr(psi.correlation_length()) + " " + "\n")

file_Ss = open( PATH + "observables/magnetization_phi%.2f.txt" % phi,"a")
file_Ss.write("  ".join(map(str, mag_x)) + " " + "\n")
file_Ss.write("  ".join(map(str, mag_y)) + " " + "\n")
file_Ss.write("  ".join(map(str, mag_z)) + " " + "\n")

file_CORR = open( PATH + "observables/nn_corr_comp_phi%.2f.txt" % phi ,"a")
file_CORR.write("  ".join(map(str,corr_hor_x)) + " " + "\n")
file_CORR.write("  ".join(map(str,corr_hor_y)) + " " + "\n")
file_CORR.write("  ".join(map(str,corr_hor_z)) + " " + "\n")
file_CORR.write("  ".join(map(str,corr_ver_x)) + " " + "\n")
file_CORR.write("  ".join(map(str,corr_ver_y)) + " " + "\n")
file_CORR.write("  ".join(map(str,corr_ver_z)) + " " + "\n")

file_CORR = open( PATH + "observables/nn_corr_phi%.2f.txt" % phi ,"a")
file_CORR.write("  ".join(map(str, np.array(corr_hor_x) + np.array(corr_hor_y) + np.array(corr_hor_z))) + " " + "\n")
file_CORR.write("  ".join(map(str,corr_ver_x + corr_ver_y + corr_ver_z)) + " " + "\n")


# file_Energy.write(repr(E) + " " + "\n")
file_ES = open( PATH + "entanglement/es_phi%.2f.txt" % phi,"a")
for i in range(0,Lx*Ly):
    file_ES.write("  ".join(map(str, ES[i])) + " " + "\n")
file_EE = open( PATH + "entanglement/ee_phi%.2f.txt" % phi,"a")
file_EE.write("  ".join(map(str, EE)) + " " + "\n")

file_STAT = open( PATH + "logs/stat_phi%.2f.txt" % phi,"a")
file_STAT.write("  ".join(map(str,eng.sweep_stats['E'])) + " " + "\n")
file_STAT.write("  ".join(map(str,eng.sweep_stats['S'])) + " " + "\n")
file_STAT.write("  ".join(map(str,eng.sweep_stats['max_trunc_err'])) + " " + "\n")
file_STAT.write("  ".join(map(str,eng.sweep_stats['norm_err'])) + " " + "\n")

with open( PATH + 'mps/psi_phi%.2f.pkl' % phi, 'wb') as f:
    pickle.dump(psi, f)

print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n\n")
