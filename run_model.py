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
J = float(sys.argv[3])
eta = float(sys.argv[4])
CHI = int(sys.argv[5])
PATH = sys.argv[6]

model_params = {
    "Lx": Lx,
    "Ly": Ly,
    "J": J,
    "eta": eta
}

print("\n\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
M = model.RING_EXCHANGE(model_params)

product_state = ["up"] * M.lat.N_sites
psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)

# if IS == 'random':
#     TEBD_params = {'N_steps': 10, 'trunc_params':{'chi_max': 20}, 'verbose': 0}
#     eng = tebd.RandomUnitaryEvolution(psi, TEBD_params)
#     eng.run()
#     psi.canonical_form() 

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
EE = psi.entanglement_entropy()

print(EE)

'''
mag_x = psi.expectation_value("Sigmax")
mag_y = psi.expectation_value("Sigmay")
mag_z = psi.expectation_value("Sigmaz")
EE = psi.entanglement_entropy()
ES = psi.entanglement_spectrum()


file_Energy = open(PATH+"/Energy.txt","a")
file_Energy.write(repr(K) + " " + repr(r) + " " + repr(E) + " " + repr(psi.correlation_length()) + " " + "\n")
file_ES = open(PATH+"/Entanglement_Spectrum.txt","a")
file_ES.write(repr(K) + " " + repr(r) + " " + "  ".join(map(str, ES[int(Ly/2)])) + " " + "\n")
file_EE = open(PATH+"/Entanglement_Entropy.txt","a")
file_EE.write(repr(K) + " " + repr(r) + " " + "  ".join(map(str, EE)) + " " + "\n")
file_Ws = open(PATH+"/Flux.txt","a")
file_Ws.write(repr(K) + " " + repr(r) + " " + "  ".join(map(str, Fluxes)) + " " + "\n")
file_Sx = open(PATH+"/Sx.txt","a")
file_Sx.write(repr(K) + " " + repr(r) + " " + "  ".join(map(str, mag_x)) + " " + "\n")
file_Sy = open(PATH+"/Sy.txt","a")
file_Sy.write(repr(K) + " " + repr(r) + " " + "  ".join(map(str, mag_y)) + " " + "\n")
file_Sz = open(PATH+"/Sz.txt","a")
file_Sz.write(repr(K) + " " + repr(r) + " " + "  ".join(map(str, mag_z)) + " " + "\n")
file_Current = open("Current.txt","a")
file_Current.write(repr(K) + " " + repr(r) + " " + "  ".join(map(str, C)) + " " + "\n")

file_STAT = open( (PATH+"/Stat_r%.2f.txt" % r) ,"a")
file_STAT.write(" " + "  ".join(map(str,eng.sweep_stats['E'])) + " " + "\n")
file_STAT.write(" " + "  ".join(map(str,eng.sweep_stats['S'])) + " " + "\n")
file_STAT.write(" " + "  ".join(map(str,eng.sweep_stats['max_trunc_err'])) + " " + "\n")
file_STAT.write(" " + "  ".join(map(str,eng.sweep_stats['norm_err'])) + " " + "\n")

filename = 'psi_r%.2f.pkl' % r
with open( filename, 'wb') as f:
    pickle.dump(psi, f)

print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n\n")
'''