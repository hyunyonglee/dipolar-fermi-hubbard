# Copyright 2022 Hyun-Yong Lee

import numpy as np
import model
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
from tenpy.algorithms import tebd
from tenpy.algorithms import tdvp
import os
import os.path
import sys
import matplotlib.pyplot as plt
import pickle
import random

def ensure_dir(f):
    d=os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)
    return d

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

L = int(sys.argv[1])
t = float(sys.argv[2])
tp = float(sys.argv[3])
U = float(sys.argv[4])
mu = float(sys.argv[5])
CHI = int(sys.argv[6])
RM = sys.argv[7]
QN = sys.argv[8]
QS = sys.argv[9]
BC_MPS = sys.argv[10]
BC = sys.argv[11]
IS = sys.argv[12]
TOL = float(sys.argv[13])
h = float(sys.argv[14])
EXC = sys.argv[15]

model_params = {
    "L": L,
    "t": t,
    "tp": tp,
    "h": h,
    "U": U,
    "mu": mu,
    "bc_MPS": BC_MPS,
    "bc": BC,
    "QN": QN,
    "QS": QS
}


print("\n\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
M = model.DIPOLAR_FERMI_HUBBARD(model_params)

# initial state
if IS == 'up':
    product_state = ['up'] * M.lat.N_sites
elif IS == 'up-down':
    product_state = ['up','down'] * int(M.lat.N_sites/2)
elif IS == 'empty-full':
    product_state = ['empty','full'] * int(M.lat.N_sites/2)
elif any( IS == frac for frac in ['0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0'] ):
    
    n = float(IS)
    Ls = round(n*L/2)
    # product_state = ['up','down'] * Ls + ['empty'] * (L-2*Ls)
    product_state = ['empty'] * (round(L/2)-Ls) +['up','down'] * Ls + ['empty'] * (round(L/2)-Ls)
    # product_state = ['empty'] * (round(L/2)-Ls) +['empty','full'] * Ls + ['empty'] * (round(L/2)-Ls)
    # Ls = round(n*L)
    # product_state = ['full'] * Ls + ['empty'] * (L-Ls)
    
psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)


if RM == 'random':
    # TEBD_params = {'N_steps': 10, 'trunc_params':{'chi_max': 32}, 'verbose': 0}
    # eng = tebd.RandomUnitaryEvolution(psi, TEBD_params)
    # eng.run()
    # psi.canonical_form() 

    model_params_tdvp = {
    "L": L,
    "t": 0.0,
    "tp": 0.0,
    "h": 1.0,
    "U": 0.0,
    "mu": mu,
    "bc_MPS": 'finite',
    "bc": 'open',
    "QN": QN,
    "QS": QS
    }

    M_tdvp = model.DIPOLAR_FERMI_HUBBARD(model_params_tdvp)

    tdvp_params = {
    'start_time': 0,
    'dt': 0.1,
    'trunc_params': {
    'chi_max': 32,
    'svd_min': 1.e-10,
    'trunc_cut': None}
    }

    tdvp_engine = tdvp.TDVPEngine(psi, M_tdvp, tdvp_params)
    tdvp_engine.run_two_sites(N_steps=10)
    # psi.canonical_form()     

chi_list = { 0: 64, 5: 16, 10: 32, 15: 64, 20: CHI}

if BC_MPS == 'infinite':
    max_sweep = 500
    disable_after = 50
    S_err = TOL
else:
    max_sweep = 500
    disable_after = 20
    S_err = TOL

dmrg_params = {
    # 'mixer': True,  # setting this to True helps to escape local minima
    'mixer' : dmrg.SubspaceExpansion,
    'mixer_params': {
        'amplitude': 1.e-2,
        'decay': 1.5,
        'disable_after': disable_after
    },
    'trunc_params': {
        'chi_max': CHI,
        'svd_min': 1.e-9
    },
    # 'lanczos_params': {
    #         'N_min': 5,
    #         'N_max': 20
    # },
    'chi_list': chi_list,
    'max_E_err': 1.0e-8,
    'max_S_err': S_err,
    'max_sweeps': max_sweep,
    'combine' : True
}

ensure_dir("observables/")
ensure_dir("entanglement/")
ensure_dir("logs/")
ensure_dir("mps/")

# ground state
eng = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)
E, psi = eng.run()  # equivalent to dmrg.run() up to the return parameters.
Nu = psi.expectation_value("Nu")
Nd = psi.expectation_value("Nd")
Cu = np.abs( psi.expectation_value("Cu") )
Cd = np.abs( psi.expectation_value("Cd") )
EE = psi.entanglement_entropy()
ES = psi.entanglement_spectrum()

if BC_MPS == 'finite':
    R = L-1
    xi = 0.
    I0 = int(L/3)
    R_CORR = int(L/3)
    
else:
    R = L
    xi = psi.correlation_length()
    I0 = 0
    R_CORR = 100

# measuring exciton condensation
hus = []
hds = []
# if BC_MPS == 'finite' and BC == 'periodic':
#     for i in range(0,int(L/2-1)): 
#         I = 2*i
#         hus.append( np.abs( psi.expectation_value_term([('Cdu',I+2),('Cu',I)]) ) )
#     hus.append( np.abs( psi.expectation_value_term([('Cdu',L-1),('Cu',L-2 )]) ) )
#     for i in range(0,int(L/2-1)):
#         I = L-1 - 2*i
#         hus.append( np.abs( psi.expectation_value_term([('Cdu',I-2),('Cu',I)]) ) )
#     hus.append( np.abs( psi.expectation_value_term([('Cdu',0),('Cu',1)]) ) )

#     for i in range(0,int(L/2-1)): 
#         I = 2*i
#         hds.append( np.abs( psi.expectation_value_term([('Cdd',I+2),('Cd',I)]) ) )
#     hds.append( np.abs( psi.expectation_value_term([('Cdd',L-1),('Cd',L-2 )]) ) )
#     for i in range(0,int(L/2-1)):
#         I = L-1 - 2*i
#         hds.append( np.abs( psi.expectation_value_term([('Cdd',I-2),('Cd',I)]) ) )
#     hds.append( np.abs( psi.expectation_value_term([('Cdd',0),('Cd',1)]) ) )

# else:
for i in range(0,R): 
    hus.append( np.abs( psi.expectation_value_term([('Cdu',i+1),('Cu',i)]) ) )
    hds.append( np.abs( psi.expectation_value_term([('Cdd',i+1),('Cd',i)]) ) )


'''
# measuring correlation functions
cor_cucu = []
cor_cdcd = []
cor_du = []
cor_du_conn = []
cor_dd = []
cor_dd_conn = []
for i in range(R_CORR):

    cor = psi.expectation_value_term([('Cdu',I0+1),('Cu',I0),('Cdu',I0+2+i),('Cu',I0+3+i)])
    cc1 = psi.expectation_value_term([('Cdu',I0+1),('Cu',I0)])
    cc2 = psi.expectation_value_term([('Cdu',I0+2+i),('Cu',I0+3+i)])
    
    cor_du.append( np.abs( cor ) )
    cor_du_conn.append( np.abs( cor - cc1*cc2 ) )   

    cor = psi.expectation_value_term([('Cdd',I0+1),('Cd',I0),('Cdd',I0+2+i),('Cd',I0+3+i)])
    cc1 = psi.expectation_value_term([('Cdd',I0+1),('Cd',I0)])
    cc2 = psi.expectation_value_term([('Cdd',I0+2+i),('Cd',I0+3+i)])
    
    cor_dd.append( np.abs( cor ) )
    cor_dd_conn.append( np.abs( cor - cc1*cc2 ) )   

    cor_cucu.append( np.abs( psi.expectation_value_term([('Cdu',I0),('Cu',I0+1+i)])))
    cor_cdcd.append( np.abs( psi.expectation_value_term([('Cdd',I0),('Cd',I0+1+i)])))

file = open( "observables/corr_cucu.txt","a")
file.write(repr(t) + " " + repr(tp) + " " + repr(U) + " " + repr(mu) + " " + "  ".join(map(str, cor_cucu)) + " " + "\n")
file.close()

file = open( "observables/corr_cdcd.txt","a")
file.write(repr(t) + " " + repr(tp) + " " + repr(U) + " " + repr(mu) + " " + "  ".join(map(str, cor_cdcd)) + " " + "\n")
file.close()

file = open( "observables/corr_du.txt","a")
file.write(repr(t) + " " + repr(tp) + " " + repr(U) + " " + repr(mu) + " " + "  ".join(map(str, cor_du)) + " " + "\n")
file.close()

file = open( "observables/corr_dd.txt","a")
file.write(repr(t) + " " + repr(tp) + " " + repr(U) + " " + repr(mu) + " " + "  ".join(map(str, cor_dd)) + " " + "\n")
file.close()

file = open( "observables/corr_du_conn.txt","a")
file.write(repr(t) + " " + repr(tp) + " " + repr(U) + " " + repr(mu) + " " + "  ".join(map(str, cor_du_conn)) + " " + "\n")
file.close()

file = open( "observables/corr_dd_conn.txt","a")
file.write(repr(t) + " " + repr(tp) + " " + repr(U) + " " + repr(mu) + " " + "  ".join(map(str, cor_dd_conn)) + " " + "\n")
file.close()

'''


file = open( "observables/energy.txt","a")
file.write(repr(t) + " " + repr(tp) + " " + repr(U) + " " + repr(mu) + " " + repr(E) + " " + repr( np.mean(Nu) ) + " " + repr( np.mean(Nd) ) + " " + repr( np.mean(Cu) ) + " " + repr( np.mean(Cd) ) + " " + repr( np.mean(hus) ) + " " + repr( np.mean(hds) ) + " " + repr(xi) + " " + "\n")
file.close()

file = open( "observables/N.txt","a")
file.write(repr(t) + " " + repr(tp) + " " + repr(U) + " " + repr(mu) + " " + "  ".join(map(str, Nu+Nd)) + " " + "\n")
file.close()

file = open( "observables/Nu.txt","a")
file.write(repr(t) + " " + repr(tp) + " " + repr(U) + " " + repr(mu) + " " + "  ".join(map(str, Nu)) + " " + "\n")
file.close()

file = open( "observables/Nd.txt","a")
file.write(repr(t) + " " + repr(tp) + " " + repr(U) + " " + repr(mu) + " " + "  ".join(map(str, Nd)) + " " + "\n")
file.close()

file = open( "observables/Cu.txt","a")
file.write(repr(t) + " " + repr(tp) + " " + repr(U) + " " + repr(mu) + " " + "  ".join(map(str, Cu)) + " " + "\n")
file.close()

file = open( "observables/Cd.txt","a")
file.write(repr(t) + " " + repr(tp) + " " + repr(U) + " " + repr(mu) + " " + "  ".join(map(str, Cd)) + " " + "\n")
file.close()

file = open( "observables/Du.txt","a")
file.write(repr(t) + " " + repr(tp) + " " + repr(U) + " " + repr(mu) + " " + "  ".join(map(str, hus)) + " " + "\n")
file.close()

file = open( "observables/Dd.txt","a")
file.write(repr(t) + " " + repr(tp) + " " + repr(U) + " " + repr(mu) + " " + "  ".join(map(str, hds)) + " " + "\n")
file.close()

file = open( "observables/entanglement_entropy.txt","a")
file.write(repr(t) + " " + repr(tp) + " " + repr(U) + " " + repr(mu) + " " + "  ".join(map(str, EE)) + " " + "\n")
file.close()


file_ES = open( "entanglement/es_t_%.3f_tp_%.3f_U_%.2f_mu_%.2f.txt" % (t,tp,U,mu),"a")
for i in range(0,R):
    file_ES.write("  ".join(map(str, ES[i])) + " " + "\n")
file_EE = open( "entanglement/ee_t_%.3f_tp_%.3f_U_%.2f_mu_%.2f.txt" % (t,tp,U,mu),"a")
file_EE.write("  ".join(map(str, EE)) + " " + "\n")

file_STAT = open( "logs/stat_t_%.3f_tp_%.3f_U_%.2f_mu_%.2f.txt" % (t,tp,U,mu),"a")
file_STAT.write("  ".join(map(str,eng.sweep_stats['E'])) + " " + "\n")
file_STAT.write("  ".join(map(str,eng.sweep_stats['S'])) + " " + "\n")
file_STAT.write("  ".join(map(str,eng.sweep_stats['max_trunc_err'])) + " " + "\n")
file_STAT.write("  ".join(map(str,eng.sweep_stats['norm_err'])) + " " + "\n")

with open( 'mps/gs_t_%.2f_tp_%.2f_U%.2f_mu%.2f.pkl' % (t,tp,U,mu), 'wb') as f:
    pickle.dump(psi, f)



print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n\n")
