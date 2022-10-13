# Copyright 2022 Hyun-Yong Lee

import numpy as np
from tenpy.models.lattice import Site, Chain, Square
from tenpy.models.model import CouplingModel, NearestNeighborModel, MPOModel, CouplingMPOModel
from tenpy.linalg import np_conserved as npc
from tenpy.tools.params import Config
from tenpy.networks.site import SpinHalfFermionSite  # if you want to use the predefined site
import matplotlib.pyplot as plt
__all__ = ['DIPOLAR_FERMI_HUBBARD']


class DIPOLAR_FERMI_HUBBARD(CouplingModel,MPOModel):
    
    def __init__(self, model_params):
        
        # 0) read out/set default parameters 
        if not isinstance(model_params, Config):
            model_params = Config(model_params, "DIPOLAR_FERMI_HUBBARD")
        L = model_params.get('L', 1)
        t = model_params.get('t', 1.)
        tp = model_params.get('tp', 0.)
        h = model_params.get('h', 0.)
        U = model_params.get('U', 0.)
        mu = model_params.get('mu', 0.)
        bc_MPS = model_params.get('bc_MPS', 'infinite')
        bc = model_params.get('bc', 'periodic')
        QN = model_params.get('QN', 'N')
        QS = model_params.get('QS', 'Sz')

        site = SpinHalfFermionSite( cons_N=QN, cons_Sz=QS, filling=1.0 )
        site.multiply_operators(['Cu','Cd'])
        site.multiply_operators(['Cd','Cu'])
        site.multiply_operators(['Cdd','Cdu'])
        site.multiply_operators(['Cdu','Cdd'])
        
        # MPS boundary condition
        if bc_MPS == 'finite' and bc == 'periodic':
            order = 'folded'
        else:
        	order = 'default'
        
        lat = Chain( L=L, site=site, bc=bc, bc_MPS=bc_MPS, order=order )
        CouplingModel.__init__(self, lat)

        # 2-site hopping
        self.add_coupling( -h, 0, 'Cdu', 0, 'Cu', 1, plus_hc=True)
        self.add_coupling( -h, 0, 'Cdd', 0, 'Cd', 1, plus_hc=True)
        
        # 3-site hopping
        self.add_multi_coupling( -t, [('Cdu', 0, 0), ('Cu Cd', 1, 0), ('Cdd', 2, 0)])
        self.add_multi_coupling( -t, [('Cdd', 0, 0), ('Cd Cu', 1, 0), ('Cdu', 2, 0)])
        
        self.add_multi_coupling( -t, [('Cd', 2, 0), ('Cdd Cdu', 1, 0), ('Cu', 0, 0)])
        self.add_multi_coupling( -t, [('Cu', 2, 0), ('Cdu Cdd', 1, 0), ('Cd', 0, 0)])
        

        # 4-site hopping
        self.add_multi_coupling( -tp, [('Cdu', 0, 0), ('Cu', 1, 0), ('Cd', 2, 0), ('Cdd', 3, 0)])
        self.add_multi_coupling( -tp, [('Cdd', 0, 0), ('Cd', 1, 0), ('Cu', 2, 0), ('Cdu', 3, 0)])

        self.add_multi_coupling( -tp, [('Cd', 3, 0), ('Cdd', 2, 0), ('Cdu', 1, 0), ('Cu', 0, 0)])
        self.add_multi_coupling( -tp, [('Cu', 3, 0), ('Cdu', 2, 0), ('Cdd', 1, 0), ('Cd', 0, 0)])
        

        # Onsite Hubbard Interaction
        self.add_onsite( U, 0, 'NuNd')

        # Chemical potential
        self.add_onsite( -mu, 0, 'Ntot')

        
        MPOModel.__init__(self, lat, self.calc_H_MPO())


