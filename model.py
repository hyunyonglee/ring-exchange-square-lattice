# Copyright 2022 Hyun-Yong Lee

import numpy as np
from tenpy.models.lattice import Site, Chain
from tenpy.models.lattice import Square
from tenpy.models.model import CouplingModel, MPOModel
from tenpy.linalg import np_conserved as npc
from tenpy.tools.params import Config
from tenpy.networks.site import SpinHalfSite
import matplotlib.pyplot as plt
__all__ = ['RING_EXCHANGE']


class RING_EXCHANGE(CouplingModel,MPOModel):
    
    def __init__(self, model_params):
        
        if not isinstance(model_params, Config):
            model_params = Config(model_params, "RING_EXCHANGE")
        
        Lx = model_params.get('Lx', 1)
        Ly = model_params.get('Ly', 2)
        phi = model_params.get('phi', 0.)
        # J = model_params.get('J', 1.0)
        # eta = model_params.get('eta', 1.0)
        bc_MPS = model_params.get('bc_MPS', 'infinite')
        bc = model_params.get('bc', ['periodic','open'])
        qn = model_params.get('qn', None)

        site = SpinHalfSite(conserve=qn)
        # lat = Square(Lx=Lx, Ly=Ly, site=site, bc=bc, bc_MPS=bc_MPS, order='snakeCstyle')
        lat = Square(Lx=Lx, Ly=Ly, site=site, bc=bc, bc_MPS=bc_MPS)
        
        CouplingModel.__init__(self, lat)
        
        
        # Heisenberg term
        s = np.sin(phi*np.pi) # s = 8 * 0.5 * ( 0.5 + eta ) # s = 8 * s * ( s + eta )
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(s/2., u1, 'Sp', u2, 'Sm', dx, plus_hc=True)
            self.add_coupling(s, u1, 'Sz', u2, 'Sz', dx)

        # Ring-exchange term
        c = np.cos(phi*np.pi) #c = 4.0 * J
        # (S_{i} . S_{i+y}) (S_{i+x} . S_{i+x+y}) 
        self.add_multi_coupling( c/4., [('Sp', [0,0], 0), ('Sm', [0,1], 0), ('Sp', [1,0], 0), ('Sm', [1,1], 0)])
        self.add_multi_coupling( c/4., [('Sm', [0,0], 0), ('Sp', [0,1], 0), ('Sp', [1,0], 0), ('Sm', [1,1], 0)])
        self.add_multi_coupling( c/2., [('Sz', [0,0], 0), ('Sz', [0,1], 0), ('Sp', [1,0], 0), ('Sm', [1,1], 0)])
        self.add_multi_coupling( c/4., [('Sp', [0,0], 0), ('Sm', [0,1], 0), ('Sm', [1,0], 0), ('Sp', [1,1], 0)])
        self.add_multi_coupling( c/4., [('Sm', [0,0], 0), ('Sp', [0,1], 0), ('Sm', [1,0], 0), ('Sp', [1,1], 0)])
        self.add_multi_coupling( c/2., [('Sz', [0,0], 0), ('Sz', [0,1], 0), ('Sm', [1,0], 0), ('Sp', [1,1], 0)])
        self.add_multi_coupling( c/2., [('Sp', [0,0], 0), ('Sm', [0,1], 0), ('Sz', [1,0], 0), ('Sz', [1,1], 0)])
        self.add_multi_coupling( c/2., [('Sm', [0,0], 0), ('Sp', [0,1], 0), ('Sz', [1,0], 0), ('Sz', [1,1], 0)])
        self.add_multi_coupling( c,    [('Sz', [0,0], 0), ('Sz', [0,1], 0), ('Sz', [1,0], 0), ('Sz', [1,1], 0)])

        # (S_{i} . S_{i+x}) (S_{i+y} . S_{i+x+y}) 
        self.add_multi_coupling( c/4., [('Sp', [0,0], 0), ('Sm', [1,0], 0), ('Sp', [0,1], 0), ('Sm', [1,1], 0)])
        self.add_multi_coupling( c/4., [('Sm', [0,0], 0), ('Sp', [1,0], 0), ('Sp', [0,1], 0), ('Sm', [1,1], 0)])
        self.add_multi_coupling( c/2., [('Sz', [0,0], 0), ('Sz', [1,0], 0), ('Sp', [0,1], 0), ('Sm', [1,1], 0)])
        self.add_multi_coupling( c/4., [('Sp', [0,0], 0), ('Sm', [1,0], 0), ('Sm', [0,1], 0), ('Sp', [1,1], 0)])
        self.add_multi_coupling( c/4., [('Sm', [0,0], 0), ('Sp', [1,0], 0), ('Sm', [0,1], 0), ('Sp', [1,1], 0)])
        self.add_multi_coupling( c/2., [('Sz', [0,0], 0), ('Sz', [1,0], 0), ('Sm', [0,1], 0), ('Sp', [1,1], 0)])
        self.add_multi_coupling( c/2., [('Sp', [0,0], 0), ('Sm', [1,0], 0), ('Sz', [0,1], 0), ('Sz', [1,1], 0)])
        self.add_multi_coupling( c/2., [('Sm', [0,0], 0), ('Sp', [1,0], 0), ('Sz', [0,1], 0), ('Sz', [1,1], 0)])
        self.add_multi_coupling( c,    [('Sz', [0,0], 0), ('Sz', [1,0], 0), ('Sz', [0,1], 0), ('Sz', [1,1], 0)])
        
        MPOModel.__init__(self, lat, self.calc_H_MPO())
        