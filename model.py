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
        J = model_params.get('J', 1.0)
        eta = model_params.get('eta', 1.0)
        bc_MPS = model_params.get('bc_MPS', 'infinite')
        bc = model_params.get('bc', 'periodic')

        site = SpinHalfSite(conserve=None)
        lat = Square(Lx=Lx, Ly=Ly, site=site, bc=bc, bc_MPS=bc_MPS)
        
        CouplingModel.__init__(self, lat)

        # Heisenberg term
        c = 8 * 0.5 * ( 0.5 + eta ) # c = 8 * s * ( s + eta )
        self.add_coupling( c, 0, 'Sx', 0, 'Sx', [1, 0])
        self.add_coupling( c, 0, 'Sy', 0, 'Sy', [1, 0])
        self.add_coupling( c, 0, 'Sz', 0, 'Sz', [1, 0])
        self.add_coupling( c, 0, 'Sx', 0, 'Sx', [0, 1])
        self.add_coupling( c, 0, 'Sy', 0, 'Sy', [0, 1])
        self.add_coupling( c, 0, 'Sz', 0, 'Sz', [0, 1])
        
        # Ring-exchange term

        # (S_{i} . S_{i+y}) (S_{i+x} . S_{i+x+y}) 
        self.add_multi_coupling( 4.0 * J , [('Sx', [0,0], 0), ('Sx', [0,1], 0), ('Sx', [1,0], 0), ('Sx', [1,1], 0)])
        self.add_multi_coupling( 4.0 * J , [('Sy', [0,0], 0), ('Sy', [0,1], 0), ('Sx', [1,0], 0), ('Sx', [1,1], 0)])
        self.add_multi_coupling( 4.0 * J , [('Sz', [0,0], 0), ('Sz', [0,1], 0), ('Sx', [1,0], 0), ('Sx', [1,1], 0)])
        self.add_multi_coupling( 4.0 * J , [('Sx', [0,0], 0), ('Sx', [0,1], 0), ('Sy', [1,0], 0), ('Sy', [1,1], 0)])
        self.add_multi_coupling( 4.0 * J , [('Sy', [0,0], 0), ('Sy', [0,1], 0), ('Sy', [1,0], 0), ('Sy', [1,1], 0)])
        self.add_multi_coupling( 4.0 * J , [('Sz', [0,0], 0), ('Sz', [0,1], 0), ('Sy', [1,0], 0), ('Sy', [1,1], 0)])
        self.add_multi_coupling( 4.0 * J , [('Sx', [0,0], 0), ('Sx', [0,1], 0), ('Sz', [1,0], 0), ('Sz', [1,1], 0)])
        self.add_multi_coupling( 4.0 * J , [('Sy', [0,0], 0), ('Sy', [0,1], 0), ('Sz', [1,0], 0), ('Sz', [1,1], 0)])
        self.add_multi_coupling( 4.0 * J , [('Sz', [0,0], 0), ('Sz', [0,1], 0), ('Sz', [1,0], 0), ('Sz', [1,1], 0)])


        # (S_{i} . S_{i+x}) (S_{i+y} . S_{i+x+y}) 
        self.add_multi_coupling( 4.0 * J , [('Sx', [0,0], 0), ('Sx', [1,0], 0), ('Sx', [0,1], 0), ('Sx', [1,1], 0)])
        self.add_multi_coupling( 4.0 * J , [('Sy', [0,0], 0), ('Sy', [1,0], 0), ('Sx', [0,1], 0), ('Sx', [1,1], 0)])
        self.add_multi_coupling( 4.0 * J , [('Sz', [0,0], 0), ('Sz', [1,0], 0), ('Sx', [0,1], 0), ('Sx', [1,1], 0)])
        self.add_multi_coupling( 4.0 * J , [('Sx', [0,0], 0), ('Sx', [1,0], 0), ('Sy', [0,1], 0), ('Sy', [1,1], 0)])
        self.add_multi_coupling( 4.0 * J , [('Sy', [0,0], 0), ('Sy', [1,0], 0), ('Sy', [0,1], 0), ('Sy', [1,1], 0)])
        self.add_multi_coupling( 4.0 * J , [('Sz', [0,0], 0), ('Sz', [1,0], 0), ('Sy', [0,1], 0), ('Sy', [1,1], 0)])
        self.add_multi_coupling( 4.0 * J , [('Sx', [0,0], 0), ('Sx', [1,0], 0), ('Sz', [0,1], 0), ('Sz', [1,1], 0)])
        self.add_multi_coupling( 4.0 * J , [('Sy', [0,0], 0), ('Sy', [1,0], 0), ('Sz', [0,1], 0), ('Sz', [1,1], 0)])
        self.add_multi_coupling( 4.0 * J , [('Sz', [0,0], 0), ('Sz', [1,0], 0), ('Sz', [0,1], 0), ('Sz', [1,1], 0)])

        
        MPOModel.__init__(self, lat, self.calc_H_MPO())
        