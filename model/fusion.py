import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
import pdb

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FUSION_TYPES = ('Mul_1x1', 'Mul_1x1_Single', 'Mul_HxW+1x1', 'Conv', 'Mul_HxW')

class FusionModule(nn.Module):
    def __init__(self, params, fusion_order='TM'):
        super(FusionModule, self).__init__()

        assert fusion_order in ['TM', 'MT']

        self.fusion_modules = nn.ModuleList()

        for p in params:
            if p['type'].lower() == 'adaptive':
                ## in_ch, dims, out_plane, kernel_size, fusion_type
                self.fusion_modules += [ AdaptiveFusion(p['in_ch'], p['dims'], p['out_plane'], p['kernel_size'], p['fusion_type']) ]

            elif p['type'].lower() == 'static':
                ## dims, out_plane, kernel_size, fusion_type
                self.fusion_modules += [ StaticFusion(p['dims'], p['out_plane'], p['kernel_size'], p['fusion_type']) ]
                
            else:
                raise NotImplementedError
        
        self.fusion_order = fusion_order
        

    def forward(self, x0, y0, x, y):
        # x0, x: RGB / y0, y: T

        B0, T0, C0, H0, W0 = x0.shape
        B, T, C, H, W = x.shape

        if self.fusion_order == 'TM':

            x = x.view(B*T, C, H, W)
            y = y.view(B*T, C, H, W)

            h1 = self.fusion_modules[0](x0, y0, x)
            h2 = self.fusion_modules[1](x0, y0, y)

            h1 = self.fusion_modules[2](x0.mean(dim=1), y0.mean(dim=1), h1)
            h2 = self.fusion_modules[3](x0.mean(dim=1), y0.mean(dim=1), h2)

            h = self.fusion_modules[4](h1, h2)

        else:
            ### Eg. (adaptive, adaptive, sum), temporal
            h1_tt = []

            for tt in range(T):
                x0_tt = x0[:,tt,:,:,:]
                y0_tt = y0[:,tt,:,:,:]

                x_tt = x[:,tt,:,:,:]
                y_tt = y[:,tt,:,:,:]
            
                h1x_tt = self.fusion_modules[0](x0_tt, y0_tt, x_tt)
                h1y_tt = self.fusion_modules[1](x0_tt, y0_tt, y_tt)

                h1_tt.append( self.fusion_modules[2]( h1x_tt, h1y_tt ) )


            # # # ## Oraclefusion: multi frame
            h = self.fusion_modules[3]( h1_tt[0], h1_tt[1] )

            ## Oraclefusion: previous frame
            # h = self.fusion_modules[3]( h1_tt[0], None )

            ## Oraclefusion: current frame
            # h = self.fusion_modules[3]( None, h1_tt[1] )

        return h

class StaticFusion(nn.Module):
    def __init__(self, dims, out_plane, kernel_size, fusion_type, fusion_resolution=None):
        
        super(StaticFusion, self).__init__()
        
        assert fusion_type in FUSION_TYPES, \
            'Unknown fusion types. Should be one of them. {:s}.'.format( '_'.join(*FUSION_TYPES) )

        self.dims = dims
        self.fusion_resolution = fusion_resolution
        self.fusion_type = fusion_type
        self.out_plane = out_plane
        self.kernel_size = kernel_size
        self.padding = math.floor(kernel_size/2)

        if self.fusion_type == 'Mul_1x1':
            self.wx = nn.Parameter( torch.Tensor(dims, 1, 1) )
            self.wy = nn.Parameter( torch.Tensor(dims, 1, 1) )
            self.bx = nn.Parameter( torch.Tensor(dims, 1, 1) )
            self.by = nn.Parameter( torch.Tensor(dims, 1, 1) )

        if self.fusion_type == 'Mul_1x1_Single':
            self.wx = nn.Parameter( torch.Tensor(dims, 1, 1) )            
            self.bx = nn.Parameter( torch.Tensor(dims, 1, 1) )            

        elif self.fusion_type == 'Mul_HxW+1x1':
            self.wx1 = nn.Parameter( torch.Tensor(1, fusion_resolution[0], fusion_resolution[1]) )
            self.wx2 = nn.Parameter( torch.Tensor(dims, 1, 1) )

            self.wy1 = nn.Parameter( torch.Tensor(1, fusion_resolution[0], fusion_resolution[1]) )
            self.wy2 = nn.Parameter( torch.Tensor(dims, 1, 1) )

            self.bx1 = nn.Parameter( torch.Tensor(1, fusion_resolution[0], fusion_resolution[1]) )
            self.bx2 = nn.Parameter( torch.Tensor(dims, 1, 1) )
                
            self.by1 = nn.Parameter( torch.Tensor(1, fusion_resolution[0], fusion_resolution[1]) )
            self.by2 = nn.Parameter( torch.Tensor(dims, 1, 1) )

        elif self.fusion_type == 'Mul_HxW':
            self.wx = nn.Parameter( torch.Tensor(1, fusion_resolution[0], fusion_resolution[1]) )        
            self.wy = nn.Parameter( torch.Tensor(1, fusion_resolution[0], fusion_resolution[1]) )            

            self.bx = nn.Parameter( torch.Tensor(1, fusion_resolution[0], fusion_resolution[1]) )                            
            self.by = nn.Parameter( torch.Tensor(1, fusion_resolution[0], fusion_resolution[1]) )            
        
        elif self.fusion_type == 'Conv':
            self.wxbx = nn.Conv2d(dims, out_plane, kernel_size=kernel_size, padding=self.padding, stride=1)
            # self.wyby = nn.Conv2d(dims, dims, kernel_size=1, padding=0, stride=1)
        
        else:
            raise NotImplementedError

        self.reset_parameters()
                

    def reset_parameters(self):
        if self.fusion_type == 'Mul_1x1':
            init.normal_(self.wx.data, std=0.01)
            init.normal_(self.wy.data, std=0.01)
            self.bx.data.zero_()
            self.by.data.zero_()

        elif self.fusion_type == 'Mul_1x1_Single':
            init.normal_(self.wx.data, std=0.01)            
            self.bx.data.zero_()            

        elif self.fusion_type == 'Mul_HxW+1x1':
            init.normal_(self.wx1.data, std=0.01)
            init.normal_(self.wx2.data, std=0.01)
            init.normal_(self.wy1.data, std=0.01)
            init.normal_(self.wy2.data, std=0.01)
            self.bx1.data.zero_()
            self.bx2.data.zero_()
            self.by1.data.zero_()
            self.by2.data.zero_()

        elif self.fusion_type == 'Mul_HxW':
            init.normal_(self.wx.data, std=0.01)            
            init.normal_(self.wy.data, std=0.01)
            self.bx.data.zero_()            
            self.by.data.zero_()            

        elif self.fusion_type == 'Conv':
            init.normal_(self.wxbx.weight.data, std=0.01)
            self.wxbx.bias.data.zero_()
        else:
            raise NotImplementedError


    def forward(self, d1, d2, x):
        
        if self.fusion_type == 'Mul_1x1':
            return x * self.wx + self.bx + y * self.wy + self.by

        elif self.fusion_type == 'Mul_1x1_Single':
            gate = F.sigmoid( self.wx )
            return x * gate  + y * (1 - gate) + self.bx

        elif self.fusion_type == 'Mul_HxW+1x1':
            x1 = x  * self.wx1 + self.bx1   # Spatial
            x2 = x1 * self.wx2 + self.bx2   # Channel        
            y1 = y  * self.wy1 + self.by1   # Spatial
            y2 = y1 * self.wy2 + self.by2   # Channel
            return x2 + y2

        elif self.fusion_type == 'Mul_HxW':
            x1 = x  * self.wx + self.bx   # Spatial            
            y1 = y  * self.wy + self.by   # Spatial            
            return x1 + y1

        elif self.fusion_type == 'Conv':
            # return 0.5*self.wxbx(x) + 0.5*self.wyby(y)
            return 0.5*self.wxbx(x)

        else:
            raise NotImplementedError

    def __str__(self):
        s = ('{name}(kernel_size={kernel_size}, padding={padding}, fusion_type={fusion_type}, resolution={fusion_resolution}, dimensions={dims})')
        return s.format(name=self.__class__.__name__, **self.__dict__)