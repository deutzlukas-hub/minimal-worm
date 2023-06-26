'''
Created on 14 Jun 2023

@author: amoghasiddhi
'''

# Built-in
from fenics import Constant
from argparse import ArgumentParser, BooleanOptionalAction, Namespace
# Third-party 
import numpy as np
import pint

# Default unit registry
ureg = pint.UnitRegistry() 

def parameter_parser():

    param = ArgumentParser(description = 'dimless-model-parameter')
        
    param.add_argument('--from_physical', action = BooleanOptionalAction, default = False, 
        help = 'If true, dimensionless parameters are calculated from given physical parameters')
    
    # Characteristic time scale
    param.add_argument('--T_c', type = lambda s: float(s)*ureg.second,
        default = 1.0 * ureg.second, help = 'Characteristic time scale')
    # Geometric parameter    
    param.add_argument('--L0', type = lambda s: float(s)*ureg.meter, 
        default = 1130 * 1e-6 * ureg.meter, help = 'Natural worm length')
    param.add_argument('--R', type = lambda s: float(s)*ureg.meter, 
        default = 32 * 1e-6 * ureg.meter, help = 'Maximum worm radius')                
    # Material parameter
    param.add_argument('--mu', type = lambda s: float(s)*ureg.pascal*ureg.second, 
        default = 1e-3*ureg.pascal*ureg.second, help = 'Fluid viscosity')                    
    param.add_argument('--E', type = lambda s: float(s)*ureg.pascal, 
        default = 1.21e5*ureg.pascal, help = "Young's modulus")
    param.add_argument('--eta', type = lambda s: float(s)*ureg.pascal*ureg.second, 
        default = 1e-2 * 1.21e5*ureg.pascal*ureg.second, help = 'Extensional viscosity')
    
    # Calculate default dimensionless parameter from default phyiscal parameter
    default_param = param.parse_args([])
    physical_to_dimless_parameters(default_param)
                                                          
    # Dimensionless model parameters                         
    param.add_argument('--C', type = float, default = default_param.C, 
        help = "Normal and tangential linear drag coefficient ratio")
    param.add_argument('--c_t', type = float, default = default_param.c_t, 
        help = 'Tangential linear drag coefficient')
    param.add_argument('--D', type = float, default = default_param.D, 
        help = 'Tangential angular and linear drag coefficient ratio ')
    param.add_argument('--Y', type = float, default = default_param.Y, 
        help = 'Normal and tangential angular drag coefficient ratio')
    param.add_argument('--a', type = float, default = default_param.a, 
        help = 'Elastic and undulation time scale ratio')
    param.add_argument('--b', type = float, default = default_param.b, 
        help = 'Viscous time scale ratio')
    param.add_argument('--p', type = float, default = 1.0 / 3.0, 
        help = 'Shear viscosity over extensional viscosity ratio')
    param.add_argument('--q', type = float, default = 1.0 / 3.0, 
        help = 'Shear viscosity over extensional viscosity ratio')
    param.add_argument('--g', type = float, default = default_param.g, 
        help = 'Cross-sectional area over second moment of area')
    param.add_argument('--a_c', type = float, default = 1.0, 
        help = 'shear correction factors')
    param.add_argument('--a_T', type = float, default = 1.0, 
        help = 'torsional correction factor')
    param.add_argument('--phi', default = None, 
        help = 'Cross-sectional radius shape function')

    # Simulation parameter         
    param.add_argument('--T', type = float, default = 5, 
        help = 'Dimensionless simulation time in units of the characteristic time T_c')    
    param.add_argument('--N', type = int, default = 250, 
        help = 'Number of centreline points')
    param.add_argument('--dt', type = float, default = 0.001, 
        help = 'Time step')
    param.add_argument('--N_report', type = lambda v: None if v.lower()=='none' else int(v), 
        default = None, help = 'Save simulation results for N_report centreline points')
    param.add_argument('--dt_report', type = lambda v: None if v.lower()=='none' else float(v), 
        default = None, help = 'Save simulation results only every dt_report time step')

    # Solver parameter
    param.add_argument('--fdo', type = int, default = 2, 
        help = 'Order of finite backwards difference')
                
    return param    
    
def RFT(param: Namespace):
    '''
    Calculates dimensionless drag coefficients as predicted by 
    resistive-force theory from geometric parameters     
    '''
            
    a = 2*param.R/param.L0 # slenderness parameter        

    assert a.dimensionality == ureg.dimensionless.dimensionality
    
    # Linear drag coefficients
    c_t = 2 * np.pi / (np.log(2/a) - 0.5)
    c_n = 4 * np.pi / (np.log(2/a) + 0.5)

    # Angular drag coefficients
    y_t = 0.25 * np.pi * a**2
    y_n = np.pi * a**2
                      
    D = y_t / c_t 
    C = c_n / c_t 
    Y = y_n / y_t 
        
    return c_t, C, D, Y   
    
def physical_to_dimless_parameters(param: Namespace):
    '''
    Converts physical model parameters to dimensionless parameters    
    '''
                                                
    # Drag coefficient ratio
    c_t, C, D, Y = RFT(param)
    
    _A = np.pi * param.R**2
    I = 0.25 * np.pi * param.R**4
                    
    # Cross-sectional area divided by second moment of area 
    g =  I / (_A * param.L0**2)                         
    # Elastic time scale                
    tau = (param.mu * param.L0**4 * c_t) / (param.E * I) 
    # Viscous time scale
    xi = param.eta / param.E
    # Dimless elastic time scale ratio 
    a = tau / param.T_c
    # Dimless viscous time scale ratio 2        
    b = xi / param.T_c 
        
    # Sanity check                                                                   
    assert a.dimensionality == ureg.dimensionless.dimensionality
    assert b.dimensionality == ureg.dimensionless.dimensionality
    assert g.dimensionality == ureg.dimensionless.dimensionality

    param.c_t = c_t
               
    param.C, param.D, param.Y = C, D, Y
    param.a, param.b, param.g = a, b, g
         
class ModelParameter():
    '''
    Dimensionless model parameters
    '''
    attr_keys = ['C', 'D', 'Y', 'phi']
                                                    
    def __init__(self, param: Namespace):            
        '''        
        :param K: Linear drag coefficient ratio
        :param y: Ratio between tangential linear and angular drag coefficient
        :param Y: Angular drag coefficient ratio
        :param a: External viscoelastic time scale ratio
        :param b: Internal viscoelastic time scale ratio
        :param param: Shear modulus G over Young's modulus 
        :param q: Shear viscosity nu over extensional viscosity eta
        :param g: Dimensionless cross-sectional area over second moment of area 
        :param a_c: Shear correction factor
        :param a_T: Torsional shear correction factor
        :param phi: Cross-section radius shape function  
        '''
                           
        for k in ModelParameter.attr_keys:
            v = getattr(param, k)
            if isinstance(v, float):
                v = pint.Quantity(v, 'dimensionless')            
            setattr(self, k, v)
                                                
        self.S = np.diag([param.a_c * param.p, 
            param.a_c * param.p, 1]) / (param.a * param.g)        
        self.S_tilde = param.b * np.diag([param.a_c * param.q, 
            param.a_c * param.q, 1]) / (param.a * param.g) 
        self.B  = np.diag([1, 1, 
            param.a_T * param.p]) / param.a                         
        self.B_tilde = param.b * np.diag([1, 1, 
            param.a_T * param.q]) / param.a 

        return
        
    def to_fenics(self):
                     
        C = ModelParameter.to_constant(self.C)
        D = ModelParameter.to_constant(self.D)
        Y = ModelParameter.to_constant(self.Y)
        S = ModelParameter.to_constant(self.S)
        S_tilde = ModelParameter.to_constant(self.S_tilde)
        B = ModelParameter.to_constant(self.B)
        B_tilde = ModelParameter.to_constant(self.B_tilde)
        
        if self.phi is not None:
            S, S_tilde = self.phi**2 * S, self.phi**2 * S_tilde
            B, B_tilde  = self.phi**4 * B, self.phi**4 * B_tilde 
                    
        return C, D, Y, S, S_tilde, B, B_tilde       
    
    @staticmethod
    def to_constant(v):
        
        if isinstance(v, pint.Quantity):
            v = Constant(v.magnitude)
        else: 
            v = Constant(v)
        return v
                