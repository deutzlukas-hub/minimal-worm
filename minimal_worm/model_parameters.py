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

DIMLESS_PARAM_KEYS = ['g', 'C', 'Y', 'D', 'p', 'q', 'a', 'b']

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
    param.add_argument('--G', type = lambda s: float(s)*ureg.pascal, 
        default = 1.21e5 / 3 *ureg.pascal , help = "Shear modulus")
    param.add_argument('--eta', type = lambda s: float(s)*ureg.pascal*ureg.second, 
        default = 1e-2 * 1.21e5*ureg.pascal*ureg.second, help = 'Extensional viscosity')
    param.add_argument('--nu', type = lambda s: float(s)*ureg.pascal*ureg.second, 
        default = 1e-2 / 3 * 1.21e5*ureg.pascal*ureg.second, help = 'Shear viscosity')

    # Calculate default dimensionless parameter from default phyiscal parameter
    default_param = param.parse_args([])
    
    TDL = ToDimless(default_param)
                                                              
    # Dimensionless model parameters                         
    param.add_argument('--C', type = float, default = TDL.C, 
        help = "Normal and tangential linear drag coefficient ratio")
    param.add_argument('--c_t', type = float, default = TDL.c_t, 
        help = 'Tangential linear drag coefficient')
    param.add_argument('--D', type = float, default = TDL.D, 
        help = 'Tangential angular and linear drag coefficient ratio ')
    param.add_argument('--Y', type = float, default = TDL.Y, 
        help = 'Normal and tangential angular drag coefficient ratio')
    param.add_argument('--a', type = float, default = TDL.a, 
        help = 'Elastic and undulation time scale ratio')
    param.add_argument('--b', type = float, default = TDL.b, 
        help = 'Viscous time scale ratio')
    param.add_argument('--p', type = float, default = TDL.p, 
        help = 'Shear viscosity over extensional viscosity ratio')
    param.add_argument('--q', type = float, default = TDL.q, 
        help = 'Shear viscosity over extensional viscosity ratio')
    param.add_argument('--g', type = float, default = TDL.g, 
        help = 'Cross-sectional area over second moment of area')
    param.add_argument('--a_c', type = float, default = 1.0, 
        help = 'shear correction factors')
    param.add_argument('--a_T', type = float, default = 1.0, 
        help = 'torsional correction factor')
    param.add_argument('--phi', default = None, 
        help = 'Cross-sectional radius shape function')
    param.add_argument('--g_from_physical', action = BooleanOptionalAction, default = False, 
        help = 'If true, g is calculated from given physical parameters')
    param.add_argument('--C_from_physical', action = BooleanOptionalAction, default = False, 
        help = 'If true, C is calculated from physical parameters')
    param.add_argument('--Y_from_physical', action = BooleanOptionalAction, default = False, 
        help = 'If true, Y is calculated from given physical parameters')
    param.add_argument('--D_from_physical', action = BooleanOptionalAction, default = False, 
        help = 'If true, D is calculated from given physical parameters')
    param.add_argument('--p_from_phyiscal', action = BooleanOptionalAction, default = False, 
        help = 'If true, p is calculated from given physical parameters')
    param.add_argument('--p_from_physical', action = BooleanOptionalAction, default = False, 
        help = 'If true, q is calculated from given physical parameters')
    param.add_argument('--q_from_physical', action = BooleanOptionalAction, default = False, 
        help = 'If true, a is calculated from given physical parameters')
    param.add_argument('--b_from_physical', action = BooleanOptionalAction, default = False, 
        help = 'If true, b is calculated from given physical parameters')

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
    
    
class ToDimless():    
    '''
    Converts phyiscal to dimensionless parameters
    '''
    def __init__(self, param: Namespace):
                
        self.param = param
        self.cache = {}
    
    @property
    def alpha(self):
        '''
        Slenderness parameter
        '''
        
        if 'alpha' in self.cache:
            return self.cache['alpha']
        
        self.cache['alpha'] = 2*self.param.R/self.param.L0
        
        return self.cache['alpha']
                             
    @property
    def A(self):
        '''
        Cross-sectional radius
        '''
                
        if 'A' in self.cache:
            return self.cache['A']
        
        self.cache['A'] = np.pi * self.param.R**2
    
        return self.cache['A']
    
    @property
    def I(self):
        '''
        Second area moment of inertia
        '''
        
        if 'I' in self.cache:
            return self.cache['I']
        
        self.cache['I'] = 0.25 * np.pi * self.param.R**4
    
        return self.cache['I']

    @property
    def g(self):
        '''
        Dimensionless second moment of area over dimensionless
        cross-sectional area  
        '''
        
        if 'g' in self.cache:
            return self.cache['g']
        
        self.cache['g'] = self.I * (self.A * self.param.L0**2)  
        
        return self.cache['g']
        
    @property        
    def c_n(self):
        '''
        Normal linear drag coefficient
        '''
                
        if 'c_n' in self.cache:
            return self.cache['c_n']
        
        self.cache['c_n'] = 4 * np.pi / (np.log(2/self.alpha) + 0.5)
        
        return self.cache['c_n']

    @property        
    def c_t(self):
        '''
        Tangential linear drag coefficient
        '''
        
        if 'c_t' in self.cache:
            return self.cache['c_t']
        
        self.cache['c_t'] = 2 * np.pi / (np.log(2/self.alpha) - 0.5)
        
        return self.cache['c_t']

    @property        
    def y_t(self):
        '''
        Tangential angular drag coefficient
        '''
        
        if 'y_t' in self.cache:
            return self.cache['y_t']
        
        self.cache['y_t'] = 0.25 * np.pi * self.alpha**2
        
        return self.cache['y_t']

    @property        
    def y_n(self):
        '''
        Normal angular drag coefficient
        '''
        
        if 'y_n' in self.cache:
            return self.cache['y_n']
        
        self.cache['y_n'] = np.pi * self.alpha**2
        
        return self.cache['y_n']
                        
    @property        
    def C(self):
        '''
        Linear drag coefficient ratio
        '''
                
        return self.c_n / self.c_t 
        
    @property        
    def Y(self):
        '''
        Angular drag coefficient ratio
        '''
        
        return self.y_n / self.y_t
        
    @property        
    def D(self):
        '''
        Tangential angular over linear drag coefficient ratio        
        '''
        
        return self.y_t / self.c_t

    @property        
    def p(self):
        '''
        Tangential angular over linear drag coefficient ratio        
        '''
        
        return self.param.G / self.param.E

    @property        
    def q(self):
        '''
        Tangential angular over linear drag coefficient ratio        
        '''
        
        return self.param.nu / self.param.eta

    @property
    def tau(self):
        '''
        Relaxation time scale
        '''
        
        return (self.param.mu*self.c_t*self.param.L0**4) / (self.param.E*self.I)

    @property
    def xi(self):
        '''
        Internal viscosity time scale
        '''
        
        return self.param.eta / self.param.E
        
    @property
    def a(self):
        '''
        Relaxation time scale ratio
        '''
                        
        return self.tau / self.param.T_c
            
    @property
    def b(self):
        '''
        Viscous time scale ratio
        '''
        
        return self.xi / self.param.T_c

            
def physical_to_dimless_parameters(param: Namespace):
    '''
    Converts physical model parameters to dimensionless parameters    
    '''
    TDL = ToDimless(param)
    
    for key in DIMLESS_PARAM_KEYS:
        
        # If from_physical is true, then all dimless parameters 
        # are calculated from the given physcial parameters 
        if param.from_physical:
            setattr(param , key, getattr(TDL, key))                
        # If from phyiscal is false, then only those dimless 
        # parameters are calculated from the given physical 
        # parameters for which key_from_physical was set to True           
        elif getattr(param, f'{key}_from_physical'):
            setattr(param , key, getattr(TDL, key))                
        
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
                