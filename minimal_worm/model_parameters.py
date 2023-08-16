'''
Created on 14 Jun 2023

@author: amoghasiddhi
'''

# Built-in
from argparse import ArgumentParser, BooleanOptionalAction, Namespace
from pathlib import Path

# Third-party 
from fenics import *
import numpy as np
import pint
import cv2
# Local import 
from minimal_worm.util import f2n

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
    param.add_argument('--p_from_physical', action = BooleanOptionalAction, default = False, 
        help = 'If true, p is calculated from given physical parameters')
    param.add_argument('--q_from_physical', action = BooleanOptionalAction, default = False, 
        help = 'If true, q is calculated from given physical parameters')
    param.add_argument('--a_from_physical', action = BooleanOptionalAction, default = False, 
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


def radius_shape_function(
        plot = False):
    '''
    C. elegans have cylindrical body shape that is tapered at the ends.
    
    Calculated the cross-sectional radius as the function of the body 
    coordinated from experimental images.         
    '''    
    import matplotlib.pyplot as plt
    from scipy.interpolate import splprep, splev
    from scipy.spatial.distance import cdist
    
            
    # Load the image
    image = cv2.imread(str(Path(__file__).parent  
        / 'c_elegans_body_shape.jpeg'))  # Replace with your image filename
                        
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    kernel = np.ones((10, 10), np.uint8)    
    _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[1]
    
    # Create an empty mask for the centerline
    centerline_mask = cv2.cvtColor(np.zeros_like(image), cv2.COLOR_BGR2GRAY)
    # Draw the selected contour on the centerline mask
    cv2.drawContours(centerline_mask, [contour], -1, 255, thickness=cv2.FILLED)
    skeleton = cv2.ximgproc.thinning(centerline_mask, thinningType=cv2.ximgproc.THINNING_GUOHALL)
    
    r_arr = np.column_stack(np.where(skeleton > 0))

    # Calculate pairwise distances
    distances = cdist(r_arr, r_arr)
    
    # Find the starting point (lowest y-coordinate)
    start_index = np.argmin(r_arr[:, 0])
    ordered_indices = [start_index]
    unprocessed_indices = list(range(len(r_arr)))
    unprocessed_indices.remove(start_index)
    
    # Order the points
    while unprocessed_indices:
        current_index = ordered_indices[-1]
        closest_index = unprocessed_indices[np.argmin(distances[current_index, unprocessed_indices])]
        ordered_indices.append(closest_index)
        unprocessed_indices.remove(closest_index)
    
    r_arr = r_arr[ordered_indices]
            
    x_cont_arr = contour.reshape(-1, 2)
    
    # Fit spline to contour
    tck, _ = splprep([x_cont_arr[:,0], x_cont_arr[:,1]], s=5e3, per=True)
    # Evaluate the spline on a finer parameterization
    u_arr = np.linspace(0, 1, num=1000)
    x_cont_arr = np.array(splev(u_arr, tck)).T
    
    # Fit spline to centreline
    tck, _ = splprep([r_arr[:-250,1], r_arr[:-250,0]], s=1e4, per=False)
    s_arr = np.linspace(0, 1, num=1000)
    r_arr = np.array(splev(s_arr, tck)).T
    
    # Identify points muscle onset
    idx1 = np.abs(s_arr - 0.05).argmin()
    idx2 = np.abs(s_arr - 0.95).argmin()
    
    R_arr = cdist(r_arr, x_cont_arr).min(axis = 1)
    phi_arr = R_arr / R_arr.max()

    # Define the degree of the polynomial
    degree = 8
    # Fit a polynomial of the specified degree
    coeff = np.polyfit(s_arr, phi_arr, degree)    
    # Generate the polynomial function using the fitted coefficient
    phi_fit = np.poly1d(coeff)
        
    # Convert polynomial to fenics Expressions         
    expr_str = "c0 * pow(x[0], 8) + c1 * pow(x[0], 7) + c2 * pow(x[0], 6) + c3 * pow(x[0], 5) + c4 * pow(x[0], 4) + c5 * pow(x[0], 3) + c6 * pow(x[0], 2) + c7 * x[0] + c8"
    phi_expr = Expression(expr_str, degree=8, 
        c0=coeff[0], c1=coeff[1], c2=coeff[2], c3=coeff[3], c4=coeff[4], 
        c5=coeff[5], c6=coeff[6], c7=coeff[7], c8=coeff[8])
    
    # Sainity check
    N = 100
    mesh = UnitIntervalMesh(N-1)
    V = FunctionSpace(mesh, 'Lagrange', 1)
    phi_func = Function(V)
    phi_func.assign(phi_expr)
    phi_fit_arr = phi_func.compute_vertex_values(mesh)    
    s_fit_arr = np.linspace(0, 1, N)
    assert np.allclose(phi_fit_arr, phi_fit(s_fit_arr))

    if plot:
        # Plot the contour using Matplotlib
        gs = plt.GridSpec(2, 1)
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])
    
        ax0.imshow(image)
        ax0.plot(x_cont_arr[:, 0], x_cont_arr[:, 1], c='g')
        ax0.plot(r_arr[:, 0], r_arr[:, 1], c='r')
        ax0.plot(r_arr[idx1, 0], r_arr[idx1, 1], 'o', 'b')
        ax0.plot(r_arr[idx2, 0], r_arr[idx2, 1], 'o', 'b')
        ax1.plot(s_arr, phi_arr)    
        ax1.plot(s_fit_arr, phi_fit_arr)            
        plt.show()
    
    return phi_expr

    
def RFT_spatial(param):
    '''
    For a non-elliptic shape functions, sbt predicts that the linear drag 
    coefficients depend on the body position s. 
    
    For C. elegans, the shape functions can be determined from microscope 
    images. Here, we show that the drag coefficient ratio remains approximately
    constant along the body. 
    
    The change in the tangential drag coefficient is small <= 10%, i.e. we approximate 
    it as constant. 
    '''
    
    import matplotlib.pyplot as plt

    
    eps = 2*param.R/param.L0 # slenderness parameter        
    assert eps.dimensionality == ureg.dimensionless.dimensionality
    
    phi_fit, _ = radius_shape_function()

    s_arr = np.linspace(0, 1, int(1e3), endpoint=True)
    
    sH = 0.1
    sT = 0.9
    
    sC_arr = s_arr[np.logical_and(sH < s_arr, s_arr < sT)]
    sH_arr = s_arr[s_arr <= sH]
    sT_arr = s_arr[s_arr >= sT]
            
    c_t_H_arr = np.ones_like(sH_arr)*2*np.pi / ( np.log( 4 / (phi_fit(sH) *eps) ) - 0.5)
    c_t_T_arr = np.ones_like(sT_arr)*2*np.pi / ( np.log( 4 / (phi_fit(sT) *eps) ) - 0.5)
    c_t_C_arr = 2*np.pi / ( np.log( 4*np.sqrt(sC_arr - sC_arr**2) / (phi_fit(sC_arr) *eps) ) - 0.5)

    plt.plot(sC_arr, np.log( 4*np.sqrt(sC_arr - sC_arr**2) / (phi_fit(sC_arr) *eps) ))
    plt.show()

    c_n_C_arr = 4*np.pi / ( np.log( 4*np.sqrt(sC_arr - sC_arr**2) / (phi_fit(sC_arr) *eps) ) + 0.5)

    c_t_arr = np.concatenate((c_t_H_arr, c_t_C_arr, c_t_T_arr))
    
    gs = plt.GridSpec(2, 1)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    
    ax0.plot(s_arr, c_t_arr)    
    ax1.plot(sC_arr, c_n_C_arr/c_t_C_arr)
    
    plt.show()
    
    return

        
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
        
        self.cache['g'] = self.I / (self.A * self.param.L0**2)  
        
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
    attr_keys = ['C', 'D', 'Y', 'g', 'a_c', 'a_T', 'p', 'q', 'a', 'b']
                                                    
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
        # All quantities should be dimensionless
        for k in ModelParameter.attr_keys:
            v = getattr(param, k)
            if isinstance(v, pint.Quantity):
                assert v.dimensionless, 'quantity must be dimensionless'
                v = v.magnitude
            elif isinstance(v, float):
                pass
            else:
                assert False, f'{k}={v}'           
            setattr(self, k, v)

        self.phi = param.phi
                
        self.S = 1.0 / (self.a * self.g) * ( 
            np.diag([self.a_c * self.p, self.a_c * self.p, 1])
        )                     
        self.S_tilde = self.b / (self.a * self.g) * ( 
            np.diag([self.a_c * self.q, self.a_c * self.q, 1])
        )  
        self.B  = 1.0 / self.a * ( 
            np.diag([1, 1, param.a_T * self.p])
        )                          
        self.B_tilde = self.b / self.a * (
            np.diag([1, 1, param.a_T * self.q])
        )  

        return
        
    def to_fenics(self):
                     
        C = Constant(self.C)
        D = Constant(self.D)
        Y = Constant(self.Y)
        S = Constant(self.S)
        S_tilde = Constant(self.S_tilde)
        B = Constant(self.B)
        B_tilde = Constant(self.B_tilde)
                
        if self.phi is not None:
            if self.phi == 'c_elegans':            
                phi = radius_shape_function()            
                D = phi**2 * D             
                S = phi**2 * S 
                S_tilde = phi**2 * S_tilde
                B = phi**4 * B 
                B_tilde = phi**4 * B_tilde
            else:
                assert False
                           
        return C, D, Y, S, S_tilde, B, B_tilde,        
    
if __name__ == '__main__':
    
    radius_shape_function(plot=True)    
    # param = parameter_parser()
    # default_param = param.parse_args([])
    # RFT_spatial(default_param)
    