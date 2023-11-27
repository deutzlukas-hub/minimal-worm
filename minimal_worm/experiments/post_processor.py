'''
Created on 10 Jan 2023

@author: lukas
'''
# Built-in
from typing import Tuple

# Third-party
import numpy as np
from scipy.integrate import trapz
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize_scalar
from scipy.interpolate import UnivariateSpline

class PostProcessor(object):
    '''
    Class to post process and analyse simulation results.
    
    n = number of time points
    N = number of bodypoints            
    '''    
    
    engery_names_from_power = {
        'D_I_dot': 'D_I', # Fluid dissipation rate     
        'D_F_dot': 'D_F', # Internal dissipation rate   
        'W_dot': 'W', # Mechanical muscle power
        'V_dot': 'V' # Elastic potential rate
    }
    
                    
    @staticmethod
    def comp_com_velocity(r: np.ndarray, t: np.ndarray, Delta_t: float = 0.0):
        '''
        Computes centre of mass velocity as a function of time
    
        :param r (n x 3 x N): centreline coordinates
        :param t (n): time stamps 
        :param Delta_t: crop time points t < Delta_t
        '''        

        idx_arr = t >= Delta_t
        t = t[idx_arr]
        r = r[idx_arr, :]                
        dt = t[1] - t[0]        
        
        # Com trajectory
        r_com = np.mean(r, axis = 2)            
        
        # Absolute com velocity as a function of time        
        v_com_vec = np.gradient(r_com, dt, axis=0, edge_order=1)    
        v_com = np.linalg.norm(v_com_vec, axis = 1)        
                
        # Time averaged absolute com velocity
        V = np.mean(v_com)
                        
        return r_com, v_com_vec, v_com, V, t

    @staticmethod
    def comp_centreline_curvature(r: np.ndarray):
        '''
        Computes centre curvature to compare against generalized
        curvature vector
    
        :param r (n x 3 x N): centreline coordinates
        '''
        #
        ds = 1 / (r.shape[2]-1)
        # Tangent vector
        t = np.gradient(r, ds, axis = 2, edge_order = 2)
        # Curvature vector
        k = np.gradient(t, ds, axis = 2, 
            edge_order = 2)
        
        t_cross_k = np.cross(t, k, axis = 1)
        # curvature sign
        sign = np.sign(t_cross_k[:, 0, :])
    
        # Scalar curvature
        k = sign*np.linalg.norm(k, axis = 1)
        
        return k 
        
    @staticmethod
    
    
    def com_pca(r_com: np.ndarray):
        '''
        Computes principal directions and components
        of the centreline centre of mass coordinates 
                        
        :param r_com (3 x n): centre of mass coordinates
        '''
        r_com = r_com.copy()        
        # Centre coordinates around mean 
        r_com_avg = r_com.mean(axis = 0)         
        r_com -= r_com_avg[None, :]

        C = np.matmul(r_com.T, r_com)            
        lam, w =  np.linalg.eig(C)

        # order from large to small
        idx_arr = lam.argsort()[::-1]
        lam = lam[idx_arr]        
        w = w[:, idx_arr]

        return lam, w, r_com_avg

    @staticmethod
    def comp_propulsion_direction(r_com: np.ndarray):
        '''
        Propulsion direction is estimated as the first principal axes
        of the com trajectory.
                
        :param r_com (3 x n):
        '''
        
        _, w, _ = PostProcessor.com_pca(r_com)
                
        # Approx swimming direction as first principal axis
        e_p = w[:, 0]
                
        # Make sure that the principal axis points 
        # in positive swimming direction         
        v = r_com[-1, :] - r_com[0, :]
        v = v / np.linalg.norm(v)        
        
        if np.dot(e_p, v) < 0:
            e_p = -e_p
                
        return e_p
                
    @staticmethod
    def comp_mean_swimming_speed(r: np.ndarray, t: np.ndarray, Delta_t: float = 0.0):
        '''
        Computes average swimming speed projected onto the first principle axis
        of centre of mass movement. This gives more accurate approximation of the 
        average speed in cases where the centre of mass wobbles around the swimming
        direction.
        
        :param r (n x 3 x N): centreline coordinates
        :param t (n): time stamps 
        :param Delta_t: crop time points t < Delta_t        
        '''        
        # crop initial transient
        idx_arr = t >= Delta_t
        dt = t[1]-t[0]
        r = r[idx_arr,:]
        t = t[idx_arr]
                
        r_com = r.mean(axis = 2)
        
        e_p = PostProcessor.comp_propulsion_direction(r_com)
                                             
        v_com_vec = np.gradient(r_com, dt, axis=0, edge_order=1)    
        
        # Project velocity on swimming direction
        U = np.sum(v_com_vec * e_p, axis = 1)                        
        # Take average
        U_avg = U.mean()
        
        return U_avg, U, t 

    @staticmethod
    def comp_mean_swimming_speed_simple(r: np.ndarray, t: np.ndarray, Delta_T: float):
        '''
        Computes average swimming speed by taking the start and endpoint
        of the centre of mass and dividing by the simulation time
        
        :param r (n x 3 x N): centreline coordinates
        :param t (n): time stamps 
        :param Delta_T: crop time points t < Delta_T                
        '''        
        
        # crop initial transient
        idx = np.abs(t - Delta_T).argmin()
        
        com_start = r[idx ,:].mean(axis = 1)
        com_end = r[-1 ,:].mean(axis = 1)
                                        
        U = np.linalg.norm(com_end - com_start) / (t[-1] - t[idx])
        
        return U 
    
    @staticmethod
    def comp_angle_of_attack(r: np.ndarray, t: np.ndarray, Delta_t: float = 0.0):
        '''
        Compute angle of attack
        
        :param r (n x 3 x N): centreline coordinates
        :param t (n x 1): timestamps
        :param Delta_t (float): crop timepoints t < Delta_T
        '''        
        # crop initial transient
        idx_arr = t >= Delta_t
        r = r[idx_arr,:]
        t = t[idx_arr]

        # Propulsion direcxtion
        r_com  = np.mean(r, axis = 2)        
        e_p = PostProcessor.comp_propulsion_direction(r_com)
                    
        # Compute tangent: Negative sign makes tangent point 
        # from tale to head in the general propulsion direction            
        r_s = - np.diff(r, axis = 2)
        # Normalize
        r_s = r_s / np.linalg.norm(r_s, axis = 1)[:, None, :]
        
        # Angle of attack is the defined as the dot product of 
        # tangent and propulsion direction along the body
        phi = np.arccos(np.sum(r_s * e_p[None, :, None], axis = 1))
        
        # Time avg
        avg_phi = phi.mean(np.abs(phi), axis = 0)
        
        # Average along body 
        avg_phi = np.mean(avg_phi, axis = 1)
        std_phi = np.std(avg_phi, axis = 1)
                                  
        return avg_phi, std_phi, phi
        
    @staticmethod
    def centreline_pca(r: np.ndarray):
        '''
        Computes principal directions and components
        of the centreline coordinates pooled over time 
        and body position
        
        :param r (np.ndarray (n x 3 x N)): centreline coordinates
        '''
                
        # Reformat into 2D array (3 x (n x N))
        r = np.swapaxes(r, 1, 2)                
        r = r.reshape((r.shape[0]*r.shape[1], r.shape[2]))
        
        # Centre coordinates around mean 
        x_avg = r.mean(axis = 0)         
        r -= x_avg[None, :]

        C = np.matmul(r.T, r)            
        lam, w =  np.linalg.eig(C)

        # order from large to small
        idx_arr = lam.argsort()[::-1]
        lam = lam[idx_arr]        
        w = w[:, idx_arr]
                    
        return lam, w, x_avg
                                       
    @staticmethod                 
    def powers_from_FS(FS):
        '''
        Returns dicitionary with powers from frame sequence
        
        :param FS (FrameSequenceNumpy): frame sequence
        '''            
        powers = {}
                
        for k, new_k in PostProcessor.rename_powers.items():
            
            powers[new_k] = getattr(FS, k)
        
        return powers
                                                                                           
    @staticmethod
    def comp_energy_from_power(
            power: np.ndarray, 
            t: np.ndarray, 
            t_start: Tuple[float, None] = None, 
            t_end: Tuple[float, None] = None):
        '''
        Computes energies from power for the given time interval
        
        :param powers (dict): power dictionary 
        :param dt (float): timestep
        '''

        idx_arr = np.arange(len(t))
                
        if t_start is not None:
            idx_arr = idx_arr[t >= t_start]
            t = t[idx_arr]        
        if t_end is not None:
            idx_arr = idx_arr[t <= t_end]
            t = t[idx_arr]
                   
        power = power[idx_arr]                                        
        dt = t[1] - t[0]                                           
               
        energy = trapz(power, dx=dt)
                          
        return energy
    
    @staticmethod
    def comp_optimal_c_and_wavelength(U, W, A, c_arr, lam_arr,
            levels = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
        '''
        Finds the shape factor c and wavelength lambda which minimizes 
        the mechanical muscle work for contour lines of equal swimming speed.
        
        To achieve a swimming speed U < U_max, which c and lambda 
        requires the least energy                                   
        '''

        U, W, A = U.T, W.T, A.T
        
        # Interpolate swimming speed surface with Spline
        U_interp = RectBivariateSpline(lam_arr, c_arr, U)    
        # Interpolate mechanical work surface with Spline        
        W_interp = RectBivariateSpline(lam_arr, c_arr, W)
        # Interpolate curvature amplitude
        A_interp = RectBivariateSpline(lam_arr, c_arr, A)
        
        # Define finer grid
        lam_arr = np.linspace(lam_arr.min(), lam_arr.max(), 100)
        c_arr = np.linspace(c_arr.min(), c_arr.max(), 100)
        
        # Remesh the interpolated surface to the finer grid
        U = U_interp(lam_arr, c_arr)
        W = W_interp(lam_arr, c_arr)
        A = A_interp(lam_arr, c_arr)
                                                                                                                      
        # Use lam and c where U is maximal as initial guess 
        j_max, i_max = np.unravel_index(U.argmax(), U.shape)
        lam_max_0, c_max_0 = lam_arr[i_max], c_arr[j_max]                    
        
        # Minimize -U_interp to find lam and c where U is maximal
        res = minimize(lambda x: -U_interp(x[0], x[1])[0], [lam_max_0, c_max_0], 
            bounds=[(lam_arr.min(), lam_arr.max()), (c_arr.min(), c_arr.max())])
                
        lam_max, c_max = res.x[0], res.x[1]                  
        A0_max = 2* np.pi * c_max / lam_max
        
        # Maximum swimming speed U_max 
        U_max = U_interp(lam_max, c_max)
        # Real curvature amplitude A_max             
        A_max = A_interp(lam_max, c_max)
        
        # Create contours for normalised swimming speed  
        U_over_U_max = U / U_max              
        LAM, C = np.meshgrid(lam_arr, c_arr)               
        CS = plt.contour(LAM, C, U_over_U_max.T, levels)    
        
        # Allocate arrays for optima on contours 
        lam_opt_arr = np.zeros_like(levels)
        c_opt_arr = np.zeros_like(levels)
        W_c_min_arr = np.zeros_like(levels)
        A_opt_arr = np.zeros_like(levels)
                                                        
        # Iterate over contour lines    
        for i, contours in enumerate(CS.collections):
            
            # W_min on contour line
            W_c_min = W.max() 
                
            # Iterate over paths which make up the current
            # contour line. If contour lines is a closed curve
            # then it has only one path                 
            for path in contours.get_paths():
    
                if not contours.get_paths():
                    assert False, 'Contour line has not path'
                        
                # Get points which make up the current path                                
                lam_on_path_arr = path.vertices[:, 0]
                c_on_path_arr = path.vertices[:, 1]
                                                
                if len(lam_on_path_arr) <= 3:
                    k = len(lam_on_path_arr)-1
                else:
                    k = 3
                                                                 
                # Compute B-spline representation of contour path
                tck, _ = splprep([lam_on_path_arr, c_on_path_arr], s=0, k = k)                
                
                # Find minimum work along the contour path
                result = minimize_scalar(lambda u: W_interp.ev(*splev(u, tck)), 
                    bounds=(0, 1), method='bounded')
                
                lam_min, c_min = splev(result.x, tck)                 
                W_p_min = W_interp.ev(lam_min, c_min)
                
                # If minimum work on path is smaller than work on any 
                # other path which belongs to the contour then set 
                # it contour minimum
                if W_p_min < W_c_min:
                    
                    W_c_min = W_p_min
                    lam_opt = lam_min
                    c_opt = c_min
                    A_opt = A_interp.ev(lam_min, c_min)
                            
            lam_opt_arr[i] = lam_opt
            c_opt_arr[i] = c_opt
            A_opt_arr[i] = A_opt
            W_c_min_arr[i] = W_c_min  
            
        plt.close()
        
        result = {}
        
        # refined mesh
        result['c_arr'] = c_arr
        result['lam_arr'] = lam_arr
        result['U'] = U
        result['W'] = W
        result['levels'] = levels
        
        # maximum speed kinematics
        result['lam_max'] = lam_max 
        result['c_max'] = c_max
        result['A0_max'] = A0_max                
        result['A_max']= A_max        
        result['U_max'] = U_max        
        
        # optimal kinematics on contours
        result['W_c_min_arr'] = W_c_min_arr
        result['lam_opt_arr'] = lam_opt_arr 
        result['c_opt_arr'] = c_opt_arr        
        result['A0_opt_arr'] = 2 * np.pi * c_opt_arr / lam_opt_arr         
        result['A_opt_arr'] = A_opt_arr
                                            
        return result
    
    @staticmethod 
    def comp_optimal_wavelength(
            U, 
            W, 
            f_arr, 
            lam_arr, 
            levels,
            N=1000):
        '''
        Finds the shape factor the wavelength lambda which minimizes 
        the mechanical muscle work
        
        To achieve a swimming speed U < U_max lower than the maximum 
        swimming speed, which lambda requires the least energy?                                   
        '''        
        
        # Define containers for results
        lam_max_arr = np.zeros(len(f_arr))    
        W_max_arr = np.zeros_like(lam_max_arr)
    
        lam_opt_mat = np.zeros((len(levels), len(f_arr)))
        W_opt_mat = np.zeros_like(lam_opt_mat)
                    
        U_mat, W_mat = np.zeros((len(f_arr), N)), np.zeros((len(f_arr), N))

        # Define finer grid to determine optimal wavelength 
        lam_refined_arr = np.linspace(lam_arr.min(), lam_arr.max(), N)   

        # Iterate over frequencies                                                        
        for i, _ in enumerate(f_arr):
            
            U_star_norm_arr = U[i, :] 
            W_star_arr = W[i, :]
                    
            U_fit = UnivariateSpline(lam_arr, U_star_norm_arr, s = 0.0)
            W_fit = UnivariateSpline(lam_arr, W_star_arr, s = 0.0)
            
            U_star_norm_refined_arr = U_fit(lam_refined_arr)
            W_star_norm_refined_arr = W_fit(lam_refined_arr) 
             
            U_mat[i, :] = U_star_norm_refined_arr
            W_mat[i, :] = W_star_norm_refined_arr
            
            # Iterate over levels and find optimal wavelength
            # to achieve level*U_max speed with minimal cost
            # for given frequency
            for j, level in enumerate(levels):            
                
                # Find wavelength for which U=level*U_max 
                zc_idx_arr = np.where(np.diff(
                    np.sign(U_star_norm_refined_arr - U_star_norm_refined_arr.max()*level)))[0]                
                
                # Find wavelength which minimizes 
                W_star_min, lam_opt = np.inf, np.nan
                    
                for idx in zc_idx_arr:                                 
                    
                    W_star_level = W_star_norm_refined_arr[idx]
                    
                    if W_star_level < W_star_min:
                        lam_opt = lam_refined_arr[idx]
                        W_star_min = W_star_level 
                        
                lam_opt_mat[j, i] = lam_opt
                W_opt_mat[j, i] = W_star_min
                            
            max_idx = U_star_norm_refined_arr.argmax()
            lam_max_arr[i] = lam_refined_arr[max_idx]
            W_max_arr[i] = W_star_norm_refined_arr[max_idx]
    
        return lam_max_arr, lam_opt_mat, W_max_arr, W_opt_mat, U_mat, W_mat, lam_refined_arr

    @staticmethod    
    def physical_2_dimless_parameters(param, **kwargs):
        '''
        Converts physical to dimensionless parameters a and b
        '''    
        
        # optional parameters
        opt_args = [
            'L0', # body length
            'R', # cross-sectional radius
            'T_c', # characteristic time
            'E', # Young's modulus
            'eta', # Extensional viscosity
            'mu', # Fluid viscosity
            'c_t', # tangential drag coefficient
        ]
        
        for arg in opt_args:
            if arg not in kwargs:
                #TODO
                kwargs[arg] = param[arg]
        
        R, L0, T_c = kwargs['R'], kwargs['L0'], kwargs['T_c']
        E, eta = kwargs['E'], kwargs['eta']
        mu, c_t = kwargs['mu'], kwargs['c_t'] 
        
        # second moment of area                
        I = 0.25 * np.pi * R**4
                        
        # Elastic time scale                
        tau = (mu * L0**4 * c_t) / (E * I) 
        # Viscous time scale
        xi = eta / E
        # Dimless elastic time scale ratio 
        a = tau / T_c
        # Dimless viscous time scale ratio 2        
        b = xi / T_c     
        
        return a, b
    
    
    @staticmethod
    def U_star_to_U(U_star, f: float, L0: float):
        '''
        Convert dimensionless swimming speed to physical units meter per seconds
        '''                
        return U_star * f * L0
    
    @staticmethod
    def E_star_to_E(E_star, mu: float, f: float, L0: float):
        '''
        Convert dimensionless energy to physical units Jouls
        '''        
        
        return E_star * mu * f * L0**3
    

    
        
                      
       
       

        
        
        
        
        
        
