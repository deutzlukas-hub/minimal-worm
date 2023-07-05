'''
Created on 10 Jan 2023

@author: lukas
'''
# Built-in
from typing import Tuple

# Third-party
import numpy as np
from scipy.integrate import trapz

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
        r_s = r_s / np.linalg.norm(r_s, axis = 1)
        
        # Angle of attack is the defined as the dot product of 
        # tangent and propulsion direction along the body
        phi = np.arccos(np.sum(t * e_p[None, :, None], axis = 1))
        
        # Average along body 
        avg_phi = np.mean(np.abs(phi), axis = 1)
            
        # Time average
        time_avg_phi = np.mean(avg_phi)
            
        return phi, avg_phi, time_avg_phi
        
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
    

    
        
                      
       
       

        
        
        
        
        
        
