'''
Created on 22 Jun 2023

@author: amoghasiddhi
'''

# Third-party
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
import pint 

from parameter_scan import ParameterGrid

# Local
from minimal_worm.experiments.undulation import log_dir, sim_dir, sweep_dir
from minimal_worm.experiments import PostProcessor

def to_quantities(param):

    for k, v in param.items():
        if isinstance(v, list):
            if len(v)==2:
                try:
                    unit = ureg(v[1])
                    param[k] = v[0]*unit                     
                except:
                    continue

ureg = pint.UnitRegistry()

def test_instantenous_power_balance(h5_filename):
    
    
    h5 = h5py.File(sweep_dir / h5_filename, 'r')
    t = h5['t'][:]
    idx_arr = t >= 0
    
    V_dot = h5['FS']['V_dot'][:, idx_arr]
    D_I_dot = h5['FS']['D_I_dot'][:, idx_arr]
    D_F_dot = h5['FS']['D_F_dot'][:, idx_arr]   
    W_dot = h5['FS']['W_dot'][:, idx_arr]
    
    D_dot = D_I_dot + D_F_dot
    
    assert np.all(D_dot <= 0)
                
    atol = 1e-1
    
    for v_dot, w_dot, d_dot in zip(V_dot, W_dot, D_dot):
        
        plt.plot(v_dot, label = r'$\dot{V}$')
        plt.plot(w_dot, label = r'$\dot{W}$')
        plt.plot(d_dot, label = r'$\dot{D}$')
        plt.plot(v_dot - w_dot - d_dot, label = r'$0$')        
        plt.legend()
        plt.show()

    assert np.allclose(V_dot - (W_dot + D_dot), 0, atol = atol)
    
    return
    
def test_W_over_S(h5_filename: str):
    
    h5 = h5py.File(sweep_dir / h5_filename, 'r')

    h5 = h5py.File(sweep_dir / h5_filename, 'r')
    PG = ParameterGrid.init_pg_from_filepath(log_dir / h5.attrs['grid_filename'])

    to_quantities(PG.base_parameter)

    W = h5['energies']['W'][:]
    U = h5['U'][:]

    log_a_arr = np.log10(PG.v_from_key('a'))
    log_b_arr = np.log10(PG.v_from_key('b'))

    U_star_interp = RectBivariateSpline(log_a_arr, log_b_arr, U)
    W_star_interp = RectBivariateSpline(log_a_arr, log_b_arr, W)

    mu_arr = np.array([1e-3, 1e1]) * ureg.pascal * ureg.second

    f_arr_list = [
        10**np.linspace(-0.5, 2, int(3*1e1)),
        10**np.linspace(-1, np.log10(3), int(3*1e1))
    ]

    L0 = PG.base_parameter['L0']

    for mu, f_arr in zip(mu_arr, f_arr_list):

        f_arr = f_arr / ureg.second
            
        a_arr, b_arr = PostProcessor.physical_2_dimless_parameters(
            PG.base_parameter, mu=mu, T_c = 1.0 / f_arr)

        log_a_arr, log_b_arr = np.log10(a_arr), np.log10(b_arr) 

        U_star = U_star_interp.ev(log_a_arr, log_b_arr)
        W_star = W_star_interp.ev(log_a_arr, log_b_arr)

        U = PostProcessor.U_star_to_U(U_star, f_arr, L0) 
        W = PostProcessor.E_star_to_E(W_star, mu, f_arr, L0)
        
        W_over_S = f_arr * W / U 
        
        assert np.allclose(W_over_S, f_arr * mu * L0**2 * W_star / U_star)
    
    print('Passed test: W over S')
    
    return
            
if __name__ == '__main__':
    
    h5_a_b = ('analysis_'
        'a_min=-2.0_a_max=3.0_step_a=0.2_'
        'b_min=-3.0_b_max=0.0_step_b=0.2_'
        'A=4.0_lam=1.0_T=5.0_'
        'N=250_dt=0.001.h5')
    
    test_W_over_S(h5_a_b)
    
    # h5_a_b = ('raw_data_'
    #     'a_min=-2.0_a_max=3.0_step_a=1.0_'
    #     'b_min=-3.0_b_max=0.0_step_b=1.0_'
    #     'A=4.0_lam=1.0_N=250_dt=0.001.h5')
        
    # h5_a_b = ('raw_data_'
    #     'a_min=-2.0_a_max=3.0_step_a=1.0_'
    #     'b_min=-3.0_b_max=0.0_step_b=1.0_'
    #     'A=4.0_lam=1.0_N=500_dt=0.0001.h5')
        
    #test_instantenous_power_balance(h5_a_b)
    
