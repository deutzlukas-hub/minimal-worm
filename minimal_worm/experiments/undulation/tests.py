'''
Created on 22 Jun 2023

@author: amoghasiddhi
'''
import h5py
import numpy as np

from minimal_worm.experiments.undulation import log_dir, sim_dir, sweep_dir

def test_instantenous_power_balance(h5_filename):
    
    
    h5 = h5py.File(sweep_dir / h5_filename, 'r')
    
    V_dot = h5['FS']['V_dot']
    D_I_dot = h5['FS']['D_I_dot']
    D_F_dot = h5['FS']['D_F_dot']    
    W_dot = h5['FS']['W_dot']
    
    D_dot = D_I_dot + D_F_dot
        
    atol = 1e-1
    
    assert np.allclose(V_dot - (W_dot + D_dot), atol = atol)
    
        
if __name__ == '__main__':
    
    h5_a_b = ('raw_data_'
        'a_min=-2.0_a_max=3.0_step_a=0.2_'
        'b_min=-3.0_b_max=0.0_step_b=0.2_'
        'A=4.0_lam=1.0_N=250_dt=0.001.h5')
    
    test_instantenous_power_balance(h5_a_b)
     
    
