'''
Created on 22 Jun 2023

@author: amoghasiddhi
'''
import h5py
import numpy as np
import matplotlib.pyplot as plt

from minimal_worm.experiments.undulation import log_dir, sim_dir, sweep_dir

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
    
        
if __name__ == '__main__':
    
    # h5_a_b = ('raw_data_'
    #     'a_min=-2.0_a_max=3.0_step_a=0.2_'
    #     'b_min=-3.0_b_max=0.0_step_b=0.2_'
    #     'A=4.0_lam=1.0_N=250_dt=0.001.h5')
    
    # h5_a_b = ('raw_data_'
    #     'a_min=-2.0_a_max=3.0_step_a=1.0_'
    #     'b_min=-3.0_b_max=0.0_step_b=1.0_'
    #     'A=4.0_lam=1.0_N=250_dt=0.001.h5')
    
    h5_a_b = ('raw_data_'
        'a_min=-2.0_a_max=3.0_step_a=1.0_'
        'b_min=-3.0_b_max=0.0_step_b=1.0_'
        'A=4.0_lam=1.0_N=500_dt=0.0001.h5')
        
    test_instantenous_power_balance(h5_a_b)
     
    
