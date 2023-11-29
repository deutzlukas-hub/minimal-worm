'''
Created on 28 Nov 2023

@author: amoghasiddhi
'''
import h5py 
import numpy as np
import matplotlib.pyplot as plt

# My packages
from parameter_scan import ParameterGrid

# Local import
from minimal_worm.experiments.undulation.dirs import create_storage_dir
from minimal_worm.experiments import PostProcessor

log_dir, sim_dir, sweep_dir = create_storage_dir()

def load_data(filename):
    '''
    Loads hdf5 simulation file
    '''

    h5_filepath = sweep_dir / filename
    h5 = h5py.File(str(h5_filepath), 'r')    
    PG_filepath = log_dir / h5.attrs['grid_filename']
    PG = ParameterGrid.init_pg_from_filepath(str(PG_filepath))

    return h5, PG

def plot_wobbling():
    
    h5 = ('raw_data_a_min=1.0_b_min=0.0032_f_range=2.5_'
        'c_min=0.4_c_max=1.6_c_step=0.1_'
        'lam_min=0.5_lam_max=2.0_lam_step=0.1_'
        'N=750_dt=0.01_T=5.0_pic_on=False.h5')
    
    h5, PG = load_data(h5)
    
    lam0_arr = PG.v_from_key('lam')
    c0_arr = PG.v_from_key('lam')
    
    r = h5['FS']['r']
    t = h5['t'][:]
        
    for k, r in enumerate(r):
    
        i, j = np.unravel_index(k, PG.shape)
        
        # Centre of mass trajectory
        r_com = r.mean(axis = -1)
        # Time average
        r_com_avg = r_com.mean(axis = 0)
                                  
        S = np.linalg.norm(r_com[-1, :] - r_com[0, :]) 
        eS, eW = PostProcessor.comp_swimming_direction(r_com)

        r_com_0 = r_com_avg - 0.5 * S * eS 
        r_com_1 = r_com_avg + 0.5 * S * eS 

        r_com_2 = r_com_avg - 0.1 * S * eW 
        r_com_3 = r_com_avg + 0.1 * S * eW 
                      
        plt.suptitle(f'lam0={lam0_arr[i]}, c0={c0_arr[j]}')              
        plt.plot(r_com[:, 1], r_com[:, 2], '-', c='k')
        plt.plot(r_com_avg[1], r_com_avg[2], 'o', c='r') 
        plt.plot([r_com_0[1], r_com_1[1]] , [r_com_0[2], r_com_0[1]], '--', c='r') 
        plt.plot([r_com_2[1], r_com_3[1]] , [r_com_2[2], r_com_3[1]], '--', c='r') 

        plt.show()
        
return

def plot_propulsive_force():
    
    h5 = ('raw_data_a_min=1.0_b_min=0.0032_f_range=2.5_'
        'c_min=0.4_c_max=1.6_c_step=0.1_'
        'lam_min=0.5_lam_max=2.0_lam_step=0.1_'
        'N=750_dt=0.01_T=5.0_pic_on=False.h5')
    
    h5, PG = load_data(h5)
    
    pass
    
    
    return


if __name__ == '__main__':
    
    pass
    
    
    
    

