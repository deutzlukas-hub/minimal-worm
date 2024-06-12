'''
Created on 26 Jun 2023

@author: amoghasiddhi
'''
# Local imports
import numpy as np
import h5py 
import matplotlib.pyplot as plt

from parameter_scan import ParameterGrid 

# Local imports
from minimal_worm.plot import plot_multiple_scalar_fields
from minimal_worm.experiments.undulation import log_dir, sweep_dir
from minimal_worm.experiments.undulation.dirs import fig_dir

def plot_chemograms(h5_filename):
    
    h5 = h5py.File(sweep_dir / h5_filename, 'r')     
    PG = ParameterGrid.init_pg_from_filepath(log_dir / h5.attrs['grid_filename'])
    
    b = 1e-2
    b_arr = PG.v_from_key('b')
    a_arr = PG.v_from_key('a')        
    idx = np.abs(b_arr-b).argmin()
        
    idx_arr = PG.flat_index(PG[:, idx])

    dir = fig_dir / 'chemograms' / f'strains_b={b}_a_min={a_arr[0]}_a_max={a_arr[-1]}'

    dir.mkdir(parents = True, exist_ok = True)

    for i, (idx, a) in enumerate(zip(idx_arr, a_arr)):
    
        k = h5['FS']['k'][idx, :, 0, :]
        sig2 = h5['FS']['sig'][idx, :, 1, :]
        sig3 = h5['FS']['sig'][idx, :, 2, :]
        r2 = h5['FS']['r_t'][idx, :, 1, :]
        r3 = h5['FS']['r_t'][idx, :, 2, :]
        
        plot_multiple_scalar_fields([k, sig2, sig3, r2, r3], 
            titles = ['$\kappa$','$\sigma_2$', '$\sigma_3$', '$u_2$', '$u_3$'],
            eps = 1e-6,
            cbar_formats = ['%.1f', '%.4f', '%.6f', '%.2f', '%.2f'])    
        
        plt.suptitle(f'a={a}')                
        plt.savefig(dir / f'{str(i).zfill(2)}.png')
        plt.close()
                
    return

def plot_instantaneous_power_balance(h5_filename: str):
    '''
    Plots the fluid dissipation rate, dissipation rate internal, time
    derivative of the potential and mechanical muscles work. 
    
    Plots the signed sum of all powers to demonstrate that energy is conserved.     
    '''    
    h5 = h5py.File(sweep_dir / h5_filename, 'r')    
    PG = ParameterGrid.init_pg_from_filepath(log_dir / h5.attrs['grid_filename'])

    t = h5['t'][:]     
    V_dot_arr = h5['FS']['V_dot'][:]
    D_I_dot_arr = h5['FS']['D_I_dot'][:]
    D_F_dot_arr = h5['FS']['D_F_dot'][:]   
    W_dot_arr = h5['FS']['W_dot'][:]
    
    plot_dir = fig_dir  / 'power_balance' / h5_filename
    
    plot_dir.mkdir(parents = True, exist_ok = True)

    a_arr = PG.v_mat_from_key('a').flatten()
    b_arr = PG.v_mat_from_key('b').flatten()
            
    for i, (a, b, V_dot, D_I_dot, D_F_dot, W_dot) in \
        enumerate(zip(a_arr, b_arr, V_dot_arr, D_I_dot_arr, D_F_dot_arr, W_dot_arr)):
        
        gs = plt.GridSpec(2, 1)
        # Create the first subplot
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])
        
        ax0.plot(t, V_dot, label=r'$\dot{V}$')
        ax0.plot(t, D_I_dot, label=r'$\dot{D}_I$')
        ax0.plot(t, D_F_dot, label=r'$\dot{D}_F$')
        ax0.plot(t, W_dot, label=r'$\dot{W}$')
        ax0.set_ylabel('Powers')
        ax0.set_title(f'$a={a}, b={b}$')        
        ax0.legend()
        
        # Create the second subplot
        balance = D_I_dot + D_F_dot + W_dot - V_dot
        ax1.plot(t, balance, color='red')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Power balance')
        
        plt.savefig(plot_dir / f'{str(i).zfill(2)}.png')
        plt.close()

    return

if __name__ == '__main__':
    
    h5_a_b = ('raw_data_'
        'a_min=-3.0_a_max=3.0_step_a=0.2_'
        'b_min=-3.0_b_max=0.0_step_b=0.2_'
        'A=4.0_lam=1.0_N=250_dt=0.001.h5')    
    
    dt = 0.01
    
    h5_a_b = ('raw_data_'
        'a_min=-2.0_a_max=3.0_step_a=1.0_'
        'b_min=-3.0_b_max=0.0_step_b=1.0_'
        f'A=4.0_lam=1.0_N=250_dt={dt}.h5')    
    
    #plot_chemograms(h5_a_b)
    plot_instantaneous_power_balance(h5_a_b)
    print('Finished!')
