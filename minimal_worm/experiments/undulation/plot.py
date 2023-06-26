'''
Created on 19 Jun 2023

@author: amoghasiddhi
'''

# Third-party
import numpy as np
from scipy.interpolate import RectBivariateSpline
import h5py 
import pint
import matplotlib.pyplot as plt

from parameter_scan import ParameterGrid 

# Local imports
from minimal_worm.experiments import PostProcessor
from minimal_worm.plot import plot_multiple_scalar_fields
from minimal_worm.experiments.undulation import log_dir, sweep_dir
from minimal_worm.experiments.undulation.dirs import fig_dir

ureg = pint.UnitRegistry()

cm_dict = {
    'U': plt.cm.plasma,
    'D': plt.cm.hot,
    'W': plt.cm.hot,
    'k_norm': plt.cm.winter}

def to_quantities(param):

    for k, v in param.items():
        if isinstance(v, list):
            if len(v)==2:
                try:
                    unit = ureg(v[1])
                    param[k] = v[0]*unit                     
                except:
                    continue
               
def plot_W_D(h5_filepath):
        
    h5 = h5py.File(sweep_dir / h5_filepath, 'r')
    PG = ParameterGrid.init_pg_from_filepath(log_dir / h5.attrs['grid_filename'])

    gs = plt.GridSpec(3, 1)
    ax00 = plt.subplot(gs[0, 0])
    ax10 = plt.subplot(gs[1, 0])
    ax20 = plt.subplot(gs[2, 0])

    log_a_arr = np.log10(PG.v_from_key('a'))
    log_b_arr = np.log10(PG.v_from_key('b'))

    log_a_grid, log_b_grid =np.meshgrid(log_a_arr, log_b_arr)

    W = h5['energies']['W'][:].T
    D_I = h5['energies']['D_I'][:].T
    D_F = h5['energies']['D_I'][:].T
    D = D_I + D_F
    assert np.all(W >= 0)
    assert np.all(D <= 0)
            
    abs_D = np.abs(D_I + D_F)    
    norm_W = W / W.max()
    norm_abs_D = abs_D / abs_D.max()
    
    levels = 6
    
    CS = ax00.contourf(log_a_grid, log_b_grid, np.log10(norm_W), 
        levels = levels, cmap = cm_dict['D'])
    ax00.contour(log_a_grid, log_b_grid, np.log10(norm_W), 
        levels = levels, linestyles = '-', colors = ('k',))        
    plt.colorbar(CS, ax = ax00, orientation = 'horizontal')

    #levels = np.arange(0, 1.01, 0.2)
    CS = ax10.contourf(log_a_grid, log_b_grid, np.log10(norm_abs_D), 
        levels = levels, cmap = cm_dict['D'])
    ax10.contour(log_a_grid, log_b_grid, np.log10(norm_abs_D), 
        levels = levels, linestyles = '-', colors = ('k',))        
    plt.colorbar(CS, ax = ax10, orientation = 'horizontal')

    #levels = np.arange(0, 1.01, 0.2)
    CS = ax20.contourf(log_a_grid, log_b_grid, np.log10(np.abs(W - D) / W), 
        levels = levels, cmap = cm_dict['U'])
    ax20.contour(log_a_grid, log_b_grid, np.log10(np.abs(W - D) / W), 
        levels = levels, linestyles = '-', colors = ('k',))        
    plt.colorbar(CS, ax = ax20, orientation = 'horizontal')

    plt.show()
    
    return
    

def plot_speed_curvature_norm_and_amplitude(h5_filename):
    '''
    Plot swimming speed, curvature and amplitude
    '''    
    h5 = h5py.File(sweep_dir / h5_filename, 'r')
    PG = ParameterGrid.init_pg_from_filepath(log_dir / h5.attrs['grid_filename'])

    _A = PG.base_parameter['_A']

    #BP = to_quantities(PG.base_parameter)
                          
    # Swimming speed
    U = h5['U'][:].T  
    k_norm = h5['k_norm'][:].T  
    sig_norm = h5['sig_norm'][:].T  

    log_a_arr = np.log10(PG.v_from_key('a'))
    log_b_arr = np.log10(PG.v_from_key('b'))

    a_grid, b_grid =np.meshgrid(log_a_arr, log_b_arr)

    gs = plt.GridSpec(2, 2)
    ax00 = plt.subplot(gs[0,0])
    ax01 = plt.subplot(gs[0,1])
    ax10 = plt.subplot(gs[1,0])
    ax11 = plt.subplot(gs[1,1])

    
    levels = np.arange(0, 1.01, 0.2)
    
    CS = ax00.contourf(a_grid, b_grid, U / U.max(), 
        levels = levels, cmap = cm_dict['U'])
    ax00.contour(a_grid, b_grid, U / U.max(), 
        levels = levels, linestyles = '-', colors = ('k',))        
    plt.colorbar(CS, ax = ax00, orientation = 'horizontal')

    CS = ax01.contourf(a_grid, b_grid, k_norm / _A, 
        levels = len(levels), cmap = cm_dict['k_norm'])
    ax01.contour(a_grid, b_grid, k_norm / _A, 
        levels = len(levels), linestyles = '-', colors = ('k',))        
    plt.colorbar(CS, ax = ax01, orientation = 'horizontal')

    CS = ax10.contourf(a_grid, b_grid, sig_norm / sig_norm.max() , 
        levels = len(levels) + 1, cmap = cm_dict['k_norm'])
    ax10.contour(a_grid, b_grid, sig_norm / sig_norm.max(), 
        levels = len(levels) + 1, linestyles = '-', colors = ('k',))        
    plt.colorbar(CS, ax = ax10, orientation = 'horizontal')

    CS = ax11.contourf(a_grid, b_grid, np.log10(sig_norm), 
        levels = len(levels) + 1, cmap = cm_dict['k_norm'])
    ax11.contour(a_grid, b_grid, np.log10(sig_norm), 
        levels = len(levels) + 1, linestyles = '-', colors = ('k',))        
    plt.colorbar(CS, ax = ax11, orientation = 'horizontal')


    plt.show()
    
    return
    
def plot_optimal_frequency(h5_filepath):
    '''
    Plot swimming speed,
    '''    
    h5 = h5py.File(sweep_dir / h5_filepath, 'r')
    PG = ParameterGrid.init_pg_from_filepath(log_dir / h5.attrs['grid_filename'])

    to_quantities(PG.base_parameter)

    log_a_arr = np.log10(PG.v_from_key('a'))
    log_b_arr = np.log10(PG.v_from_key('b'))

    a_grid, b_grid = np.meshgrid(log_a_arr, log_b_arr) 
                          
    # Swimming speed
    U = h5['U'][:]  
    D_I = h5['energies']['D_I'][:]
    D_F = h5['energies']['D_F'][:]
    D = np.abs(D_I + D_F) 

    gs = plt.GridSpec(2, 2)
    ax00 = plt.subplot(gs[0, 0])
    ax01 = plt.subplot(gs[0, 1])
    ax10 = plt.subplot(gs[1, 0])
    ax11 = plt.subplot(gs[1, 1])
                
    levels = np.arange(0, 1.01, 0.2)
    
    CS = ax00.contourf(a_grid, b_grid, U.T / U.max(), 
        levels = levels, cmap = cm_dict['U'])
    ax00.contour(a_grid, b_grid, U.T / U.max(), 
        levels = levels, linestyles = '-', colors = ('k',))        
    plt.colorbar(CS, ax = ax00, orientation = 'horizontal')

    #levels = np.arange(0, 1.01, 0.2)
    CS = ax01.contourf(a_grid, b_grid, np.log10(D.T / D.max()), 
        levels = len(levels), cmap = cm_dict['W'])
    ax01.contour(a_grid, b_grid, np.log10(D.T / D.max()), 
        levels = len(levels), linestyles = '-', colors = ('k',))        
    plt.colorbar(CS, ax = ax01, orientation = 'horizontal')
    
    U_interp = RectBivariateSpline(log_a_arr, log_b_arr, U)
    W_interp = RectBivariateSpline(log_a_arr, log_b_arr, D)

    # Fluid viscosity
    mu_list = np.array([1e-3, 1e-2, 1e-1, 1e0, 1e1]) * ureg.pascal * ureg.second
    
    f_exp_arr_list = [
        np.linspace(-1, 2, int(3*1e1)),
        np.linspace(-1, 2, int(3*1e1)),
        np.linspace(-1, 2, int(3*1e1)),
        np.linspace(-1, 1.5, int(3*1e1)),
        np.linspace(-1, np.log10(3), int(3*1e1))
    
    ]

    markers = ['o', '^', 's', 'x', 'D']
    L0 = PG.base_parameter['L0']

    for mu, f_exp_arr, marker in zip(mu_list, f_exp_arr_list, markers):

        f_arr = 10**f_exp_arr / ureg.second
            
        a_arr, b_arr = PostProcessor.physical_2_dimless_parameters(
            PG.base_parameter, mu=mu, T_c = 1.0 / f_arr)

        log_a_arr, log_b_arr = np.log10(a_arr), np.log10(b_arr) 
        
        ax00.plot(log_a_arr[::2], log_b_arr[::2], '-', marker=marker , c='k')
        ax01.plot(log_a_arr[::2], log_b_arr[::2], '-', marker=marker , c='k')
        
        U_star = U_interp.ev(log_a_arr, log_b_arr)                
        W_star = W_interp.ev(log_a_arr, log_b_arr)
        
        U = PostProcessor.U_star_to_U(U_star, f_arr, L0)
        W = PostProcessor.E_star_to_E(W_star, mu, f_arr, L0)

        ax10.loglog(f_arr, U_star, '-', marker = marker, c='k')
        twinx10 = ax10.twinx()
        twinx10.loglog(f_arr, U / L0, marker = marker, c='r')

        ax11.loglog(f_arr, W_star / W_star.max() , '-', marker = marker, c='k')
        twinx11 = ax11.twinx()
        twinx11.loglog(f_arr , W /  W.max(), marker = marker, c='r')

        # twin_x = ax10.twinx()                
        # twin_x.semilogx(f_arr, U_star)
        
        # W_star = W_interp.ev(log_a_arr, log_b_arr)
    #
    # U = PostProcessor.U_star_to_U(U_star, f_arr, BP['L0'])
    # W = PostProcessor.E_star_to_E(W_star, mu, f_arr, BP['L0'])                                
    # ax10.semilogx(U, f_arr)
    # ax10.semilogx(U_star, f_arr)
    #
    # ax10.semilogx(U, f_arr)
    # ax10.semilogx(U_star, f_arr)
    
    plt.show()
    
    return

if __name__ == '__main__':
    
    # h5_a_b = ('analysis_'
    #     'a_min=-3.0_a_max=3.0_step_a=1.0_'
    #     'b_min=-3.0_b_max=0.0_step_b=1.0_'
    #     '_A=4.0_lam=1.0_N=250_dt=0.001.h5')    
    
    h5_a_b = ('analysis_'
        'a_min=-2.0_a_max=3.0_step_a=0.2_'
        'b_min=-3.0_b_max=0.0_step_b=0.2_'
        '_A=4.0_lam=1.0_N=250_dt=0.001.h5')


    plot_W_D(h5_a_b)
    plot_instantaneous_power_balance()
    #plot_speed_curvature_norm_and_amplitude(h5_a_b)
    #plot_optimal_frequency(h5_a_b)
        
        
        
    


