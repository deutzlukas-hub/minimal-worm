'''
Created on 19 Jun 2023

@author: amoghasiddhi
'''
# Built-in
from pathlib import Path

# Third-party
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import minimize
from scipy.integrate import trapezoid
import h5py 
import pint
import matplotlib.pyplot as plt


from parameter_scan import ParameterGrid 

# Local imports
from minimal_worm.experiments import PostProcessor
from minimal_worm.experiments.undulation import log_dir, sweep_dir, create_storage_dir
from minimal_worm.experiments.undulation.dirs import fig_dir
from minimal_worm.experiments.undulation.sweeps import fang_yen_fit,\
    fang_yen_data, david_gagnon, sznitzman, fit_gagnon_sznitman, rikmenspoel_1978,\
    fit_rikmenspoel_1978
from minimal_worm.plot import plot_scalar_field, plot_multiple_scalar_fields

ureg = pint.UnitRegistry()
markers = ['o', 's', '^', 'v', '<', '>', 'x', '+', '*', 'D', 'p', 'h', '1', '2', '3', '4', '8']

cm_dict = {
    'U': plt.cm.plasma,
    'V': plt.cm.plasma,
    'D': plt.cm.hot,
    'W': plt.cm.hot,
    'k_norm': plt.cm.winter,
    'sig_norm': plt.cm.hot,
    'D_I_over_D': plt.cm.seismic}

def to_quantities(param):

    for k, v in param.items():
        if isinstance(v, list):
            if len(v)==2:
                try:
                    unit = ureg(v[1])
                    param[k] = v[0]*unit                     
                except:
                    continue

def plot_fang_yen_sznitman_and_gagnon():

    mu_arr_1, lam_arr_1, f_arr_1, A_arr_1 = fang_yen_data()
    mu_arr_2, U_arr_2, f_arr_2, c_arr_2, A_real_arr = david_gagnon()
    mu_arr_3, U_arr_3, f_arr_3 = sznitzman()
    
    lam_arr_2 = c_arr_2 / f_arr_2
    
    lam_fit, f_fit, A_fit= fang_yen_fit()
    U_fit = fit_gagnon_sznitman()
        
    mu_log_min = np.min([np.log10(mu_arr_1).min(), np.log10(mu_arr_2).min()])
    mu_log_max = np.max([np.log10(mu_arr_1).max(), np.log10(mu_arr_2).max()])
        
    log_mu_fit_arr_1 = np.linspace(mu_log_min, mu_log_max, int(1e2))
         
    f_fit_arr_1 = f_fit(log_mu_fit_arr_1)
    lam_fit_arr_1 = lam_fit(log_mu_fit_arr_1)
    A_fit_arr_1 = A_fit(log_mu_fit_arr_1)
    U_fit_arr = U_fit(log_mu_fit_arr_1)

    _, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(5, 1, 
        figsize = (8, 16), sharex=True)
                                  
    ax0.semilogx(10**log_mu_fit_arr_1, f_fit_arr_1, '-', c='k')
    ax0.semilogx(mu_arr_1, f_arr_1, 's', c = 'k')
    ax0.semilogx(mu_arr_2, f_arr_2, 'o', c = 'k')
    ax0.semilogx(mu_arr_3, f_arr_3, 'x', c = 'k')    

    ax1.semilogx(10**log_mu_fit_arr_1, lam_fit_arr_1, '-', c='k')
    ax1.semilogx(mu_arr_1, lam_arr_1, 's', c = 'k')
    ax1.semilogx(mu_arr_2, lam_arr_2, 'o', c = 'k')

    ax2.semilogx(mu_arr_1, A_arr_1, 's', c = 'k')
    ax2.semilogx(10**log_mu_fit_arr_1, A_fit_arr_1, '-', c='k')

    ax3.semilogx(10**log_mu_fit_arr_1, U_fit_arr, '-', c = 'k')
    ax3.semilogx(mu_arr_2, U_arr_2, 'o', c = 'k')
    ax3.semilogx(mu_arr_3, U_arr_3, 'x', c = 'k')
    
    ax0.set_ylabel('$f$', fontsize = 20)
    ax1.set_ylabel('$\lambda$', fontsize = 20)
    ax2.set_ylabel('$A$', fontsize = 20)
    ax3.set_ylabel('$U$ [mm/s]', fontsize = 20)    
    ax4.set_ylabel('$A \cdot \lambda$', fontsize = 20)    
    ax4.set_xlabel('$\log(\mu)$', fontsize = 20)
    
    ax4.semilogx(10**log_mu_fit_arr_1, A_fit_arr_1 * lam_fit_arr_1, '-', c = 'k')
    
    ax0.set_ylim([0, 2.5])
    ax1.set_ylim([0, 2.5])
    ax2.set_ylim([0, 10])
    ax3.set_ylim([0, 0.5])
        
    plt.tight_layout()        
                
    plt.show()
    
    return
    
def plot_work_dissipation_balance(h5_filename: str, from_storage: bool = False):
    '''
    Plots the dissipated energy and mechanical work done during one undulation period. 
    
    The dissipated energy, i.e. the energy which leaves the system must be equal to 
    energy inputed into the system.        
    '''
    # if from_storage:
    #     log_dir, sim_dir, sweep_dir = create_storage_dir()        
    
    h5 = h5py.File(sweep_dir / h5_filename, 'r')    
    PG = ParameterGrid.init_pg_from_filepath(log_dir / h5.attrs['grid_filename'])

    log_a_arr = np.log10(PG.v_from_key('a'))
    log_b_arr = np.log10(PG.v_from_key('b'))
    log_a_grid, log_b_grid = np.meshgrid(log_a_arr, log_b_arr)

    V = h5['energies']['V'][:].T
    W = h5['energies']['W'][:].T
    D_I = h5['energies']['D_I'][:].T
    D_F = h5['energies']['D_F'][:].T
    D = D_I + D_F
    
    assert np.all(W >= 0)
    assert np.all(D <= 0)

    gs = plt.GridSpec(2, 2)
    ax00 = plt.subplot(gs[0,0])
    ax01 = plt.subplot(gs[0,1])
    ax10 = plt.subplot(gs[1,0])
    ax11 = plt.subplot(gs[1,1])
    
    levels = 6

    # log potential energy
    CS = ax00.contourf(log_a_grid, log_b_grid, np.log10(np.abs(V)), 
        levels = levels, cmap = cm_dict['V'])
    ax00.contour(log_a_grid, log_b_grid, np.log10(np.abs(V)), 
        levels = levels, linestyles = '-', colors = ('k',))        
    plt.colorbar(CS, ax = ax00, orientation = 'horizontal')

    # log potential energy normalized by average amplitude of 
    CS = ax01.contourf(log_a_grid, log_b_grid, np.log10(np.abs(D)), 
        levels = levels, cmap = cm_dict['D'])
    ax01.contour(log_a_grid, log_b_grid, np.log10(np.abs(D)), 
        levels = levels, linestyles = '-', colors = ('k',))        
    plt.colorbar(CS, ax = ax01, orientation = 'horizontal')

    # log dissipated energy
    CS = ax11.contourf(log_a_grid, log_b_grid, np.log10(W), 
        levels = levels, cmap = cm_dict['D'])
    ax11.contour(log_a_grid, log_b_grid, np.log10(W), 
        levels = levels, linestyles = '-', colors = ('k',))        
    plt.colorbar(CS, ax = ax11, orientation = 'horizontal')

    # log energy balance normalized by mechanical work
    CS = ax10.contourf(log_a_grid, log_b_grid, np.log10(np.abs(W + D) / W), 
        levels = levels, cmap = cm_dict['U'])
    ax10.contour(log_a_grid, log_b_grid, np.log10(np.abs(W + D) / W), 
        levels = levels, linestyles = '-', colors = ('k',))        
    plt.colorbar(CS, ax = ax10, orientation = 'horizontal')

    plt.show()
    
    return

def plot_speed_curvature_norm_and_amplitude(h5_filename):
    '''
    Plot swimming speed, curvature and amplitude
    '''    
    h5 = h5py.File(sweep_dir / h5_filename, 'r')
    PG = ParameterGrid.init_pg_from_filepath(log_dir / h5.attrs['grid_filename'])
    A = PG.base_parameter['A']

    #BP = to_quantities(PG.base_parameter)
                          
    # Swimming speed
    U = h5['U'][:].T  
    k_norm = h5['k_norm'][:].T  
    # sig_norm = h5['sig_norm'][:].T  
    A_max = h5['A_max'][:].T  
    A_min = h5['A_min'][:].T  
    
    log_a_arr = np.log10(PG.v_from_key('a'))
    log_b_arr = np.log10(PG.v_from_key('b'))

    a_grid, b_grid =np.meshgrid(log_a_arr, log_b_arr)

    gs = plt.GridSpec(2, 2)
    ax00 = plt.subplot(gs[0,0])
    ax01 = plt.subplot(gs[0,1])
    ax10 = plt.subplot(gs[1,0])
    ax11 = plt.subplot(gs[1,1])

    fz = 16
    
    levels = np.arange(0, 1.01, 0.2)
    
    CS = ax00.contourf(a_grid, b_grid, U / U.max(), 
        levels = levels, cmap = cm_dict['U'])
    ax00.contour(a_grid, b_grid, U / U.max(), 
        levels = levels, linestyles = '-', colors = ('k',))        
    cbar = plt.colorbar(CS, ax = ax00, orientation = 'horizontal')
    cbar.set_label(r'$U / \max(U)$', fontsize = fz)

    ax00.contour(a_grid, b_grid, U / U.max(), 
        levels = [0.5], linestyles = '--', colors = ('k',))        

    CS = ax01.contourf(a_grid, b_grid, k_norm / A, 
        levels = len(levels), cmap = cm_dict['k_norm'])
    ax01.contour(a_grid, b_grid, k_norm / A, 
        levels = len(levels), linestyles = '-', colors = ('k',))        
    cbar = plt.colorbar(CS, ax = ax01, orientation = 'horizontal')
    cbar.set_label(r'$L_2(\kappa) / A$', fontsize = fz)

    CS = ax10.contourf(a_grid, b_grid, A_max / A, 
        levels = len(levels), cmap = cm_dict['k_norm'])
    ax10.contour(a_grid, b_grid, A_max / A, 
        levels = len(levels), linestyles = '-', colors = ('k',))        
    cbar = plt.colorbar(CS, ax = ax10, orientation = 'horizontal')
    cbar.set_label(r'$A_\mathrm{max} / A_0$', fontsize = fz)

    CS = ax11.contourf(a_grid, b_grid, np.abs(A_max + A_min) / A_max, 
        levels = len(levels), cmap = cm_dict['k_norm'])
    ax11.contour(a_grid, b_grid, np.abs(A_max + A_min) / A_max, 
        levels = len(levels), linestyles = '-', colors = ('k',))        
    cbar = plt.colorbar(CS, ax = ax11, orientation = 'horizontal')
    cbar.set_label(r'$(A_\mathrm{max} + A_\mathrm{min}) / A_\mathrm{max}$', fontsize = fz)

    # CS = ax10.contourf(a_grid, b_grid, sig_norm / sig_norm.max() , 
    #     levels = len(levels) + 1, cmap = cm_dict['k_norm'])
    # ax10.contour(a_grid, b_grid, sig_norm / sig_norm.max(), 
    #     levels = len(levels) + 1, linestyles = '-', colors = ('k',))        
    # plt.colorbar(CS, ax = ax10, orientation = 'horizontal')
    #
    # CS = ax11.contourf(a_grid, b_grid, np.log10(sig_norm), 
    #     levels = len(levels) + 1, cmap = cm_dict['k_norm'])
    # ax11.contour(a_grid, b_grid, np.log10(sig_norm), 
    #     levels = len(levels) + 1, linestyles = '-', colors = ('k',))        
    # plt.colorbar(CS, ax = ax11, orientation = 'horizontal')


    plt.show()
    
    return

def plot_everything_for_different_c_and_lam(h5_filename):
    '''
    Plots: 
        - swimming speed U
        - L2 norm of real and preferred curvature difference k - k0
        - dissipated energy D=D_I+D_F
        - relative internal dissipation energy D_I / D
    
    for different c and lambda.
    
    Does the swimming speed and the L2 norm k - k0 show the same profile as 
    over the entire c lambda parameter range?
    
    How does the position of the transition band between the optimal undulation region
    and the struggle region changes?
    
    Does the relative internal dissipation energy D_I / D retains its sigmoidal profile. 
    Does sigmoid shifts as a function of lam and c?          
    '''    
    

    plot_dir = fig_dir / 'swimming_speed_and_curvature_norm' / Path(h5_filename).stem
    plot_dir.mkdir(parents = 'True', exist_ok = True)
        
    h5 = h5py.File(sweep_dir / h5_filename, 'r')
    PG = ParameterGrid.init_pg_from_filepath(log_dir / h5.attrs['grid_filename'])
    to_quantities(PG.base_parameter)

    lam_arr = PG.v_from_key('lam')
    
    if PG.base_parameter['use_c']:
        y_arr = PG.v_from_key('c')
    else:                          
        y_arr = PG.v_from_key('A')
         
    log_a_arr = np.log10(PG.v_from_key('a'))
    log_b_arr = np.log10(PG.v_from_key('b'))

    a_grid, b_grid = np.meshgrid(log_a_arr, log_b_arr)

    fz = 16

    k = 0
    
    for i, y in enumerate(y_arr):
        for j, lam in enumerate(lam_arr):

            if PG.base_parameter['use_c']:
                A = np.round(2 * np.pi * y / lam, 1)  
                c = y
            else:
                A = y
                c = 2*np.pi*A / lam

            U = h5['U'][i, j, :].T
            k_norm = h5['k_norm'][i, j, :].T
            sig_norm = h5['sig_norm'][i, j, :].T  

            W = h5['energies']['W'][i, j, :].T
            D_I = h5['energies']['D_I'][i, j, :].T
            D_F = h5['energies']['D_F'][i, j, :].T
            D = D_I + D_F
            D_I_over_D = D_I / D

            fig = plt.figure(figsize = (18, 16))
            gs = plt.GridSpec(2, 3)
            ax00 = plt.subplot(gs[0,0])
            ax01 = plt.subplot(gs[0,1])
            ax02 = plt.subplot(gs[0,2])
            ax10 = plt.subplot(gs[1,0])
            ax11 = plt.subplot(gs[1,1])
            ax12 = plt.subplot(gs[1,2])

            fig.suptitle(f'$A={A}, \lambda={lam}, c={c}$', fontsize = fz)
    
            levels = np.arange(0, 1.01, 0.2)
            
            CS = ax00.contourf(a_grid, b_grid, U / U.max(), 
                levels = levels, cmap = cm_dict['U'])
            ax00.contour(a_grid, b_grid, U / U.max(), 
                levels = levels, linestyles = '-', colors = ('k',))        
            cbar = plt.colorbar(CS, ax = ax00, orientation = 'horizontal')
            cbar.set_label(r'$U / \max(U)$', fontsize = fz)
        
            CS = ax01.contourf(a_grid, b_grid, k_norm / A, 
                levels = len(levels), cmap = cm_dict['k_norm'])
            ax01.contour(a_grid, b_grid, k_norm / A, 
                levels = len(levels), linestyles = '-', colors = ('k',))        
            cbar = plt.colorbar(CS, ax = ax01, orientation = 'horizontal')
            cbar.set_label(r'$|\kappa|/A$', fontsize = fz)

            CS = ax02.contourf(a_grid, b_grid, np.log10(np.abs(sig_norm)), 
                levels = len(levels) + 2, cmap = cm_dict['sig_norm'])
            ax02.contour(a_grid, b_grid, np.log10(np.abs(sig_norm)), 
                levels = len(levels) + 2, linestyles = '-', colors = ('k',))        
            cbar = plt.colorbar(CS, ax = ax02, orientation = 'horizontal')
            cbar.set_label(r'$\log(|W|)$', fontsize = fz)

                         
            CS = ax10.contourf(a_grid, b_grid, np.log10(W), 
                cmap = cm_dict['W'])
            ax10.contour(a_grid, b_grid, np.log10(W),
                linestyles = '-', colors = ('k',))        
            cbar = plt.colorbar(CS, ax = ax10, orientation = 'horizontal')
            cbar.set_label(r'$\log(|W|)$', fontsize = fz)
        
            CS = ax11.contourf(a_grid, b_grid, np.log10(np.abs(D)), 
                cmap = cm_dict['W'])
            ax11.contour(a_grid, b_grid, np.log10(np.abs(D)), 
                linestyles = '-', colors = ('k',))        
            cbar = plt.colorbar(CS, ax = ax11, orientation = 'horizontal')
            cbar.set_label(r'$\log(|D|)$', fontsize = fz)
                         
            CS = ax12.contourf(a_grid, b_grid, np.log10(D_I_over_D), 
                cmap = plt.cm.seismic)
            ax12.contour(a_grid, b_grid, np.log10(D_I_over_D), 
                linestyles = '-', colors = ('k',))        
            cbar = plt.colorbar(CS, ax = ax12, orientation = 'horizontal')
            cbar.set_label(r'$D_\mathrm{I}/D)$', fontsize = fz)

            plt.savefig(plot_dir / f'{str(k).zfill(2)}.png')
            k += 1
            plt.close()
    
    return
   
def plot_everything_over_a_b_for_different_C(h5_filename: str):
    '''
    Plots: 
        - swimming speed U
        - L2 norm k - k0 of real and preferred curvature difference
        - dissipated energy D=D_I+D_F
        - relative internal dissipation energy D_I / D
    
    Why?
    
    Does the swimming speed and the L2 norm k - k0 show the same profile as 
    function of a and b for different C?
    
    Does the relative internal dissipation energy D_I / D retains its 
    sigmoidal profile. Does sigmoid shifts as a function of C.      
    
    C moves midpoint of sigmoidal to the left for a fixed b        
    '''
    
    # Create figure directory if it does not exist
    plot_dir = fig_dir / 'everything_over_a_and_b' / Path(h5_filename).stem
    plot_dir.mkdir(parents = True, exist_ok = True)
    
    # Load Data
    h5 = h5py.File(sweep_dir / h5_filename, 'r')
    PG = ParameterGrid.init_pg_from_filepath(log_dir / h5.attrs['grid_filename'])
    to_quantities(PG.base_parameter)

    # Create a, b grid from ParameterGird
    log_a_arr = np.log10(PG.v_from_key('a'))
    log_b_arr = np.log10(PG.v_from_key('b'))
    a_grid, b_grid = np.meshgrid(log_a_arr, log_b_arr) 

    C_arr = PG.v_from_key('C')        
    fz = 18
        
    for i, C in enumerate(C_arr):

        U = h5['U'][i, :].T
        k_norm = h5['k_norm'][i, :].T
        sig_norm = h5['sig_norm'][i, :].T  

        W = h5['energies']['W'][i, :].T
        D_I = h5['energies']['D_I'][i, :].T
        D_F = h5['energies']['D_F'][i, :].T
        D = D_I + D_F
        D_I_over_D = D_I / D

        fig = plt.figure(figsize = (18, 16))
        gs = plt.GridSpec(2, 3)
        ax00 = plt.subplot(gs[0,0])
        ax01 = plt.subplot(gs[0,1])
        ax02 = plt.subplot(gs[0,2])
        ax10 = plt.subplot(gs[1,0])
        ax11 = plt.subplot(gs[1,1])
        ax12 = plt.subplot(gs[1,2])

        fig.suptitle(f'$C={C}$', fontsize = fz)

        levels = np.arange(0, 1.01, 0.2)
        
        CS = ax00.contourf(a_grid, b_grid, U / U.max(), 
            levels = levels, cmap = cm_dict['U'])
        ax00.contour(a_grid, b_grid, U / U.max(), 
            levels = levels, linestyles = '-', colors = ('k',))        
        cbar = plt.colorbar(CS, ax = ax00, orientation = 'horizontal')
        cbar.set_label(r'$U / \max(U)$', fontsize = fz)
    
        CS = ax01.contourf(a_grid, b_grid, k_norm / PG.base_parameter['A'], 
            levels = len(levels), cmap = cm_dict['k_norm'])
        ax01.contour(a_grid, b_grid, k_norm / PG.base_parameter['A'], 
            levels = len(levels), linestyles = '-', colors = ('k',))        
        cbar = plt.colorbar(CS, ax = ax01, orientation = 'horizontal')
        cbar.set_label(r'$|\kappa|/A$', fontsize = fz)

        CS = ax02.contourf(a_grid, b_grid, np.log10(np.abs(sig_norm)), 
            levels = len(levels) + 2, cmap = cm_dict['sig_norm'])
        ax02.contour(a_grid, b_grid, np.log10(np.abs(sig_norm)), 
            levels = len(levels) + 2, linestyles = '-', colors = ('k',))        
        cbar = plt.colorbar(CS, ax = ax02, orientation = 'horizontal')
        cbar.set_label(r'$\log(|W|)$', fontsize = fz)
                     
        CS = ax10.contourf(a_grid, b_grid, np.log10(W), 
            cmap = cm_dict['W'])
        ax10.contour(a_grid, b_grid, np.log10(W),
            linestyles = '-', colors = ('k',))        
        cbar = plt.colorbar(CS, ax = ax10, orientation = 'horizontal')
        cbar.set_label(r'$\log(|W|)$', fontsize = fz)
    
        CS = ax11.contourf(a_grid, b_grid, np.log10(np.abs(D)), 
            cmap = cm_dict['W'])
        ax11.contour(a_grid, b_grid, np.log10(np.abs(D)), 
            linestyles = '-', colors = ('k',))        
        cbar = plt.colorbar(CS, ax = ax11, orientation = 'horizontal')
        cbar.set_label(r'$\log(|D|)$', fontsize = fz)
                     
        CS = ax12.contourf(a_grid, b_grid, np.log10(D_I_over_D), 
            cmap = plt.cm.seismic)
        ax12.contour(a_grid, b_grid, np.log10(D_I_over_D), 
            linestyles = '-', colors = ('k',))        
        cbar = plt.colorbar(CS, ax = ax12, orientation = 'horizontal')
        cbar.set_label(r'$D_\mathrm{I}/D)$', fontsize = fz)
    
        plt.savefig(plot_dir / f'{str(i).zfill(2)}.png')
        plt.close()
        
    return
         
def plot_transition_bands_for_different_c_lam(h5_filename):
    '''
    Plots: 
        - transition bands defined by contourline U/U_max = 0.5
    
    for different c and lambda.
    
    How to the band position changes as a function of c and lambda?     
    '''    
    
    plot_dir = fig_dir / 'transition_bands' / Path(h5_filename).stem
    plot_dir.mkdir(parents = 'True', exist_ok = True)
        
    h5 = h5py.File(sweep_dir / h5_filename, 'r')
    PG = ParameterGrid.init_pg_from_filepath(log_dir / h5.attrs['grid_filename'])
    to_quantities(PG.base_parameter)

    lam_arr = PG.v_from_key('lam')
    
    if PG.base_parameter['use_c']:
        y_arr = PG.v_from_key('c')
    else:                          
        y_arr = PG.v_from_key('A')
         
    log_a_arr = np.log10(PG.v_from_key('a'))
    log_b_arr = np.log10(PG.v_from_key('b'))

    a_grid, b_grid = np.meshgrid(log_a_arr, log_b_arr)

    fz = 16
    k = 0    
    
    gs = plt.GridSpec(1,1)
    ax00 = plt.subplot(gs[0, 0])

    cmap = plt.get_cmap('viridis')        
    norm = plt.Normalize(vmin=lam_arr.min(), vmax=lam_arr.max())
        
    for i, y in enumerate(y_arr):

        
        for j, lam in enumerate(lam_arr):

            if PG.base_parameter['use_c']:
                A = np.round(2 * np.pi * y / lam, 1)  
                c = y
            else:
                A = y
                c = 2*np.pi*A / lam

            U = h5['U'][i, j, :].T

            cont = ax00.contour(a_grid, b_grid, U / U.max(), 
                levels = [0.5], linestyles = '--', colors = [cmap(norm(lam))])        

            plt.clabel(cont, inline=False, fontsize=14, fmt='%s' % [f'c={np.round(c,2)}'])

    
    fz = 16    
    ax00.set_xlabel('$\log(a)$', fontsize = fz)
    ax00.set_ylabel('$\log(b)$', fontsize = fz)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm)    
    cbar.set_label(r'$\lambda$', fontsize = fz)
    
    plt.show()
    
    return         
           
def plot_transition_bands_different_lam(h5_filename):
    '''
    Plots: 
        - transition bands defined by contourline U/U_max = 0.5
    
    for different c and lambda.
    
    How to the band position changes as a function of c and lambda?     
    '''    
    
    plot_dir = fig_dir / 'transition_bands' / Path(h5_filename).stem
    plot_dir.mkdir(parents = 'True', exist_ok = True)
        
    h5 = h5py.File(sweep_dir / h5_filename, 'r')
    PG = ParameterGrid.init_pg_from_filepath(log_dir / h5.attrs['grid_filename'])
    to_quantities(PG.base_parameter)

    lam_arr = PG.v_from_key('lam')
    
    assert PG.base_parameter['use_c']
    
    c_arr = PG.v_from_key('c')

    c_idx = np.abs(c_arr - 1.0).argmin()
         
    log_a_arr = np.log10(PG.v_from_key('a'))
    log_b_arr = np.log10(PG.v_from_key('b'))

    a_grid, b_grid = np.meshgrid(log_a_arr, log_b_arr)

    fz = 16
    k = 0    
    
    gs = plt.GridSpec(1,1)
    ax00 = plt.subplot(gs[0, 0])

    cmap = plt.get_cmap('viridis')        
    norm = plt.Normalize(vmin=lam_arr.min(), vmax=lam_arr.max())
            
    for i, lam in enumerate(lam_arr):

        U = h5['U'][c_idx, i, :].T

        cont = ax00.contour(a_grid, b_grid, U / U.max(), 
            levels = [0.5], linestyles = '--', colors = [cmap(norm(lam))])        

        plt.clabel(cont, inline=False, fontsize=14, fmt='%s' % [fr'lambda={lam}'])
    
    fz = 16    
    ax00.set_xlabel('$\log(a)$', fontsize = fz)
    ax00.set_ylabel('$\log(b)$', fontsize = fz)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm)    
    cbar.set_label(r'$\lambda$', fontsize = fz)
    
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

    gs = plt.GridSpec(4, 2)
    ax00 = plt.subplot(gs[0, 0])
    ax01 = plt.subplot(gs[0, 1])
    ax10 = plt.subplot(gs[1, 0])
    ax11 = plt.subplot(gs[1, 1])
    ax20 = plt.subplot(gs[2, 0])
    ax21 = plt.subplot(gs[2, 1])
    ax30 = plt.subplot(gs[3, 0])
    ax31 = plt.subplot(gs[3, 1])
                                
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
    mu_exp_arr = np.arange(-3, 1.01, 0.5)
        
    mu_list = 10 ** mu_exp_arr * ureg.pascal * ureg.second
    
    f_exp_arr_list = [
        np.linspace(-0.5, 2, int(3*1e1)),
        np.linspace(-0.5, 2, int(3*1e1)),
        np.linspace(-1, 2, int(3*1e1)),
        np.linspace(-1, 2, int(3*1e1)),
        np.linspace(-1, 2, int(3*1e1)),
        np.linspace(-1, 2, int(3*1e1)),
        np.linspace(-1, 1.5, int(3*1e1)),
        np.linspace(-1, 1.0, int(3*1e1)),
        np.linspace(-1, np.log10(3), int(3*1e1))
    
    ]

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
        idx = np.argmax(U)
        
        U_over_L0 = U / L0
        
        a_max = log_a_arr[idx]
        b_max = log_b_arr[idx]
        
        ax00.plot(a_max, b_max, marker=marker, c = 'r')
        ax01.plot(a_max, b_max, marker=marker, c = 'r')
                
        W_over_S = f_arr * W  / U

        assert np.allclose(W_over_S, f_arr * mu * L0**2 * W_star / U_star)
        
        ax10.loglog(f_arr, U_over_L0, '-', marker = marker, c='k')        
        ax10.loglog(f_arr[idx], U_over_L0[idx], marker = marker, c='r')        
        ax11.plot(f_arr, W_over_S, '-', marker = marker, c='k')
        
        ax20.loglog(f_arr, U / f_arr, '-', marker = marker, c='k')
        ax21.loglog(f_arr, W_over_S, '-', marker = marker, c='k')
             
        ax31.plot(f_arr, W_star / U_star, '-', marker = marker, c='k')
        ax30.loglog(f_arr, W, '-', marker = marker, c='k')
                                
    ax10.set_xlabel(r'$U/L_0$', fontsize = 16)
    ax11.set_xlabel(r'$W/S$', fontsize = 16)

    ax10.set_xlabel(r'Frequency [Hz]', fontsize = 16)
    ax11.set_xlabel(r'Frequency [Hz]', fontsize = 16)
    
    plt.show()
    
    return

def plot_swimming_speed_sperm(h5_filename: str):

    # Sperm parameters from 
    # "Movement of sea urchin sperm flagella"
    # by Rikmenspoel
    L0 = 40 * 1e-6 * ureg.meter # Length
    alpha = 0.01 * ureg.dimensionless # Slenderness parameter
    mu = 1e-3 * ureg.newton / ureg.meter **2 * ureg.second # Fluid viscosity
    c_t = 2*np.pi / (np.log(2/alpha) - 0.5) * ureg.dimensionless # Tangential drag coefficient     
    EI = 1e-21 * ureg.newton * ureg.meter**2 # stiffness 
    f = 35 / ureg.second # undulation frequency

    # Load data
    h5 = h5py.File(sweep_dir / h5_filename, 'r')
    PG = ParameterGrid.init_pg_from_filepath(log_dir / h5.attrs['grid_filename'])
    to_quantities(PG.base_parameter)


    # Create a, b grid from ParameterGird
    log_a_arr = np.log10(PG.v_from_key('a'))
    log_b_arr = np.log10(PG.v_from_key('b'))
    a_grid, b_grid = np.meshgrid(log_a_arr, log_b_arr) 
    
    U = h5['U'][:]
    U_interp = RectBivariateSpline(log_a_arr, log_b_arr, U)
    
    f_arr = 10 ** np.linspace(0, 2, int(0.5e2)) / ureg.second 
    
    tau = (c_t * mu * L0**4 / EI).to_compact()         
    assert tau.units == ureg.second    
    log_a_arr = np.log10(tau * f_arr) 

    fz = 16

    gs = plt.GridSpec(2,1)
    ax00 = plt.subplot(gs[0,0])
    ax10 = plt.subplot(gs[1,0])
    # ax20 = plt.subplot(gs[2, 0])
        
    levels = np.arange(0, 1.01, 0.2)
    
    CS = ax00.contourf(a_grid, b_grid, U.T / U.max(), 
        levels = levels, cmap = cm_dict['U'])
    ax00.contour(a_grid, b_grid, U.T / U.max(), 
        levels = levels, linestyles = '-', colors = ('k',))        
    cbar = plt.colorbar(CS, ax = ax00, orientation = 'horizontal')
    cbar.set_label(r'$U / \max(U)$', fontsize = fz)
    
    for exp_xi, marker in zip(np.linspace(-4, -2, 5), markers):            
        
        xi = 10**exp_xi * ureg.second
            
        log_b_arr = np.log10(xi * f_arr)

        U_star_arr = U_interp.ev(log_a_arr, log_b_arr)
        U_arr = PostProcessor.U_star_to_U(U_star_arr, f_arr, L0) / L0

        ax00.plot(log_a_arr, log_b_arr, '-', marker = marker, c = 'k')    
        ax10.plot(f_arr, U_arr, '-', marker = marker, c = 'k', label = fr'$\log(\eta/E)={exp_xi}$')
        # ax20.plot(f_arr, U_star_arr, '-', marker = marker, c = 'k')

    ax10.axvspan(f.magnitude, 52, facecolor='red', alpha=0.5)
    ax10.axvline(f.magnitude, color = 'r', linestyle = '--')
        
    ax00.set_xlabel('$a$', fontsize = fz)
    ax00.set_ylabel('$b$', fontsize = fz)
    ax10.set_xlabel('$f [Hz]$', fontsize = fz)
    ax10.set_ylabel('$U/L$', fontsize = fz)
    ax10.legend(fontsize = 12)
    
    plt.show()
            
    return
    
    
def plot_dissipation_ratio(h5_filename: str):
    '''
    Plots the ratio between the internal and external (fluid) dissipation rate
    as a function of the time scale ratios a and b.
    
    Plots the sigmoidal transition for fixed b and varying a. 
    
    This can be interpreted as varyin 
    
        
    '''
    # Plot directory
    plot_dir = fig_dir / 'dissipation_ratio'  
    plot_dir.mkdir(parents = 'True', exist_ok = True)
      
    # Load data  
    h5 = h5py.File(sweep_dir / h5_filename, 'r')
    PG = ParameterGrid.init_pg_from_filepath(log_dir / h5.attrs['grid_filename'])
    to_quantities(PG.base_parameter)
    
    # Create time scale ratio mesh
    log_a_arr = np.log10(PG.v_from_key('a'))
    log_b_arr = np.log10(PG.v_from_key('b'))
    a_grid, b_grid = np.meshgrid(log_a_arr, log_b_arr) 
                          
    # Dissipation rates 
    D_I = h5['energies']['D_I'][:]
    D_F = h5['energies']['D_F'][:]
    D = D_I + D_F 

    D_I_over_D = D_I / D
    D_I_over_D_interp = RectBivariateSpline(log_a_arr, log_b_arr, D_I_over_D)


    plt.figure(figsize = (18, 8))
    gs = plt.GridSpec(1, 3)
    ax00 = plt.subplot(gs[0, 0])
    ax01 = plt.subplot(gs[0, 1])
    ax02 = plt.subplot(gs[0, 2])

    # Plot D_I/D as contour plot                                
    CS = ax00.contourf(a_grid, b_grid, D_I.T / D.T, 
        cmap = plt.cm.seismic)
    ax00.contour(a_grid, b_grid, D_I.T / D.T, 
        linestyles = '-', colors = ('k',))        
    plt.colorbar(CS, ax = ax00, 
        orientation = 'vertical', label = r'$D_\mathrm{I}/D$')
           
    log_a_arr = np.linspace(log_a_arr.min(), log_a_arr.max(), int(1e3)) 
           
    # Plot three horizontal lines for fixed b
    # to demonstrate that the transition shifts
    ls = ['-', '--', ':']    
    for i, log_b in enumerate([-3.0, -2.0, -1.0]):#np.linspace(log_b_arr.min(), log_b_arr.max(), 4): 
    
        ax00.axhline(log_b, linestyle = ls[i], c='k', lw = 4.0)
    
        D_I_over_D_arr = D_I_over_D_interp.ev(
            log_a_arr, log_b*np.ones_like(log_a_arr))
                
        ax01.plot(log_a_arr, D_I_over_D_arr, c = 'k', ls = ls[i])
        
    N = int(1e2)
    log_a_midpoint_arr = np.zeros(N)
    log_b_arr = np.linspace(-3.0, -1.0, N)

    for i, log_b in enumerate(log_b_arr):
        
        D_I_over_D_arr = D_I_over_D_interp.ev(log_a_arr, log_b*np.ones_like(log_a_arr))
                
        idx = np.abs(D_I_over_D_arr - 0.5).argmin()        
        log_a_midpoint_arr[i] = log_a_arr[idx]
       
    ax02.plot(log_b_arr, log_a_midpoint_arr, c = 'k')
            
    ax00.set_xlabel(r'$\log(a)$', fontsize = 16)
    ax00.set_ylabel(r'$\log(b)$', fontsize = 16)    
    ax01.set_xlabel(r'$\log(a)$', fontsize = 16)
    ax02.set_xlabel(r'$\log(a_\mathrm(mp))$', fontsize = 16)
    ax02.set_xlabel(r'$\log(b)$', fontsize = 16)
        
    plt.show()
    
    return

def plot_dissipation_ratio_fang_yen(h5_filename):
    '''
    Plots 
    - D_I over D as a function over c and lambda 
    
    for different fluid viscosities mu and undulation frequency f 
    taken from Fang Yen.
        
    '''    
    plot_dir = fig_dir / 'D_I_over_D' / Path(h5_filename).stem 
    plot_dir.mkdir(parents = True, exist_ok = True)
        
    # Load data  
    h5 = h5py.File(sweep_dir / h5_filename, 'r')
    PG = ParameterGrid.init_pg_from_filepath(log_dir / h5.attrs['grid_filename'])
    #to_quantities(PG.base_parameter)
    
    # Create time scale ratio mesh
    mu_arr = PG.v_from_key('mu')
    c_arr = PG.v_from_key('c')
    lam_arr = PG.v_from_key('lam')
    lam_grid, c_grid = np.meshgrid(lam_arr, c_arr) 
    
    for i, mu in enumerate(mu_arr):

        plt.figure()
        ax0 = plt.subplot(111)
        plt.suptitle(fr'$\log(\mu)={np.log10(mu)}$')
    
        D_I = h5['energies']['D_I'][i, :]
        D_F = h5['energies']['D_F'][i, :]
        D = D_I + D_F
        D_I_over_D = D_I / D
                   
        levels = np.arange(0, 1.01, 0.2)
                   
        CS = ax0.contourf(lam_grid, c_grid, D_I_over_D, 
            levels = levels, cmap = cm_dict['D_I_over_D'])
        ax0.contour(lam_grid, c_grid, D_I_over_D,   
            levels = levels, linestyles = '-', colors = ('k',))        
        plt.colorbar(CS, ax = ax0, 
            orientation = 'horizontal', label = r'$D / D_\mathrm{I}$')
    
        ax0.set_xlabel('$\lambda$')
        ax0.set_ylabel('$c$')
                    
        plt.tight_layout()    
        plt.savefig(plot_dir / f'{str(i).zfill(2)}.png')
        plt.close()
            
    return

def plot_optima_on_contours_eta_mu(h5_filename: str):
    '''
    Plots 
        - swimming speed contours over lambda and c 
        - mechanical work over lambda and c
        - dissipated fluid energy over lambda and c
        - dissipated internal energy over lambda and c
        
    for different internal fluid viscosity mu, eta     
    '''        
    # Load data
    h5 = h5py.File(sweep_dir / h5_filename, 'r')
    PG = ParameterGrid.init_pg_from_filepath(log_dir / h5.attrs['grid_filename'])
    to_quantities(PG.base_parameter)
    
    eta = PG.v_from_key('eta')
    E = PG.base_parameter['E']
    log_xi_arr = np.log10(eta.magnitude / E.magnitude)     
    mu_arr = PG.v_from_key('mu')
    log_mu_arr = np.round(np.log10(mu_arr), 2)
        
    c_arr = PG.v_from_key('c')
    lam_arr = PG.v_from_key('lam')

    # Create directory for figures
    plot_dir = fig_dir / 'optimal_kinematics' / 'contours' / Path(h5_filename).stem 
    plot_dir.mkdir(parents = True, exist_ok = True)

    k = 0
    
    for i, log_xi in enumerate(log_xi_arr):
        for j, log_mu in enumerate(log_mu_arr):
                        
            U = h5['U'][i, j, :]          
            A = h5['A_max'][i, j, :]
            W = h5['energies']['W'][i, j, :]            
            D_I = h5['energies']['D_I'][i, j, :]            
            D_F = h5['energies']['D_F'][i, j, :]            
                                                
            plot_optima_on_contours(
                f'{str(k).zfill(2)}', 
                plot_dir, 
                fr'$\log(\xi)={round(log_xi,2)}, \log(\mu)={log_mu}$', 
                c_arr, 
                lam_arr, 
                U, 
                W, 
                A, 
                D_I, 
                D_F)
            
            k += 1
            
    return


def plot_optima_on_contours_C_eta_mu(h5_filename: str):
    '''
    Plots 
        - swimming speed contours over lambda and c 
        - mechanical work over lambda and c
        - dissipated fluid energy over lambda and c
        - dissipated internal energy over lambda and c
        
    for different drag coefficient ratio C, internal viscosity eta     
    and fluid viscosity mu 
    '''    
    # Create directory for figures
    plot_dir = fig_dir / 'optimal_kinematics' / 'contours' / Path(h5_filename).stem 
    plot_dir.mkdir(parents = True, exist_ok = True)
    
    # Load data
    h5 = h5py.File(sweep_dir / h5_filename, 'r')
    PG = ParameterGrid.init_pg_from_filepath(log_dir / h5.attrs['grid_filename'])
    to_quantities(PG.base_parameter)
    
    C = PG.v_from_key('C')
    eta = PG.v_from_key('eta')
    E = PG.base_parameter['E']
    log_xi_arr = np.log10(eta.magnitude / E.magnitude)     
    mu_arr = PG.v_from_key('mu')
    log_mu_arr = np.round(np.log10(mu_arr), 2)
        
    c_arr = PG.v_from_key('c')
    lam_arr = PG.v_from_key('lam')

    l = 0
    
    for i, C in enumerate(C):
        for j, log_xi in enumerate(log_xi_arr):
            for k, log_mu in enumerate(log_mu_arr):

                print(l)
                                        
                U = h5['U'][i, j, k, :]          
                A = h5['A_max'][i, j, k, :]
                W = h5['energies']['W'][i, j, k, :]            
                D_I = h5['energies']['D_I'][i, j, k, :]            
                D_F = h5['energies']['D_F'][i, j, k, :]            
                                                
                plot_optima_on_contours(
                    f'{str(l).zfill(2)}', 
                    plot_dir, 
                    fr'$C={C}, \log(\xi)={log_xi}, \log(\mu)={log_mu}$', 
                    c_arr, 
                    lam_arr, 
                    U, 
                    W, 
                    A, 
                    D_I, 
                    D_F)
                
                l += 1
            
    return


def plot_optima_on_contours(
        figname: str,
        plot_dir: Path,
        title: str,
        c_arr: np.ndarray,
        lam_arr: np.ndarray,
        U: np.ndarray, 
        W: np.ndarray,
        A: np.ndarray, 
        D_I: np.ndarray, 
        D_F: np.ndarray
): 
    '''
    Plots 
        - swimming speed contours and optima
        - swimming mechanical work and optima
                             
    '''        

    D_I, D_F = np.abs(D_I), np.abs(D_F) 
                                      
    res_opt_W = PostProcessor.comp_optimal_c_and_wavelength(
        U, W, A, c_arr, lam_arr)
            
    res_opt_D_I = PostProcessor.comp_optimal_c_and_wavelength(
        U, np.abs(D_I), A, c_arr, lam_arr)

    res_opt_D_F = PostProcessor.comp_optimal_c_and_wavelength(
        U, np.abs(D_F), A, c_arr, lam_arr)
    
    # remeshed finer resolution
    U, W = res_opt_W['U'], res_opt_W['W']
    D_I, D_F  = res_opt_D_I['W'], res_opt_D_F['W']
                      
    lam_fine_arr, c_fine_arr  = res_opt_W['lam_arr'], res_opt_W['c_arr']
    lam_grid, c_grid = np.meshgrid(lam_fine_arr, c_fine_arr)                                                
    
    lam_max, c_max = res_opt_W['lam_max'], res_opt_W['c_max']          

    #------------------------------------------------------------------------------ 
    # Plotting

    plt.figure(figsize = (10, 10))
    gs = plt.GridSpec(2,2)
    
    ax00 = plt.subplot(gs[0, 0])
    ax01 = plt.subplot(gs[0, 1])
    ax10 = plt.subplot(gs[1, 0])
    ax11 = plt.subplot(gs[1, 1])
        
    levels = np.arange(0, 1.01, 0.1)
    
    # U/U_max contour
    CS = ax00.contourf(lam_grid, c_grid, U.T / U.max(), 
        levels = levels, cmap = cm_dict['U'])
    ax00.contour(lam_grid, c_grid, U.T / U.max(),   
        levels = levels, linestyles = '-', colors = ('k',))        
    cbar = plt.colorbar(CS, ax = ax00, orientation = 'horizontal', 
        label = r'$U / U_\mathrm{max}$')                        
    
    #colors = .tolist()
 
    colors = res_opt_W['levels']
     
    # W contour
    CS = ax01.contourf(lam_grid, c_grid, np.log10(W.T), 
        cmap = cm_dict['W'])        
    ax01.contour(lam_grid, c_grid, U.T / U.max(),   
        levels = levels, linestyles = '--', colors = ('k',))        
    plt.colorbar(CS, ax = ax01, 
        orientation = 'horizontal', label = r'$\log(W/\max(W))$')
    
    ax01.scatter(res_opt_W['lam_opt_arr'], res_opt_W['c_opt_arr'], 
        marker ='o', c = colors, edgecolors = 'k')

    # D_F contour
    CS = ax10.contourf(lam_grid, c_grid, np.log10(D_F.T / W.max()), 
        cmap = cm_dict['D'])        
    ax10.contour(lam_grid, c_grid, U.T / U.max(),   
        levels = levels, linestyles = '--', colors = ('k',))        
    plt.colorbar(CS, ax = ax10, orientation = 'horizontal', 
        label = r'$\log(|D_\mathrm{F}| / \max(W))$')
        
    ax10.scatter(res_opt_D_F['lam_opt_arr'], res_opt_D_F['c_opt_arr'], 
        marker = 'o', c = colors, edgecolors = 'k')

    # D_I contour
    CS = ax11.contourf(lam_grid, c_grid, np.log10(D_I.T / W.max()), 
        cmap = cm_dict['D'])        
    ax11.contour(lam_grid, c_grid, U.T / U.max(),   
        levels = levels, linestyles = '--', colors = ('k',))        
    plt.colorbar(CS, ax = ax11, orientation = 'horizontal', 
        label = r'$\log(D_\mathrm{I}) / \max(W)$')

    ax11.scatter(res_opt_D_I['lam_opt_arr'], res_opt_D_I['c_opt_arr'], 
        marker = 'o', c = colors, edgecolors = 'k')
    
    # Plot lambda and c for which swimming speed is maximal 
    for ax in [ax00, ax01, ax10, ax11]:        
        ax.scatter(lam_max, c_max, marker='^', c = [cbar.cmap(1.0)],
            edgecolor = 'k')
        
    #------------------------------------------------------------------------------ 
    # Figure layout 
    plt.suptitle(title)

    ax00.set_xlabel('$\lambda$')
    ax00.set_ylabel('$c$')
    ax01.set_xlabel('$\lambda$')
    ax01.set_ylabel('$c$')
    ax10.set_xlabel('$\lambda$')
    ax10.set_ylabel('$c$')
    ax11.set_xlabel('$\lambda$')
    ax11.set_ylabel('$c$')
                                
    plt.tight_layout()    
    plt.savefig(plot_dir / figname)
    plt.close()
            
    return

def plot_optimal_wave_length_for_different_eta_and_mu_0(h5_filename):
    '''
    Plots 
    
    - optimal lambda 
    - optimal curvature amplitude 
    
    over mu for swimming speed contours U=c*U_max where c <= 1 
    
    Create individual figure for every eta.    
    '''
    # Create directory for figures
    plot_dir = fig_dir / 'optimal_kinematics' / Path(h5_filename).stem 
    plot_dir.mkdir(parents = True, exist_ok = True)
    
    # Load data
    h5 = h5py.File(sweep_dir / h5_filename, 'r')
    PG = ParameterGrid.init_pg_from_filepath(log_dir / h5.attrs['grid_filename'])
    to_quantities(PG.base_parameter)

    L0 = PG.base_parameter['L0'].magnitude

    log_xi_arr = np.log10(PG.v_from_key('eta').magnitude / PG.base_parameter['E'].magnitude)     
    mu_arr = PG.v_from_key('mu')
    log_mu_arr = np.log10(mu_arr)    
    c_arr = PG.v_from_key('c')
    lam_arr = PG.v_from_key('lam')

    mu_fang_yen_arr, lam_fang_yen_arr, _, A_fang_yen_arr = fang_yen_data()
    log_mu_fang_yen_arr = np.log10(mu_fang_yen_arr)
        
    lam_fit, f_fit = fang_yen_fit()

    f_arr = f_fit(log_mu_arr)
        
    log_mu_min = np.min([log_mu_arr.min(), log_mu_fang_yen_arr.min()])
    log_mu_max = np.max([log_mu_arr.max(), log_mu_fang_yen_arr.max()])
                         
    log_mu_fit_arr = np.linspace(log_mu_min, log_mu_max, int(1e2))
    lam_fit_arr = lam_fit(log_mu_fit_arr)

    for i, xi in enumerate(log_xi_arr):

        lam_opt_list = []        
        A0_opt_list = []
        A_opt_list = []        
        
        lam_max_arr = np.zeros_like(mu_arr)
        A0_max_arr = np.zeros_like(mu_arr)
        A_max_arr = np.zeros_like(mu_arr)
        U_star_max_arr = np.zeros_like(mu_arr)
            
        for j in range(len(mu_arr)):
                        
            U = h5['U'][i, j, :]          
            W = h5['energies']['W'][i, j, :]            
            A = h5['A_max'][i, j, :]
                                    
            result = PostProcessor.comp_optimal_c_and_wavelength(
                U, W, A, c_arr, lam_arr)
                                        
            lam_opt_arr = result['lam_opt_arr'][:]
            A0_opt_arr = result['A0_opt_arr'][:]            
            A_opt_arr = result['A_opt_arr'][:]

            lam_opt_list.append(lam_opt_arr)
            A0_opt_list.append(A0_opt_arr)
            A_opt_list.append(A_opt_arr)
            
            # remeshed finer resolution
            U, W = result['U'], result['W']
                   
            U_star_max_arr[j] = result['U_max']                                           
            lam_max_arr[j] = result['lam_max']
            A0_max_arr[j] = result['A0_max']          
            A_max_arr[j]  = result['A_max'] 
                           
        U_max_arr = PostProcessor.U_star_to_U(U_star_max_arr, f_arr, L0)
        levels = result['levels']
                                                                                                                                                                                                                                             
        plt.figure(figsize = (10, 10))
        gs = plt.GridSpec(3,1)
        
        ax00 = plt.subplot(gs[0,0])
        ax10 = plt.subplot(gs[1,0])
        ax20 = plt.subplot(gs[2,0])
        
        ax00.semilogx(10**log_mu_fit_arr, lam_fit_arr, '-', c='k')
        ax00.semilogx(mu_fang_yen_arr, lam_fang_yen_arr, 's', c='k')
        ax00.semilogx(mu_arr, lam_max_arr, 'x', c='k')
                        
        ax10.semilogx(mu_arr, A_max_arr, 'x', c='k', label = r'$A_\mathrm{max}$')
        ax10.semilogx(mu_arr, A0_max_arr, 'x', c='r', label = r'$A0_\mathrm{max}$')                
        ax10.semilogx(mu_fang_yen_arr, A_fang_yen_arr, 's', c='k')
                
        ax20.semilogx(mu_arr, U_max_arr / L0, 'x', c = 'k')
                 
        for j, (mu, lam_opt_arr, A0_opt_arr, A_opt_arr, U_max) in enumerate(
            zip(mu_arr, lam_opt_list, A0_opt_list, A_opt_list, U_max_arr)):
        
            U_opt_arr = np.array(levels)*U_max / L0
        
            ax00.semilogx(mu * np.ones_like(lam_opt_arr), lam_opt_arr, 'o', c='k')
            ax10.semilogx(mu * np.ones_like(A_opt_arr), A_opt_arr, 'o', c='k')
            ax10.semilogx(mu * np.ones_like(A0_opt_arr), A0_opt_arr, 'o', c='r')
            ax20.semilogx(mu * np.ones_like(U_opt_arr), U_opt_arr, 'o', c='r')
                        
        plt.suptitle(fr'$\xi={xi}$', fontsize = 20)
        ax00.set_ylabel(r'$\lambda$', fontsize = 20)
        ax10.set_ylabel(r'$A$', fontsize = 20)
        ax20.set_ylabel(r'$U/L_0$', fontsize = 20)        
        ax20.set_xlabel(r'$\log(\mu)$', fontsize = 20)        
        plt.savefig(plot_dir / f'{str(i).zfill(2)}')
        plt.close()
            
    return

def plot_optimal_wave_length_for_different_eta_and_mu_1(h5_filename):
    '''
    Plots
    
    - optimal lambda swimming speed contours U = c * U_max where c <= 1

    over mu for different eta.

    Optimal wavelength decreases sigmoidally for higher fluid viscosity.    
    Larger internal damping eta should shift the sigmoids midpoints to the right.                          
    '''    
    plot_dir = fig_dir / 'optimal_lambda' / Path(h5_filename).stem 
    plot_dir.mkdir(parents = True, exist_ok = True)
        
    # Load data  
    h5 = h5py.File(sweep_dir / h5_filename, 'r')
    PG = ParameterGrid.init_pg_from_filepath(log_dir / h5.attrs['grid_filename'])
    to_quantities(PG.base_parameter)
    
    # Create time scale ratio mesh
    log_xi_arr = np.log10(PG.v_from_key('eta').magnitude / PG.base_parameter['E'].magnitude) 
    
    mu_arr = PG.v_from_key('mu')
    c_arr = PG.v_from_key('c')
    lam_arr = PG.v_from_key('lam')

    mu_fang_yen_arr, lam_fang_yen_arr, _ =fang_yen_data()
        
    lam_fit = fang_yen_fit()[0]

    log_mu_fang_yen_arr = np.log10(mu_fang_yen_arr)
    log_mu_fang_yen_fit_arr = np.linspace(
        log_mu_fang_yen_arr.min(), log_mu_fang_yen_arr.max(), 100)

    lam_arr_fang_yen_fit_arr = lam_fit(log_mu_fang_yen_fit_arr)

    lam_opt_avg_list = []
    lam_max_list = []
    U_opt_avg_list = []
    
    for i in range(len(log_xi_arr)):
    
        lam_opt_avg_arr = np.zeros_like(mu_arr)
        lam_max_arr = np.zeros_like(mu_arr)
        
        #TODO: Second axis with amplitude
        # A_opt_arr = np.zeros_like(mu_arr)
        # A_max_arr = np.zeros_like(mu_arr)
        
        for j in range(len(mu_arr)):
            
            U = h5['U'][i, j, :]          
            W = h5['ener    plot_optimal_wave_length_for_different_eta_and_mu(h5_filename)gies']['W'][i, j, :]
                                                                    
            result = PostProcessor.comp_optimal_c_and_wavelength(
                U, W, c_arr, lam_arr)
                                        
            lam_opt_arr = result['lam_opt_arr']
                    
            # remeshed finer resolution
            U, W = result['U'], result['W']
                                          
            lam_max, c_max = result['lam_max'], result['c_max']          
            U_max = result['U_max']
                                                                                              
            lam_opt_avg_arr[j] = lam_opt_arr[:].mean()
            lam_max_arr[j] = lam_max
            U_
                                
        lam_opt_avg_list.append(lam_opt_avg_arr)
        lam_max_list.append(lam_max_arr)
        
    plt.figure(figsize = (10, 10))
    gs = plt.GridSpec(1,1)
    ax00 = plt.subplot(gs[0,0])
    
    ax00.semilogx(10**log_mu_fang_yen_fit_arr, lam_arr_fang_yen_fit_arr, '-', c='k')
    ax00.semilogx(mu_fang_yen_arr, lam_fang_yen_arr, 's', c='k')
    
    for log_xi, lam_opt_avg_arr, lam_max_arr in zip(log_xi_arr, lam_opt_avg_list, lam_max_list):
        
        ax00.semilogx(mu_arr, lam_opt_avg_arr, 'o', label = fr'$\log(xi)={log_xi}$') 
        ax00.semilogx(mu_arr, lam_max_arr, 'x', c='k') 

    ax00.legend()
    plt.show()
            
    return

def plot_optimal_wave_length_for_different_C_eta_and_mu(h5_filename):
    '''
    Plots 
    
    - optimal lambda 
    - optimal curvature amplitude 
    
    over mu for swimming speed contours U=c*U_max where c <= 1 
    
    Create individual figure for every C and eta.    
    '''
    # Create directory for figures
    plot_dir = fig_dir / 'optimal_kinematics' / Path(h5_filename).stem 
    plot_dir.mkdir(parents = True, exist_ok = True)
    
    # Load data
    h5 = h5py.File(sweep_dir / h5_filename, 'r')
    PG = ParameterGrid.init_pg_from_filepath(log_dir / h5.attrs['grid_filename'])
    to_quantities(PG.base_parameter)

    L0 = PG.base_parameter['L0'].magnitude

    C_arr = PG.v_from_key('C')
    log_xi_arr = np.log10(PG.v_from_key('eta').magnitude / PG.base_parameter['E'].magnitude)             
    mu_arr = PG.v_from_key('mu')
    log_mu_arr = np.log10(mu_arr)    
    c_arr = PG.v_from_key('c')
    lam_arr = PG.v_from_key('lam')

    mu_fang_yen_arr, lam_fang_yen_arr, _, A_fang_yen_arr = fang_yen_data()
    log_mu_fang_yen_arr = np.log10(mu_fang_yen_arr)

    mu_gangon_arr, U_gangon_arr, _, _, _ = david_gagnon()
    
    lam_fit, f_fit, A_fit = fang_yen_fit()
    U_fit = fit_gagnon_sznitman()

    f_arr = f_fit(log_mu_arr)
        
    log_mu_min = np.min([log_mu_arr.min(), log_mu_fang_yen_arr.min()])
    log_mu_max = np.max([log_mu_arr.max(), log_mu_fang_yen_arr.max()])
                     
    log_mu_fit_arr = np.linspace(log_mu_min, log_mu_max, int(1e2))
    lam_fit_arr = lam_fit(log_mu_fit_arr)
    A_fit_arr = A_fit(log_mu_fit_arr)
    U_fit_arr = U_fit(log_mu_fit_arr)
    
    levels = np.arange(0.4, 0.91, 0.1)
    cmap = cm_dict['U']
    
    c_max = cmap(1.0)
    colors = cmap(levels)
    
    l = 0

    for i, C in enumerate(C_arr):        
        for j, xi in enumerate(log_xi_arr):

            lam_opt_list = []        
            A0_opt_list = []
            A_opt_list = []        
        
            lam_max_arr = np.zeros_like(mu_arr)
            A0_max_arr = np.zeros_like(mu_arr)
            A_max_arr = np.zeros_like(mu_arr)
            U_star_max_arr = np.zeros_like(mu_arr)
            
            for k in range(len(mu_arr)):
                            
                U = h5['U'][i, j, k, :]          
                W = h5['energies']['W'][i, j, k, :]            
                A = h5['A_max'][i, j, k, :]
                                    
                result = PostProcessor.comp_optimal_c_and_wavelength(
                    U, W, A, c_arr, lam_arr, levels)
                                        
                lam_opt_arr = result['lam_opt_arr'][:]
                A0_opt_arr = result['A0_opt_arr'][:]            
                A_opt_arr = result['A_opt_arr'][:]
    
                lam_opt_list.append(lam_opt_arr)
                A0_opt_list.append(A0_opt_arr)
                A_opt_list.append(A_opt_arr)
            
                # remeshed finer resolution
                U, W = result['U'], result['W']
                       
                U_star_max_arr[k] = result['U_max']                                           
                lam_max_arr[k] = result['lam_max']
                A0_max_arr[k] = result['A0_max']          
                A_max_arr[k]  = result['A_max'] 
                           
            U_max_arr = PostProcessor.U_star_to_U(U_star_max_arr, f_arr, L0)
                                                                                                                                                                                                                                                             
            plt.figure(figsize = (10, 10))
            gs = plt.GridSpec(3,1)
    
            ax00 = plt.subplot(gs[0,0])
            ax10 = plt.subplot(gs[1,0])
            ax20 = plt.subplot(gs[2,0])
                        
            ax00.semilogx(10**log_mu_fit_arr, lam_fit_arr, '-', c='k')
            ax00.semilogx(mu_fang_yen_arr, lam_fang_yen_arr, 's', c='k')
            #ax00.semilogx(mu_arr, lam_max_arr, 'x', c='k')                    
                            
            ax10.semilogx(10**log_mu_fit_arr, A_fit_arr, '-', c='k')        
            #ax10.semilogx(mu_arr, A_max_arr, 'x', c='k', label = r'$A_\mathrm{max}$')
            ax10.semilogx(mu_fang_yen_arr, A_fang_yen_arr, 's', c='k')
            
            # Plot preferred curvature amplitude
            #ax10.semilogx(mu_arr, A0_max_arr, 'x', c='r', label = r'$A0_\mathrm{max}$')                            
            # ax10.scatter(log_mu_arr, lam_max_arr, marker = '^', 
            #     c = colors, edgecolor = 'k')            
                    
            #ax20.semilogx(mu_arr, U_max_arr / L0, 'x', c = 'k')
            
            ax20.semilogx(mu_gangon_arr, U_gangon_arr, 's', c='k') 
            ax20.semilogx(10**log_mu_fit_arr, U_fit_arr, '-', c='k') 
                        
            for mu, lam_opt_arr, A0_opt_arr, A_opt_arr, U_max in zip(
                mu_arr, lam_opt_list, A0_opt_list, A_opt_list, U_max_arr):
            
                U_opt_arr = np.array(levels)*U_max / L0
            
                # ax00.semilogx(mu * np.ones_like(lam_opt_arr), lam_opt_arr, 'o', c='k')
                # ax10.semilogx(mu * np.ones_like(A_opt_arr), A_opt_arr, 'o', c='k')
                # ax10.semilogx(mu * np.ones_like(A0_opt_arr), A0_opt_arr, 'o', c='r')
                # ax20.semilogx(mu * np.ones_like(U_opt_arr), U_opt_arr, 'o', c='r')

                ax00.scatter(mu * np.ones_like(lam_opt_arr), lam_opt_arr, marker ='o', 
                    c=colors, edgecolor = 'k')
                ax10.scatter(mu * np.ones_like(A_opt_arr), A_opt_arr, marker ='o', 
                    c=colors, edgecolor = 'k')
                ax20.scatter(mu * np.ones_like(U_opt_arr), U_opt_arr, marker ='o', 
                    c=colors, edgecolor = 'k')
            
            ax00.scatter(mu_arr, lam_max_arr, marker = '^', 
                c = [c_max], edgecolor = 'k')
            
            ax10.scatter(mu_arr, A_max_arr, marker = '^', 
                c = [c_max], edgecolor = 'k')

            ax20.scatter(mu_arr, U_max_arr / L0, marker = '^', 
                c = [c_max], edgecolor = 'k')
                                                                      
            plt.suptitle(fr'$C={C}, \xi={xi}$', fontsize = 20)
            ax00.set_ylabel(r'$\lambda$', fontsize = 20)
            ax10.set_ylabel(r'$A$', fontsize = 20)
            ax20.set_ylabel(r'$U/L_0$', fontsize = 20)        
            ax20.set_xlabel(r'$\log(\mu)$', fontsize = 20)        
            plt.savefig(plot_dir / f'{str(l).zfill(2)}')
            plt.close()
            l += 1
            
    return

def plot_error_C_eta_lam_c(h5_filename):
    '''    
    Plot 
    
    - error between simulation results and experimental data     
    
    for different C, eta and mu.     
    '''
    
    # Create directory for figures
    plot_dir = fig_dir / 'optimal_kinematics' / Path(h5_filename).stem 
    plot_dir.mkdir(parents = True, exist_ok = True)
    
    # Load data
    h5 = h5py.File(sweep_dir / h5_filename, 'r')
    PG = ParameterGrid.init_pg_from_filepath(log_dir / h5.attrs['grid_filename'])
    to_quantities(PG.base_parameter)

    L0 = PG.base_parameter['L0'].magnitude

    C_arr = PG.v_from_key('C')
    log_xi_arr = np.log10(PG.v_from_key('eta').magnitude / PG.base_parameter['E'].magnitude)             
    mu_arr = PG.v_from_key('mu')
    log_mu_arr = np.log10(mu_arr)    
    c_arr = PG.v_from_key('c')
    lam_arr = PG.v_from_key('lam')
    
    lam_fit, f_fit, A_fit = fang_yen_fit()
    
    david_gagnon()
    



    log_mu_fit_arr = np.linspace(log_mu_min, log_mu_max, int(1e2))
    lam_fit_arr = lam_fit(log_mu_fit_arr)
    A_fit_arr = A_fit(log_mu_fit_arr)


    error = np.zeros((len(C_arr), len(log_xi_arr)))

    levels = np.arange(0.3, 0.1, 1.0)

    for i in range(C_arr):
        for j in range(log_xi_arr):
                        
            for k in range(mu_arr):
                #Compute error
                
                U = h5['U'][i, j, k, :]          
                W = h5['energies']['W'][i, j, k, :]            
                A = h5['A_max'][i, j, k, :]
                                    
                result = PostProcessor.comp_optimal_c_and_wavelength(
                    U, W, A, c_arr, lam_arr, levels)
                                        
                lam_opt_arr = result['lam_opt_arr'][:]
                A0_opt_arr = result['A0_opt_arr'][:]            
                A_opt_arr = result['A_opt_arr'][:]
    
                lam_opt_list.append(lam_opt_arr)
                A0_opt_list.append(A0_opt_arr)
                A_opt_list.append(A_opt_arr)
            
                # remeshed finer resolution
                U, W = result['U'], result['W']
                       
                U_star_max_arr[k] = result['U_max']                                           
                lam_max_arr[k] = result['lam_max']
                A0_max_arr[k] = result['A0_max']          
                A_max_arr[k]  = result['A_max'] 
                
                
    
    return

    

def plot_swimming_speed_C_a_b(h5_filename):
    
    h5 = h5py.File(sweep_dir / h5_filename, 'r')
    PG = ParameterGrid.init_pg_from_filepath(log_dir / h5.attrs['grid_filename'])
    plot_dir = fig_dir / 'kinematics' / 'C' / 'U_over_a_b' 
    plot_dir.mkdir(parents = True, exist_ok = True)

    # Create time scale ratio mesh
    C_arr = PG.v_from_key('C')
    log_a_arr = np.log10(PG.v_from_key('a'))
    log_b_arr = np.log10(PG.v_from_key('b'))
    
    f = 1.0 / PG.base_parameter['T_c'][0]
        
    log_a_grid, log_b_grid = np.meshgrid(log_a_arr, log_b_arr) 

    for i, C in enumerate(C_arr):
        
        ax00 = plt.subplot(111)
        
        U = h5['U'][i, :, :].T * f
                        
        CS = ax00.contourf(log_a_grid, log_b_grid, U, 
            cmap = cm_dict['U'])
        ax00.contour(log_a_grid, log_b_grid, U,   
            linestyles = '-', colors = ('k',))        
        plt.colorbar(CS, ax = ax00, orientation = 'horizontal', 
            label = r'$U / L0$')
        
        plt.suptitle(f'$C={C}$')
        
        ax00.set_xlabel(r'$\log(a)$', fontsize = 20)
        ax00.set_ylabel(r'$\log(b)$', fontsize = 20)
        
        plt.savefig(plot_dir / f'{str(i).zfill(2)}')
        plt.close()
    
    return

def plot_swimming_speed_C_eta_mu_fang_yen(h5_filename):

    h5 = h5py.File(sweep_dir / h5_filename, 'r')
    PG = ParameterGrid.init_pg_from_filepath(log_dir / h5.attrs['grid_filename'])

    plot_dir = fig_dir / 'kinematics' / 'K' / Path(h5_filename).stem 
        
    plot_dir.mkdir(parents = True, exist_ok = True)
                
    C_arr = PG.v_from_key('C')
    eta_arr = PG.v_from_key('eta')    
    mu_arr = PG.v_from_key('mu')
    
    f_arr = 1.0 / PG.v_from_key('T_c')
            
    # mu_arr_1, U_arr_1, _ = sznitzman()    
    # mu_arr_2, U_arr_2, _, _, _ = david_gagnon()
                    
    for j, C in enumerate(C_arr):
        
        ax00 = plt.subplot(111)        
        plt.suptitle(f'$C={C}$')
        
        for i, eta in enumerate(eta_arr):
                                                            
            U_star_arr = h5['U'][j, i, :]
            U_arr = U_star_arr * f_arr 
            ax00.semilogx(mu_arr, U_arr, label = f'$\eta={eta}$')

        ax00.set_ylabel(f'$U/L_0$', fontsize = 20)    
        ax00.set_xlabel(f'$\log(\mu)$', fontsize = 20)
        ax00.legend(fontsize = 16)
        ax00.plo
        
                
        plt.savefig(plot_dir / f'{str(i).zfill(2)}')
        plt.close()

    return

def plot_optima_on_contours_C_mu(h5_filename: str):
    '''
    Plots 
        - swimming speed contours over lambda and c 
        - mechanical work over lambda and c
        - dissipated fluid energy over lambda and c
        - dissipated internal energy over lambda and c
        
    for different internal fluid viscosity mu, eta     
    '''    
    
    # Create directory for figures
    plot_dir = fig_dir / 'optimal_kinematics' / 'contours' / Path(h5_filename).stem 
    plot_dir.mkdir(parents = True, exist_ok = True)
    
    # Load data
    h5 = h5py.File(sweep_dir / h5_filename, 'r')
    PG = ParameterGrid.init_pg_from_filepath(log_dir / h5.attrs['grid_filename'])
    to_quantities(PG.base_parameter)
    
    C_arr = PG.v_from_key('C')
    mu_arr = PG.v_from_key('mu')
    log_mu_arr = np.round(np.log10(mu_arr), 2)
        
    c_arr = PG.v_from_key('c')
    lam_arr = PG.v_from_key('lam')

    k = 0
    
    for i, C in enumerate(C_arr):
        for j, log_mu in enumerate(log_mu_arr):
                        
            U = h5['U'][i, j, :]          
            A = h5['A_max'][i, j, :]
            W = h5['energies']['W'][i, j, :]            
            D_I = h5['energies']['D_I'][i, j, :]            
            D_F = h5['energies']['D_F'][i, j, :]            
                                                
            plot_optima_on_contours(
                f'{str(k).zfill(2)}', 
                plot_dir, 
                fr'$C={C}, \log(\mu)={log_mu}$', 
                c_arr, 
                lam_arr, 
                U, 
                W, 
                A, 
                D_I, 
                D_F)
            
            k += 1
            
    return

def plot_maximum_swimming_speed_C_mu_c_lam(h5_filename):
    '''
    Plots 
    
    - maximum swimming speed U_max over mu
    
    for different values of C.
    
    '''    
    h5 = h5py.File(sweep_dir / h5_filename, 'r')
    PG = ParameterGrid.init_pg_from_filepath(log_dir / h5.attrs['grid_filename'])

    plot_dir = fig_dir / 'kinematics' / 'K' 
    plot_dir.mkdir(parents = True, exist_ok = True)
                
    C_arr = PG.v_from_key('C')
    mu_arr = PG.v_from_key('mu')
    log_mu_arr = np.log10(mu_arr)
    
    lam_arr = PG.v_from_key('lam')
    c_arr = PG.v_from_key('c')
        
    f_arr = 1.0 / PG.v_from_key('T_c')
    
    ax00 = plt.subplot(111)
    
    for i, C in enumerate(C_arr):
        
        U_star_max_arr = np.zeros_like(log_mu_arr) 
                                
        for j in range(len(log_mu_arr)):            
                
            U_star = h5['U'][i, j][:]
            U_star_interp = RectBivariateSpline(lam_arr, c_arr, U_star.T)    

            # Use lam and c where U is maximal as initial guess 
            j_max, i_max = np.unravel_index(U_star.argmax(), U_star.shape)
            lam_max_0, c_max_0 = lam_arr[i_max], c_arr[j_max]                    
        
            # Minimize -U_interp to find lam and c where U is maximal
            res = minimize(lambda x: -U_star_interp(x[0], x[1])[0], [lam_max_0, c_max_0], 
                bounds=[(lam_arr.min(), lam_arr.max()), (c_arr.min(), c_arr.max())])
                
            lam_max, c_max = res.x[0], res.x[1]                  
            U_star_max_arr[j] = U_star_interp(lam_max, c_max)
        
        U_max_over_L0_arr = U_star_max_arr * f_arr 
        
        ax00.semilogx(mu_arr, U_max_over_L0_arr, label = f'K={C}')

    ax00.set_ylabel(f'$U/L_0$', fontsize = 20)    
    ax00.set_xlabel(f'$\log(\mu)$', fontsize = 20)
    ax00.legend(fontsize = 16)
        
    figname = Path(h5_filename).stem + '.png'
        
    plt.savefig(plot_dir / figname)
    plt.close()
    
    return

def plot_optimal_wave_length_for_different_C_and_mu_0(h5_filename):
    '''
    Plots 
    
    - optimal lambda 
    - optimal curvature amplitude 
    
    over mu for swimming speed contours U=c*U_max where c <= 1 
    
    Create individual figure for every eta.    
    '''
    # Create directory for figures
    plot_dir = fig_dir / 'optimal_kinematics' / Path(h5_filename).stem 
    plot_dir.mkdir(parents = True, exist_ok = True)
    
    # Load data
    h5 = h5py.File(sweep_dir / h5_filename, 'r')
    PG = ParameterGrid.init_pg_from_filepath(log_dir / h5.attrs['grid_filename'])
    to_quantities(PG.base_parameter)

    L0 = PG.base_parameter['L0'].magnitude

    C_arr = PG.v_from_key('C')     
    mu_arr = PG.v_from_key('mu')
    log_mu_arr = np.log10(mu_arr)    
    c_arr = PG.v_from_key('c')
    lam_arr = PG.v_from_key('lam')

    mu_fang_yen_arr, lam_fang_yen_arr, _, A_fang_yen_arr = fang_yen_data()
    log_mu_fang_yen_arr = np.log10(mu_fang_yen_arr)
        
    lam_fit, f_fit = fang_yen_fit()

    f_arr = f_fit(log_mu_arr)
        
    log_mu_min = np.min([log_mu_arr.min(), log_mu_fang_yen_arr.min()])
    log_mu_max = np.max([log_mu_arr.max(), log_mu_fang_yen_arr.max()])
                         
    log_mu_fit_arr = np.linspace(log_mu_min, log_mu_max, int(1e2))
    lam_fit_arr = lam_fit(log_mu_fit_arr)

    for i, C in enumerate(C_arr):

        lam_opt_list = []        
        A0_opt_list = []
        A_opt_list = []        
        
        lam_max_arr = np.zeros_like(mu_arr)
        A0_max_arr = np.zeros_like(mu_arr)
        A_max_arr = np.zeros_like(mu_arr)
        U_star_max_arr = np.zeros_like(mu_arr)
            
        for j in range(len(mu_arr)):
                        
            U = h5['U'][i, j, :]          
            W = h5['energies']['W'][i, j, :]            
            A = h5['A_max'][i, j, :]
                                    
            result = PostProcessor.comp_optimal_c_and_wavelength(
                U, W, A, c_arr, lam_arr)
                                        
            lam_opt_arr = result['lam_opt_arr'][:]
            A0_opt_arr = result['A0_opt_arr'][:]            
            A_opt_arr = result['A_opt_arr'][:]

            lam_opt_list.append(lam_opt_arr)
            A0_opt_list.append(A0_opt_arr)
            A_opt_list.append(A_opt_arr)
            
            # remeshed finer resolution
            U, W = result['U'], result['W']
                   
            U_star_max_arr[j] = result['U_max']                                           
            lam_max_arr[j] = result['lam_max']
            A0_max_arr[j] = result['A0_max']          
            A_max_arr[j]  = result['A_max'] 
                           
        U_max_arr = PostProcessor.U_star_to_U(U_star_max_arr, f_arr, L0)
        levels = result['levels']
                                                                                                                                                                                                                                                     
        plt.figure(figsize = (10, 10))
        gs = plt.GridSpec(3,1)
        
        ax00 = plt.subplot(gs[0,0])
        ax10 = plt.subplot(gs[1,0])
        ax20 = plt.subplot(gs[2,0])
        
        ax00.semilogx(10**log_mu_fit_arr, lam_fit_arr, '-', c='k')
        ax00.semilogx(mu_fang_yen_arr, lam_fang_yen_arr, 's', c='k')
        ax00.semilogx(mu_arr, lam_max_arr, 'x', c='k')
                        
        ax10.semilogx(mu_arr, A_max_arr, 'x', c='k', label = r'$A_\mathrm{max}$')
        ax10.semilogx(mu_arr, A0_max_arr, 'x', c='r', label = r'$A0_\mathrm{max}$')                
        ax10.semilogx(mu_fang_yen_arr, A_fang_yen_arr, 's', c='k')
                
        ax20.semilogx(mu_arr, U_max_arr / L0, 'x', c = 'k')
                 
        for j, (mu, lam_opt_arr, A0_opt_arr, A_opt_arr, U_max) in enumerate(
            zip(mu_arr, lam_opt_list, A0_opt_list, A_opt_list, U_max_arr)):
        
            U_opt_arr = np.array(levels)*U_max / L0
        
            ax00.semilogx(mu * np.ones_like(lam_opt_arr), lam_opt_arr, 'o', c='k')
            ax10.semilogx(mu * np.ones_like(A_opt_arr), A_opt_arr, 'o', c='k')
            ax10.semilogx(mu * np.ones_like(A0_opt_arr), A0_opt_arr, 'o', c='r')
            ax20.semilogx(mu * np.ones_like(U_opt_arr), U_opt_arr, 'o', c='r')
                        
        plt.suptitle(fr'$C={C}$', fontsize = 20)
        ax00.set_ylabel(r'$\lambda$', fontsize = 20)
        ax10.set_ylabel(r'$A$', fontsize = 20)
        ax20.set_ylabel(r'$U/L_0$', fontsize = 20)        
        ax20.set_xlabel(r'$\log(\mu)$', fontsize = 20)        
        plt.savefig(plot_dir / f'{str(i).zfill(2)}')
        plt.close()
            
    return

def plot_chemograms_fang_yen_1(h5_filename):
    '''
    Plot chemograms to check if there is different between the 
    preferred and real curvature 
    '''
    # Load simulation data
    h5 = h5py.File(sweep_dir / h5_filename, 'r')
    PG = ParameterGrid.init_pg_from_filepath(log_dir / 
        h5.attrs['grid_filename'])
    
    C_arr = PG.v_from_key('C')
    eta_arr = PG.v_from_key('eta')
        
    mu_arr = PG.v_from_key('mu')

    # Select drag coefficient ratio to plot
    eta = 0.01
    eta_idx = np.abs(eta_arr - eta).argmin()
        
    C0 = 1.5        
    C_idx = np.abs(C_arr - C0).argmin()

    save_dir = fig_dir / Path(h5_filename).stem /'chemograms'

    for i, mu in enumerate(mu_arr):

        k = h5['k'][C_idx, eta_idx, i, :]

        ax = plt.subplot(111)        
        plot_scalar_field(ax, 
            k, title = [f'$\mu={mu}$'])        
        plt.savefig(save_dir / '{str(i).zfill(2)}.png')

    return
    
def plot_rikmenspoel_0(h5_f):
    
    #===============================================================================
    # Load data 
    #===============================================================================    
    h5 = h5py.File(sweep_dir / h5_f, 'r')
    PG = ParameterGrid.init_pg_from_filepath(log_dir / h5.attrs['grid_filename'])
    to_quantities(PG.base_parameter)
    
    T_c_arr = PG.v_from_key('T_c')
    f_arr = 1.0 / T_c_arr        
    lam_arr = PG.v_from_key('lam')

    U_star_arr = h5['U'][:]
    U_arr = f_arr*h5['U'][:]

    gs = plt.GridSpec(2,1)
    ax0 = plt.subplot(gs[0])
    ax0.plot(f_arr, U_arr, 'o-')
    ax0_twin = ax0.twinx()
    ax0_twin.plot(f_arr, U_star_arr, 'o-', c='r')
    
    plt.show()
    
    return
        
def plot_rikmenspoel(h5_f_c_lam):
    '''
    Plots contours for sperm simulations
    '''

    # Load data
    h5 = h5py.File(sweep_dir / h5_f_c_lam, 'r')
    PG = ParameterGrid.init_pg_from_filepath(log_dir / h5.attrs['grid_filename'])
    to_quantities(PG.base_parameter)
    
    T_c_arr = PG.v_from_key('T_c')
    f_arr = 1.0 / T_c_arr        
    c_arr = PG.v_from_key('c')
    lam_arr = PG.v_from_key('lam')
    
    lam_grid, c_grid = np.meshgrid(lam_arr, c_arr)       

    # Create directory for figures
    plot_dir = fig_dir / 'optimal_kinematics' / Path(h5_f_c_lam).stem 
    plot_dir.mkdir(parents = True, exist_ok = True)

    for i, f in enumerate(f_arr):
    
        U = h5['U'][i, :]
        W = h5['energies']['W'][i, :]
        A = h5['A_max'][i, :]

        opt_res = PostProcessor.comp_optimal_c_and_wavelength(U, W, A, c_arr, lam_arr)
        
        lam_max, c_max = opt_res['lam_max'], opt_res['c_max']         
        lam_opt_arr, c_opt_arr = opt_res['lam_opt_arr'], opt_res['c_opt_arr']
        opt_levels = opt_res['levels']
        
        #------------------------------------------------------------------------------ 
        # Plotting
    
        plt.figure(figsize = (10, 10))
        gs = plt.GridSpec(1,2)
        
        ax00 = plt.subplot(gs[0, 0])
        ax01 = plt.subplot(gs[0, 1])
            
        levels = np.arange(0, 1.01, 0.1)
        
        # U/U_max contour
        CS = ax00.contourf(lam_grid, c_grid, U / U.max(), 
            levels = levels, cmap = cm_dict['U'])
        ax00.contour(lam_grid, c_grid, U / U.max(),   
            levels = levels, linestyles = '-', colors = ('k',))        
        cbar = plt.colorbar(CS, ax = ax00, orientation = 'horizontal', 
            label = r'$U / U_\mathrm{max}$')                        
             
        # W contour
        CS = ax01.contourf(lam_grid, c_grid, np.log10(W), 
            cmap = cm_dict['W'])        
        ax01.contour(lam_grid, c_grid, U / U.max(),   
            levels = levels, linestyles = '--', colors = ('k',))        
        plt.colorbar(CS, ax = ax01, orientation = 'horizontal', 
            label = r'$\log(W/\max(W))$')
        
        ax01.scatter(lam_opt_arr, c_opt_arr, 
            marker ='o', c = cbar.cmap(opt_levels), edgecolors = 'k')
        
        # Plot lambda and c for which swimming speed is maximal 
        for ax in [ax00, ax01]:        
            ax.scatter(lam_max, c_max, marker='^', c = [cbar.cmap(1.0)],
                edgecolor = 'k')        
        
        #=======================================================================
        # Layout 
        #=======================================================================
        
        plt.suptitle(f'f={round(f, 0)}')
    
        ax00.set_xlabel('$\lambda$')
        ax00.set_ylabel('$c$')
        ax01.set_xlabel('$\lambda$')
        ax01.set_ylabel('$c$')
                                    
        plt.tight_layout()    
        plt.savefig(plot_dir / f'{str(i).zfill(2)}.png')
        plt.close()

    return
        
def plot_rikmenspoel_2(h5_f_c_lam):
    '''
    Plot experimantal and wavelength for sperm
    '''
        
    #===============================================================================
    # Simulation data
    #===============================================================================
    h5 = h5py.File(sweep_dir / h5_f_c_lam, 'r')
    PG = ParameterGrid.init_pg_from_filepath(log_dir / h5.attrs['grid_filename'])
    to_quantities(PG.base_parameter)
        
    T_c_arr = PG.v_from_key('T_c')
    f_arr = 1.0 / T_c_arr        
    c_arr = PG.v_from_key('c')
    lam_arr = PG.v_from_key('lam')
    
    #===========================================================================
    # Experimental data 
    #===========================================================================
    
    exp_data = rikmenspoel_1978()
    
    # Create directory for figures
    plot_dir = fig_dir / 'optimal_kinematics' 
    plot_dir.mkdir(parents = True, exist_ok = True)
    
    levels = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    lam_max_arr = np.zeros(len(f_arr))
    c_max_arr = np.zeros_like(lam_max_arr)
        
    lam_opt_mat = np.zeros((len(levels), len(f_arr)))
    c_opt_mat = np.zeros_like(lam_opt_mat)
         
    for i, f in enumerate(f_arr):
    
        U = h5['U'][i, :]
        W = h5['energies']['W'][i, :]
        A = h5['A_max'][i, :]

        opt_res = PostProcessor.comp_optimal_c_and_wavelength(
            U, W, A, c_arr, lam_arr, levels)

        lam_max_arr[i] = opt_res['lam_max']
        c_max_arr[i] = opt_res['c_max']
        
        lam_opt_mat[:, i] = opt_res['lam_opt_arr']
        c_opt_mat[:, i] = opt_res['c_opt_arr']
        

    # Get optimal bands 
    lam_opt_min_arr = lam_opt_mat.min(axis = 0)
    lam_opt_max_arr = lam_opt_mat.max(axis = 0)

    # A_opt_min_arr = A_opt_mat.min(axis = 0)
    # A_opt_max_arr = A_opt_mat.max(axis = 0)


    #===============================================================================
    # Plotting 
    #===============================================================================
            
    gs = plt.GridSpec(1, 1)        
    ax00 = plt.subplot(gs[0])
    
    lam_fit = fit_rikmenspoel_1978()[0]        

    f_fit_arr = np.linspace(
        exp_data['f'].magnitude.min(), exp_data['f'].magnitude.max(), int(1e3))
    ax00.plot(f_fit_arr, lam_fit(f_fit_arr), '-', c='k')
    ax00.plot(exp_data['f'].magnitude, 
        exp_data['lam_star'].magnitude, 's', c='k')    
    ax00.plot(f_arr, lam_max_arr, 'x')
    
    ax00.fill_between(f_arr, lam_opt_max_arr, lam_opt_min_arr,
        interpolate=True, color = 'r', alpha = 0.5) 
        
    #=======================================================================
    # Layout 
    #=====================================================================
        
    ax00.set_xlabel('$f$')
    ax00.set_ylabel('$\lambda$')
    
                        
    plt.savefig(plot_dir / f'{Path(h5_f_c_lam).stem}.png')
    plt.close()
 
    return
                       
def plot_rikmenspoel_chemograms(h5_f_lam):
    '''
    Plot chemograms to epxlore how the simulator handles the 
    decreasing curvature amplitude
    '''
    #===============================================================================
    # Load data 
    #===============================================================================    
    h5 = h5py.File(sweep_dir / h5_f_lam, 'r')
    PG = ParameterGrid.init_pg_from_filepath(log_dir / 
        h5.attrs['grid_filename'])
    
    f_arr = 1.0 / PG.v_from_key('T_c')
    lam_arr = PG.v_from_key('lam')


    #===============================================================================
    # Plotting 
    #===============================================================================
                
    save_dir = fig_dir / 'sperm' / 'chemograms' / Path(h5_f_lam).stem 
    save_dir.mkdir(parents = True, exist_ok = True)        
    shape = (len(f_arr), len(lam_arr))
            
    for i, (k, k0) in enumerate(zip(h5['FS']['k'], h5['CS']['k0'])):
          
        print(h5['FS']['k'].shape[0]-i)
          
        m, n = np.unravel_index(i, shape)          
                
        plot_multiple_scalar_fields(
            [k0[:,0,:], k[:,0,:]],
            cmaps = [plt.cm.seismic, plt.cm.seismic],            
            grid_layout=(2, 1))
        
        plt.suptitle(f'$f={np.round(f_arr[m].magnitude, 0)}, \lambda={np.round(lam_arr[n], 2)}$')               
        plt.savefig(save_dir / f'{str(i).zfill(3)}.png')
            
    return    


def plot_rikmenspoel_3(h5_f_lam):
    '''
    Plot wavelength which optimizes speed
    '''

    #===============================================================================
    # Simulation data
    #===============================================================================
    h5 = h5py.File(sweep_dir / h5_f_lam, 'r')
    PG = ParameterGrid.init_pg_from_filepath(log_dir / h5.attrs['grid_filename'])
    to_quantities(PG.base_parameter)
        
    T_c_arr = PG.v_from_key('T_c')
    f_arr = 1.0 / T_c_arr        
    lam_arr = PG.v_from_key('lam')
    
    #===========================================================================
    # Experimental data 
    #===========================================================================
    
    exp_data = rikmenspoel_1978()
    lam_fit = fit_rikmenspoel_1978()[0]
    
    # Create directory for figures
    plot_dir = fig_dir / 'optimal_kinematics' / 'sperm'
    plot_dir.mkdir(parents = True, exist_ok = True)
    
    lam_max_arr = np.zeros(len(f_arr))
    lam_opt_arr = np.zeros(len(f_arr))

    gs = plt.GridSpec(1, len(f_arr))
                  
    for i, _ in enumerate(f_arr):
    
        ax = plt.subplot(gs[i])    
        ax.plot(lam_arr, h5['U'][i, :])
    
        U = h5['U'][i, :]
        W = h5['energies']['W'][i, :]
        
        W_over_S = W / U
        
        lam_max_arr[i] = lam_arr[U.argmax()]
        lam_opt_arr[i] = lam_arr[W_over_S.argmin()]
                        
    #===============================================================================
    # Plotting 
    #===============================================================================
            
    gs = plt.GridSpec(1, 1)        
    ax00 = plt.subplot(gs[0])
            
    f_fit_arr = np.linspace(
        exp_data['f'].magnitude.min(), exp_data['f'].magnitude.max(), int(1e3))
    ax00.plot(f_fit_arr, lam_fit(f_fit_arr), '-', c='k')
    ax00.plot(exp_data['f'].magnitude, exp_data['lam_star'].magnitude, 's', c='k')    
    ax00.plot(f_arr, lam_max_arr, 'x')
    # ax00.plot(f_arr, lam_opt_arr, 'o')
            
    #=======================================================================
    # Layout 
    #=====================================================================
        
    ax00.set_xlabel('$f$')
    ax00.set_ylabel('$\lambda$')

    plt.show()
                            
    plt.savefig(plot_dir / f'{Path(h5_f_lam).stem}.png')
    plt.close()
 
    return

def plot_rikmenspoel_4(h5_xi_f_lam):
    '''
    Compare wavelength for maximum speed against experimental data
    '''
    
    h5 = h5py.File(sweep_dir / h5_xi_f_lam, 'r')
    PG = ParameterGrid.init_pg_from_filepath(log_dir / h5.attrs['grid_filename'])

    U_arr = h5['U'] 
    W_arr = h5['energies']['W']

    f_arr = 1.0 / PG.v_from_key('T_c')    
    lam_arr = PG.v_from_key('lam')
    eta_arr = PG.v_from_key('eta')
    xi_arr = eta_arr.magnitude / PG.base_parameter['E'].magnitude
    log_xi_arr = np.round(np.log10(xi_arr), 2)
    
    data = rikmenspoel_1978()
    
    levels = [0.8, 0.9]

    gs = plt.GridSpec(len(levels)+1, 1)
    ax0 = plt.subplot(gs[0])
        
    axes = []
        
    for i, (log_xi, U, W) in enumerate(zip(log_xi_arr, U_arr, W_arr)):
    
        results = PostProcessor.comp_optimal_wavelength(U, W, f_arr, lam_arr, levels) 
        lam_max_arr = results[0]
        lam_opt_mat = results[1]    
        ax0.plot(f_arr, lam_max_arr, label = fr'$\xi={log_xi}$')

        for j, level in enumerate(levels):
            ax = plt.subplot(gs[j+1])                        
            ax.plot(f_arr, lam_opt_mat[j, :], '-o')
            ax.set_title(f'level={level}')
            axes.append(ax)
    
    ax0.legend()
    ax0.plot(data['f'].magnitude , data['lam_star'].magnitude, 's', c='k')

    # ax0.set_xlim(20, 50)

    for ax in axes:
        ax.plot(data['f'].magnitude , data['lam_star'].magnitude, 's', c='k')
        # ax.set_xlim(20, 50)
    plt.show()
    
    return 
    


def plot_rikmenspoel_power_balance(h5_f_lam):
    
    h5 = h5py.File(sweep_dir / h5_f_lam, 'r')
    PG = ParameterGrid.init_pg_from_filepath(log_dir / h5.attrs['grid_filename'])

    D_I_dot_arr = h5['FS']['D_I_dot'][:]
    D_F_dot_arr = h5['FS']['D_F_dot'][:]
    D_dot_arr = D_I_dot_arr + D_F_dot_arr     
    V_dot_arr = h5['FS']['V_dot'][:]
    W_dot_arr = h5['FS']['W_dot'][:]
    
    t = h5['t'][:]
    dt = t[1] - t[0]
    
    for D_dot, V_dot, W_dot in  zip(D_dot_arr, V_dot_arr, W_dot_arr):
    
        gs = plt.GridSpec(2,1)
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])
        ax0.plot(t, D_dot, label = r'$\dot{D}$')
        ax0.plot(t, W_dot, label = '$\dot{W}$')
        ax0.plot(t, V_dot, label = '$\dot{V}$')
        plt.legend()
        ax1.plot(t, D_dot + W_dot + V_dot)
    
        W = trapezoid(W_dot[t >= 4], dx = dt)
        print(W)
    
        plt.show()
        plt.close()


def plot_rikmenspoel_power_balance_compare(h5_f_gmso_true, h5_f_gmso_false):
    '''
    Compare power balance for simulation with and without gradual muscle onset 
    at the head an tale.
    
    Check if the power balance without gradual muscle onset is not fulfilled due
    to missing boundary condition
    '''

    h5_1 = h5py.File(sweep_dir / h5_f_gmso_true, 'r')
    PG_1 = ParameterGrid.init_pg_from_filepath(log_dir / h5_1.attrs['grid_filename'])

    h5_2 = h5py.File(sweep_dir / h5_f_gmso_false, 'r')
    PG_2 = ParameterGrid.init_pg_from_filepath(log_dir / h5_2.attrs['grid_filename'])

    f_arr = 1.0 / PG_1.v_from_key('T_c')

    D_I_dot_arr_1 = h5_1['FS']['D_I_dot'][:]
    D_F_dot_arr_1 = h5_1['FS']['D_F_dot'][:]
    D_dot_arr_1 = D_I_dot_arr_1 + D_F_dot_arr_1     
    V_dot_arr_1 = h5_1['FS']['V_dot'][:]
    W_dot_arr_1 = h5_1['FS']['W_dot'][:]

    D_I_dot_arr_2 = h5_1['FS']['D_I_dot'][:]
    D_F_dot_arr_2 = h5_1['FS']['D_F_dot'][:]
    D_dot_arr_2 = D_I_dot_arr_2 + D_F_dot_arr_2     
    V_dot_arr_2 = h5_2['FS']['V_dot'][:]
    W_dot_arr_2 = h5_2['FS']['W_dot'][:]

    t = h5_1['t'][:]
    dt = t[1] - t[0]
    
    save_dir = fig_dir / 'sperm' / 'power_balance' / Path(h5_f_gmso_true).stem
    save_dir.mkdir(parents = True, exist_ok = True)
    
    k = 0
    
    for f, D_dot_1, V_dot_1, W_dot_1, D_dot_2, V_dot_2, W_dot_2 in zip(f_arr,
        D_dot_arr_1, V_dot_arr_1, W_dot_arr_1, D_dot_arr_2, V_dot_arr_2, W_dot_arr_2):
    
        gs = plt.GridSpec(2,2)
        ax00 = plt.subplot(gs[0, 0])
        ax10 = plt.subplot(gs[1, 0])
        ax01 = plt.subplot(gs[0, 1])
        ax11 = plt.subplot(gs[1, 1])
        
        ax00.plot(t, D_dot_1, label = r'$\dot{D}$')
        ax00.plot(t, W_dot_1, label = '$\dot{W}$')
        ax00.plot(t, V_dot_1, label = '$\dot{V}$')
        ax00.legend()
        W_amplitude = np.max(np.abs(W_dot_1[t >= 4]))
        
        ax10.plot(t, (D_dot_1 + W_dot_1 + V_dot_1) / W_amplitude)
        
        W = trapezoid(W_dot_1[t >= 4], dx = dt)
        ax00.set_title(f'gsmo=True, W={np.round(W, 1)}')

        ax01.plot(t, D_dot_2, label = r'$\dot{D}$')
        ax01.plot(t, W_dot_2, label = '$\dot{W}$')
        ax01.plot(t, V_dot_2, label = '$\dot{V}$')
        ax11.plot(t, D_dot_2 + W_dot_2 + V_dot_2)
    
        W = trapezoid(W_dot_2[t >= 4], dx = dt)
        ax01.set_title(f'gsmo=False, W={np.round(W, 1)}')

        plt.suptitle(f'f={np.round(f,0)}')
    
        plt.savefig(save_dir / f'{str(k).zfill(2)}.png')
        k += 1
    
        plt.close()
    
    return

    
def plot_rikmenspoel_energy_balance(h5_f_lam):
    '''
    Plot energy balance
    '''
    
    h5 = h5py.File(sweep_dir / h5_f_lam, 'r')
    PG = ParameterGrid.init_pg_from_filepath(log_dir / h5.attrs['grid_filename'])

    f_arr = 1.0 / PG.v_from_key('T_c')
    lam_arr = PG.v_from_key('lam')

    lam_grid, f_grid = np.meshgrid(lam_arr, f_arr.magnitude)

    V = h5['energies']['V'][:]
    W = h5['energies']['W'][:]
    DI = h5['energies']['D_I'][:]
    DF = h5['energies']['D_F'][:]
    D = DI + DF

    gs = plt.GridSpec(2, 2)
    ax00 = plt.subplot(gs[0])
    ax01 = plt.subplot(gs[1])
    ax10 = plt.subplot(gs[2])
    ax11 = plt.subplot(gs[3])

    levels = 8

    # log potential energy
    CS = ax00.contourf(lam_grid, f_grid, np.log10(np.abs(V)), 
        levels = levels, cmap = cm_dict['V'])
    ax00.contour(lam_grid, f_grid, np.log10(np.abs(V)), 
        levels = levels, linestyles = '-', colors = ('k',))        
    plt.colorbar(CS, ax = ax00, orientation = 'horizontal')

    CS = ax01.contourf(lam_grid, f_grid, np.log10(np.abs(D)), 
        levels = levels, cmap = cm_dict['D'])
    ax01.contour(lam_grid, f_grid, np.log10(np.abs(D)), 
        levels = levels, linestyles = '-', colors = ('k',))        
    plt.colorbar(CS, ax = ax01, orientation = 'horizontal')

    # log dissipated energy
    CS = ax10.contourf(lam_grid, f_grid, np.log10(np.abs(W)), 
        levels = levels, cmap = cm_dict['W'])
    ax10.contour(lam_grid, f_grid, np.log10(np.abs(W)), 
        levels = levels, linestyles = '-', colors = ('k',))        
    plt.colorbar(CS, ax = ax10, orientation = 'horizontal')

    # log energy balance normalized by mechanical work
    CS = ax11.contourf(lam_grid, f_grid, np.log10(np.abs(W + D) / np.abs(W)), 
        levels = levels, cmap = cm_dict['U'])
    ax11.contour(lam_grid, f_grid, np.log10(np.abs(W + D) / np.abs(W)), 
        levels = levels, linestyles = '-', colors = ('k',))        
    plt.colorbar(CS, ax = ax11, orientation = 'horizontal')

    plt.show()

    return
                  
if __name__ == '__main__':
    
    # h5_a_b = ('analysis_'
    #     'a_min=-3.0_a_max=3.0_step_a=1.0_'
    #     'b_min=-3.0_b_max=0.0_step_b=1.0_'
    #     '_A=4.0_lam=1.0_N=250_dt=0.001.h5')    
    
    # h5_a_b = ('analysis_'
    #     'a_min=-2.0_a_max=3.0_step_a=0.2_'
    #     'b_min=-3.0_b_max=0.0_step_b=0.2_'
    #     '_A=4.0_lam=1.0_N=250_dt=0.001.h5')

    h5_a_b = ('analysis_'
        'a_min=-2.0_a_max=3.0_a_step=0.5_'
        'b_min=-4.0_b_max=0.0_b_step=0.5_'
        'A=4.0_lam=1.0_T=5.0_N=250_dt=0.0001.h5')

    h5_A_lam_a_b = ('analysis_'
        'A_min=0.5_A_max=2.0_A_step=0.5_'
        'lam_min=0.5_lam_max=2.0_lam_step=0.5_'
        'a_min=-2.0_a_max=3.0_step_a=1.0_'
        'b_min=-3.0_b_max=0.0_step_b=1.0_'
        'A=4.0_lam=1.0_T=5.0_N=250_dt=0.001.h5')

    # h5_c_lam_a_b = ('analysis_'
    #     'c_min=0.4_c_max=1.4_c_step=0.2_'
    #     'lam_min=0.5_lam_max=2.0_lam_step=0.5_'
    #     'a_min=-2.0_a_max=3.0_a_step=1.0_'
    #     'b_min=-3.0_b_max=0.0_b_step=1.0_'
    #     'T=5.0_N=250_dt=0.001.h5')

    h5_c_lam_a_b = ('analysis_'
        'c_min=0.4_c_max=1.4_c_step=0.2_'
        'lam_min=0.5_lam_max=2.0_lam_step=0.5_'
        'a_min=-0.0_a_max=4.0_a_step=0.2_'
        'b_min=-3.0_b_max=0.0_b_step=0.2_'
        'T=5.0_N=250_dt=0.001.h5')

    h5_C_a_b = ('analysis_'
        'C_min=2.0_C_max=10.0_C_step=1.0_'
        'a_min=-2.0_a_max=3.0_a_step=1.0_'
        'b_min=-3.0_b_max=0.0_b_step=1.0_'
        'A=4.0_lam=1.0_T=5.0_'
        'N=250_dt=0.001.h5')   

    # h5_mu_c_lam_fang_yen = ('analysis_fang_yeng_'
    #     'mu_min=-3.0_mu_max=1.0_mu_step=1.0_'
    #     'c_min=0.4_c_max=1.4_c_step=0.1_'
    #     'lam_min=0.5_lam_max=2.0_lam_step=0.1_'
    #     'T=5.0_N=250_dt=0.001.h5')

    h5_mu_c_lam_fang_yen = ('analysis_fang_yeng_'
        'mu_min=-3.0_mu_max=2.0_mu_step=0.5_'
        'c_min=0.4_c_max=1.4_c_step=0.1_'
        'lam_min=0.5_lam_max=2.0_lam_step=0.1_'
        'T=5.0_N=250_dt=0.001.h5')

    h5_eta_mu_c_lam = ('analysis_fang_yeng_'
        'eta_min=-3.0_eta_max=-1.0_eta_step=1.0_'
        'mu_min=-3.0_mu_max=2.0_mu_step=0.5_'
        'c_min=0.4_c_max=1.4_c_step=0.1_'
        'lam_min=0.5_lam_max=2.0_lam_step=0.1_'
        'T=5.0_N=250_dt=0.001.h5')

    h5_C_a_b = ('analysis_'
        'C_min=1.0_C_max=10.0_C_step=2.0_'
        'a_min=-3.0_a_max=2.0_a_step=0.5_'
        'b_min=-3.0_b_max=0.0_b_step=0.5_'
        'A=True_lam=1.8_T=5.0_N=250_dt=0.001.h5')

    h5_C_c_lam = ('analysis_'
        'C_min=2.0_C_max=10.0_C_step=2.0_'
        'lam_min=0.6_lam_max=2.0_lam_step=0.2_'
        'c_min=0.4_c_max=1.4_c_step=0.2_'
        'T=5.0_N=250_dt=0.001.h5')

    h5_C_c_lam_fang_yen = ('analysis_fang_yeng_'
        'C_min=2.0_C_max=10.0_C_step=2.0_'
        'mu_min=-3.0_mu_max=2.0_mu_step=0.5_'
        'c_min=0.4_c_max=1.6_c_step=0.1_'
        'lam_min=0.4_lam_max=2.0_lam_step=0.1_'
        'T=5.0_N=250_dt=0.001.h5'
    )

    h5_C_xi_mu = ('raw_data_fang_yeng_'
        'C_min=1.5_C_max=4.0_C_step=0.5_'
        'xi_min=-4.0_xi_max=-1.0_xi_step=0.5_'
        'mu_min=-4.0_mu_max=2.0_mu_step=0.5_'
        'T=5.0_N=250_dt=0.001.h5')

    # h5_C_eta_mu_fang_yen = ('analysis_fang_yeng'
    #     '_C_min=1.5_C_max=5.0_C_step=0.5_'
    #     'eta_min=-3.0_eta_max=-1.0_eta_step=0.5_'
    #     'mu_min=-3_mu_max=1_mu_step=0.5_'
    #     'T=5.0_N=250_dt=0.001.h5'
    # )    

    h5_C_eta_mu_fang_yen = ('analysis_fang_yeng_'
        'C_min=1.5_C_max=3.0_C_step=0.5_'
        'mu_min=-3.0_mu_max=2.0_mu_step=0.5_'
        'c_min=0.4_c_max=1.6_c_step=0.1_'
        'lam_min=0.4_lam_max=2.0_lam_step=0.1_'
        'T=5.0_N=250_dt=0.001.h5')


    h5_xi_mu_c_lam = ('analysis_fang_yeng_'
        'xi_min=-2.0_xi_max=-1.5_xi_step=0.1_'
        'mu_min=-4.0_mu_max=2.0_mu_step=1.0_'
        'c_min=0.4_c_max=1.6_c_step=0.1_'
        'lam_min=0.4_lam_max=2.0_lam_step=0.2_'
        'T=5_N=250_dt=0.001.h5')    

    h5_f_c_lam_rikmenspoel = ('analysis_rikmenspoel_'
        'f_min=10_f_max=50_f_step=5.0_'
        'c_min=0.4_c_max=1.6_c_step=0.2_'
        'lam_min=0.4_lam_max=2.0_'
        'f_step=0.2_phi=None_T=5_N=250_dt=0.001.h5')

    h5_f_lam_rikmenspoel = ('raw_data_rikmenspoel_'
        'f_min=10_f_max=50_f_step=5.0_'
        'lam_min=0.4_lam_max=2.0_f_step=0.1_'
        'phi=None_T=5_N=250_dt=0.001.h5')


    h5_f_lam_rikmenspoel_analysis = ('analysis_rikmenspoel_'
        'f_min=10_f_max=50_f_step=5.0_'
        'lam_min=0.5_lam_max=2.0_lam_step=0.05_'
        'phi=None_T=5_N=250_dt=0.001.h5'
        )

    h5_f_lam_rikmenspoel_analysis = ('analysis_rikmenspoel_'
        'f_min=5.0_f_max=55.0_f_step=5.0_'
        'lam_min=0.5_lam_max=2.0_lam_step=0.2_'
        'phi=None_T=5_N=250_dt=0.0001.h5')

    h5_f_lam_rikmenspoel_raw_data = ('raw_data_rikmenspoel_'
        'f_min=5.0_f_max=55.0_f_step=5.0_'
        'lam_min=0.5_lam_max=2.0_lam_step=0.2_'
        'phi=None_T=5_N=250_dt=0.0001.h5'
    )   

    # h5_f = ('analysis_rikmenspoel_'
    #     'f_min=10.0_f_max=100.0_f_step=2.0_'
    #     'const_A=False_phi=None_T=5_N=250_dt=0.001.h5'
    # )

    # h5_f = ('analysis_rikmenspoel_'
    #     'f_min=10.0_f_max=50.0_f_step=10.0_'
    #     'const_A=False_phi=None_T=5_N=250_dt=0.001.h5')


    h5_f_raw_data_gsmo_True = ('raw_data_rikmenspoel_'
        'f_min=10.0_f_max=50.0_f_step=5.0_'
        'const_A=False_phi=None_T=5_gmso=True_'
        'N=500_dt=0.001.h5'
    )

    h5_f_raw_data_gsmo_False = ('raw_data_rikmenspoel_'
        'f_min=10.0_f_max=50.0_f_step=5.0_'
        'const_A=False_phi=None_T=5_gmso=False_'
        'N=500_dt=0.001.h5'
    )

    h5_f_raw_data_gsmo_True = ('raw_data_rikmenspoel_'
        'f_min=10.0_f_max=50.0_f_step=5.0_'
        'const_A=False_phi=None_T=5_'
        'gmso=True_N=500_dt=0.0001.h5'
    )

    h5_f_lam_rikmenspoel_analysis = ('analysis_rikmenspoel_'
        'f_min=10.0_f_max=50.0_f_step=5.0_'
        'lam_min=0.5_lam_max=2.0_lam_step=0.1_'
        'phi=None_T=5_gmso=True_N=500_dt=0.001.h5'
    )

    h5_f_lam_rikmenspoel_analysis = ('analysis_rikmenspoel_'
        'f_min=10.0_f_max=50.0_f_step=5.0_'
        'lam_min=0.5_lam_max=2.0_lam_step=0.1_'
        'phi=None_T=5_gmso=True_N=500_dt=0.0001.h5'
    )

    h5_xi_f_lam_rikmenspoel_analysis = ('analysis_rikmenspoel_'
        'xi_min=-3_xi_max=-1.0_xi_step=0.5_'
        'f_min=10_f_max=50_f_step=5.0_'
        'lam_min=0.5_lam_max=2.0_lam_step=0.1_'
        'phi=None_T=5_gmso=True_N=500_dt=0.001.h5'
    )

    #plot_speed_curvature_norm_and_amplitude(h5_a_b)

    # plot_everything_over_a_b_for_different_C(h5_C_a_b)
    # plot_swimming_speed_and_curvature_norm(h5_c_lam_a_b)    
    # plot_everythin_over_a_b(h5_C_a_b)    
    # plot_swimming_speed_sperm(h5_a_b)
    # plot_transition_bands_for_different_c_lam(h5_c_lam_a_b)        
    # plot_transition_bands_different_lam(h5_c_lam_a_b)
                
    #plot_dissipation_ratio(h5_a_b)        
    #plot_work_dissipation_balance(h5_a_b)
    #plot_speed_curvature_norm_and_amplitude(h5_a_b)
    #plot_optimal_frequency(h5_a_b)
    #plot_dissipation_ratio_fang_yen(h5_mu_c_lam_fang_yen)    
        
    #plot_optimal_wave_length_mu(h5_mu_c_lam_fang_yen)
    
    #plot_optimal_wave_length_over_mu(h5_mu_c_lam_fang_yen)
    #plot_optimal_wave_length_over_K(h5_C_c_lam)
    #plot_optimal_wave_length_C(h5_C_c_lam)
    
    
    #plot_optima_on_contours_eta_mu(h5_eta_mu_c_lam)
    #plot_optimal_wave_length_for_different_eta_and_mu_0(h5_eta_mu_c_lam)    
    #plot_optimal_wave_length_for_different_eta_and_mu(h5_filename)    
    
    #plot_swimming_speed_C_a_b(h5_C_a_b)
    #plot_maximum_swimming_speed_C_mu_c_lam(h5_C_c_lam_fang_yen)
    #plot_optima_on_contours_C_mu(h5_C_c_lam_fang_yen)
    #plot_optimal_wave_length_for_different_C_and_mu_0(h5_C_c_lam_fang_yen)        
    # plot_fang_yen_sznitman_and_gagnon()    
    #plot_chemograms_fang_yen_1(h5_C_xi_mu)

    #plot_swimming_speed_C_eta_mu_fang_yen()    
    #plot_optimal_wave_length_for_different_C_eta_and_mu(h5_C_eta_mu_fang_yen)
    #plot_optima_on_contours_C_eta_mu(h5_C_eta_mu_fang_yen)
    
    #plot_optima_on_contours_eta_mu(h5_xi_mu_c_lam)
    
    #plot_rikmenspoel(h5_f_c_lam_rikmenspoel)
    #plot_rikmenspoel_2(h5_f_c_lam_rikmenspoel)
    #plot_rikmenspoel_chemograms(h5_f_lam_rikmenspoel)
    #plot_rikmenspoel_3(h5_f_lam_rikmenspoel_analysis)
    #plot_rikmenspoel_0(h5_f)
    #plot_rikmenspoel_energy_balance(h5_f_lam_rikmenspoel_analysis)
    
    plot_rikmenspoel_4(h5_xi_f_lam_rikmenspoel_analysis)
    
    # plot_rikmenspoel_power_balance_compare(h5_f_raw_data_gsmo_True, h5_f_raw_data_gsmo_False)
    


    
    print('Finished!')
        
        
        
    


