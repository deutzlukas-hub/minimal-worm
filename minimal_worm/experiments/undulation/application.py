'''
Created on 30 Dec 2023

@author: amoghasiddhi
'''
# Built-in
from pathlib import Path
from sys import argv
from argparse import ArgumentParser

#from decimal import Decimal

# Third-party
from parameter_scan import ParameterGrid
import pint
import numpy as np
from scipy.optimize import curve_fit

# Local imports
from minimal_worm import FRAME_KEYS
from minimal_worm.worm import CONTROL_KEYS
from minimal_worm.experiments import Sweeper
from minimal_worm.experiments.undulation import UndulationExperiment
from minimal_worm.experiments.undulation import create_storage_dir
from minimal_worm.experiments.undulation.analyse_sweeps import analyse
from minimal_worm.experiments.undulation.thesis import default_sweep_parameter

from minimal_worm import physical_to_dimless_parameters

from minimal_worm.model_parameters import ureg

#===============================================================================
# Experimental data
#===============================================================================

def fang_yen_data():
    '''
    Undulation wavelength, amplitude, frequencies for different viscosities
    from Fang-Yen 2010 paper    
    '''
    # Experimental Fang Yeng 2010
    mu_arr = 10**(np.array([0.000, 0.966, 2.085, 2.482, 2.902, 3.142, 3.955, 4.448])-3) # Pa*s            
    lam_arr = np.array([1.516, 1.388, 1.328, 1.239, 1.032, 0.943, 0.856, 0.799])        
    f_arr = [1.761, 1.597, 1.383, 1.119, 0.790, 0.650, 0.257, 0.169] # Hz
    A_arr = [2.872, 3.126, 3.290, 3.535, 4.772, 4.817, 6.226, 6.735]
    
    return mu_arr, lam_arr, f_arr, A_arr

def david_gagnon_data():
    '''
    Swimming speed, undulation frequencies, curvature amplitude for different
    viscosities from David Gagnon paper        
    '''

    mu_arr_U = np.array([1.02, 2.02, 3.68, 12.27, 151.89, 218.18, 309.68, 434.62, 645.85])
    U_arr = np.array([0.38, 0.3, 0.37, 0.31, 0.27, 0.25, 0.23, 0.22, 0.18])

    mu_arr_f = np.array([ 1., 1.98, 3.62, 12.08, 147.73, 216.56, 310.18, 434.09, 643.76])
    f_arr = np.array([1.84, 1.47, 2.08, 1.43, 1.4, 1.3, 1.39, 1.16, 1.05])

    mu_arr_c = np.array([1., 1.96, 3.58, 11.94, 146.03, 214.06, 306.6, 429.09, 643.76 ])
    c_arr = np.array([3.77, 3.45, 4.39, 3.05, 2.22, 2.06, 2.06, 1.86, 1.72])

    mu_arr_A = np.array([1.01, 2., 3.65, 12.26, 148.32, 217.17, 310.38,433.76, 642.42])    
    A_arr = np.array([0.26, 0.31, 0.33, 0.29, 0.24, 0.24, 0.22, 0.21,0.21])
    
    # average and convert to from mPa to Pa    
    mu_arr = 1e-3 * np.mean(np.vstack([mu_arr_U, mu_arr_f, mu_arr_c, mu_arr_A]), axis = 0)
    
    return mu_arr, U_arr, f_arr, c_arr, A_arr

def rikmenspoel_1978():
    '''
    
    '''

    f = np.array([8.26, 10.08, 26.13, 31.82, 36.23, 42.79, 45.69, 51.91]) * ureg.hertz
    lam = np.array([49.11, 46.11, 34.15, 32.75, 32.28, 30.51, 30.81, 27.93]) * ureg.micrometer
    A_real = np.array([9.12, 8.86, 5.23, 4.61, 4.51, 4.65, 4.77, 3.81]) * ureg.micrometer
    
    s = np.array([1.0, 0.9, 1.1, 1.2, 1.2, 5.4, 5.5, 4.5, 5.6, 8.6, 
        10.0, 10.4, 11.1, 12.4, 19.7, 20.0, 20.7, 21.4, 22.0, 29.4, 
        30.3, 30.6, 31.5, 30.0, 39.2, 40.7, 41.0]) * ureg.micrometer    
    
    A = np.array([5244.0, 4052.0, 3826.0, 3456.0, 3158.0, 2705.0, 2523.0, 
        2350.0, 2095.0, 1635.0, 1635.0, 1889.0, 1707.0, 1721.0, 1630.0,
        1405.0, 1659.0, 1549.0, 1709.0, 1494.0, 1930.0, 1581.0, 1450.0, 
        1240.0, 1381.0, 1445.0, 1227.0]) / ureg.centimeter

    L0 = sperm_param_dict['L0']    

    A_real_star = A_real / L0
    A_real_star.ito_base_units()
    A_star = A * L0
    A_star.ito_base_units()
    s_star = s / L0
    s_star.ito_base_units()
    
    lam_star = lam / L0    
    lam_star.ito_base_units()
        
    f_avg = 35
      
    data = {                
        'L0': 43*ureg.micrometer,
        'f': f,
        'lam': lam,
        'A_real': A_real,
        's': s,
        'A': A,                
        'f_avg': f_avg,
        'lam_star': lam_star,
        'A_real_star': A_real_star,
        'A_star': A_star,
        's_star': s_star
    }
    
    return data
    
def fit_rikmenspoel_1978():
    '''
    Fit rikmenspoel data
    '''
    
    data = rikmenspoel_1978()
    
    f, s = data['f'], data['s_star'] 
    lam, A_real, A = data['lam_star'], data['A_real_star'], data['A_star']
    
    # # Fit the polynomial
    c = np.polyfit(f.magnitude, lam.magnitude, 2)    
    # Create a polynomial function using the c
    lam_fit = np.poly1d(c)

    #Fit the polynomial
    c = np.polyfit(f.magnitude, A_real.magnitude, 2)    
    # Create a polynomial function using the c    
    A_real_fit = np.poly1d(c)
    
    # def sigmoid(s, a, b, c, d):
    #     y = a / (1 + np.exp(-c*(s-b))) + d
    #     return y
    #
    # A_avg = A[s > 0.25].mean()
    #
    # # Fit the sigmoid to wavelength
    # popt_lam, _ = curve_fit(sigmoid, 
    #     s.magnitude, A.magnitude, p0=[10, 5, -10, A_avg])
    # A_fit = lambda s: sigmoid(s, *popt_lam)

    #Fit the polynomial
    
    #assign higher weight to maximum value
    w = np.ones(len(A.magnitude))
    w[A.magnitude.argmax()] = 2
       
    c = np.polyfit(s.magnitude, A.magnitude, 6, w = w)    
    # Create a polynomial function using the c    
    A_fit = np.poly1d(c)
        
    return lam_fit, A_real_fit, A_fit

def backholm_data():
    
    D = np.array([57.3508, 63.713 , 64.351 , 62.2147, 62.554 , 66.864 , 65.7654, 70.017 , 62.3164])
    D_err = np.array([1.3161, 2.5584, 2.989 , 1.724 , 1.9295, 1.2164, 1.8063, 1.7684, 1.9611])
    B = np.array([1.4409e-13, 2.4748e-13, 9.8556e-14, 1.3987e-13, 3.0343e-13, 2.8179e-13, 171.384e-15, 293.704e-15, 32.6191e-15])
    B_err = np.array([3.2447e-14, 1.1539e-14, 3.6424e-15, 1.0303e-14, 1.1216e-13, 2.7779e-14, 113.835e-15, 72.7351e-15, 7.9552e-15])

    return

#===============================================================================
# Fits
#===============================================================================

def sznitzman_data():
    '''
    Sznitman 2010 (a) figure 4
    '''
    mu_arr_U = np.array([0.95, 1.21, 1.52, 1.91, 3.46, 6.04, 11.54])
    U_arr = np.array([0.37, 0.32, 0.35, 0.35, 0.33, 0.38,0.35])    
    
    mu_arr_f = np.array([0.99, 1.24, 1.57, 1.97, 3.59, 6.33, 12.01])                    
    f_arr = np.array([2.02, 1.97, 1.87, 1.77, 1.62, 1.69, 1.67])

    # average and convert to from mPa to Pa
    mu_arr = np.mean(np.vstack([mu_arr_U, mu_arr_f]), axis = 0) * 1e-3
    
    return mu_arr, U_arr, f_arr

def fang_yen_fit():
    '''
    Fit sigmoids to fang yen data
    '''
    mu_arr, lam_arr, f_arr, A_arr = fang_yen_data()

    log_mu_arr = np.log10(mu_arr)

    # Define the sigmoid function
    def sigmoid(x, a, b, c, d):
        y = a / (1 + np.exp(-c*(x-b))) + d
        return y

    # Fit the sigmoid to wavelength
    popt_lam, _ = curve_fit(sigmoid, log_mu_arr,lam_arr)
    lam_sig_fit = lambda log_mu: sigmoid(log_mu, *popt_lam)

    # Fit the sigmoid to frequency
    popt_f, _ = curve_fit(sigmoid, log_mu_arr, f_arr)
    f_sig_fit = lambda log_mu: sigmoid(log_mu, *popt_f)

    # Fit the sigmoid to amplitude
    a0 = 3.95
    b0 = 0.12    
    c0 = 2.13
    d0 = 2.94
    p0 = [a0, b0, c0, d0] 
    popt_A, _ = curve_fit(sigmoid, log_mu_arr, A_arr, p0=p0)
    A_sig_fit = lambda log_mu: sigmoid(log_mu, *popt_A)
    
    return lam_sig_fit, f_sig_fit, A_sig_fit

def fit_gagnon_sznitman():

    mu_arr_0, U_arr_0 = david_gagnon_data()[:2]    
    mu_arr_1, U_arr_1 =  sznitzman_data()[:2]

    U_arr = np.concatenate((U_arr_0, U_arr_1))
    mu_arr = np.concatenate((mu_arr_0, mu_arr_1))
    log_mu_arr = np.log10(mu_arr)

    # Fit 
    def sigmoid(x, a, b, c, d):
        y = a / (1 + np.exp(-c*(x-b))) + d
        return y

    popt_U, _ = curve_fit(sigmoid, log_mu_arr, U_arr)
    U_sig_fit = lambda log_mu: sigmoid(log_mu, *popt_U)
    
    return U_sig_fit

def sweep_mu_fang_yen(argv):
    '''
    Sweeps over
    
    - drag coefficient ratio C
    - internal time scale xi/E
    - fluid viscosity mu         
    
    Fit frequency f, lam and A over log of fluid viscosity mu 
    to Fang Yeng data                       
    '''
    
    # Parse sweep parameter
    sweep_parser = default_sweep_parameter()    

    sweep_parser.add_argument('--mu', 
        type=float, nargs=3, default = [-3, 1, 0.2])        
        
    sweep_param = sweep_parser.parse_known_args(argv)[0]    
    
    # The argumentparser for the sweep parameter has a boolean argument 
    # for ever frame key and control key which can be set to true
    # if it should be saved 
    FK = [k for k in FRAME_KEYS if getattr(sweep_param, k)]    
    CK = [k for k in CONTROL_KEYS if getattr(sweep_param, k)]

    print(f'FK={FK}')

    # Parse model parameter
    model_parser = UndulationExperiment.parameter_parser()
    model_param = model_parser.parse_known_args(argv)[0]

    # Customize parameter
    model_param.Ds_h = 0.01
    model_param.Ds_t = 0.01
    model_param.s0_h = 0.05
    model_param.s0_t = 0.95
    model_param.T = 5.0    
    model_param.use_c = False
    model_param.a_from_physical = True
    model_param.b_from_physical = True
        
    # Print all model parameter whose value has been
    # set via the command line
    cml_args = {k: v for k, v in vars(model_param).items() 
        if v != model_parser.get_default(k)}
    
    if len(cml_args) != 0: 
        print(cml_args)
    
    #===============================================================================
    # Init ParameterGrid 
    #===============================================================================
    
    mu_exp_min, mu_exp_max = sweep_param.mu[0], sweep_param.mu[1]
    mu_exp_step = sweep_param.mu[2]

    mu_exp_arr = np.arange(mu_exp_min, mu_exp_max + 0.1 * mu_exp_step, mu_exp_step)        
    mu_arr = 10**mu_exp_arr                
        
    lam_mu, f_mu, A_mu = fang_yen_fit()    
    T_c_arr = 1.0 / f_mu(mu_exp_arr)
    
    lam_arr = lam_mu(mu_exp_arr)
    A_arr = A_mu(mu_exp_arr)

    T_c_param = {'v_arr': T_c_arr.tolist(), 'round': 3, 'quantity': 'second'}    
    lam_param = {'v_arr': lam_arr.tolist(), 'round': 3}
    A_param = {'v_arr': A_arr.tolist(), 'round': 3}    
    mu_param = {'v_arr': mu_arr.tolist(), 'round': 6, 'quantity': 'pascal*second'}
    
    grid_param = {  
        ('T_c', 'mu', 'A', 'lam'): (T_c_param, mu_param, A_param, lam_param), 
    }
    
    sweep_parser = default_sweep_parameter()    
        
    PG = ParameterGrid(vars(model_param), grid_param)

    if sweep_param.save_to_storage:
        log_dir, sim_dir, sweep_dir = create_storage_dir()     
    else:
        from minimal_worm.experiments.undulation import sweep_dir, log_dir, sim_dir

    #=======================================================================
    # Run experiments 
    #=======================================================================
        
    # Experiments are run using the Sweeper class for parallelization 
    if sweep_param.run:
        Sweeper.run_sweep(
            sweep_param.worker, 
            PG, 
            UndulationExperiment.stw_control_sequence, 
            FK,
            log_dir, 
            sim_dir, 
            sweep_param.overwrite, 
            sweep_param.debug,
            'UExp')

    PG_filepath = PG.save(log_dir)
    print(f'Finished sweep! Save ParameterGrid to {PG_filepath}')
        
    # Pool and save simulation results to hdf5
    filename = Path(
        f'raw_data_'
        f'mu_min={mu_exp_min}_mu_max={mu_exp_max}_mu_step={mu_exp_step}'        
        f'N={model_param.N}_dt={model_param.dt}_'                
        f'T={model_param.T}.h5')
    
    h5_filepath = sweep_dir / filename

    if sweep_param.pool:        
        Sweeper.save_sweep_to_h5(PG, h5_filepath, sim_dir, FK, CK)

    #===============================================================================
    # Post analysis 
    #===============================================================================
    if sweep_param.analyse:
        sweep_param.A = True
        sweep_param.lam = True
        sweep_param.psi = True        
        analyse(h5_filepath, what_to_calculate=sweep_param)    
        
    return
    

def sweep_mu_fang_yen_test(argv):
    '''
    Sweeps over
    
    - drag coefficient ratio C
    - internal time scale xi/E
    - fluid viscosity mu         
    
    Fit frequency f, lam and A over log of fluid viscosity mu 
    to Fang Yeng data                       
    '''

    # Parse sweep parameter
    sweep_parser = default_sweep_parameter()    

    sweep_parser.add_argument('--mu', 
        type=float, nargs=3, default = [-3, 1, 0.2])        
        
    sweep_param = sweep_parser.parse_known_args(argv)[0]    
    
    # The argumentparser for the sweep parameter has a boolean argument 
    # for ever frame key and control key which can be set to true
    # if it should be saved 
    FK = [k for k in FRAME_KEYS if getattr(sweep_param, k)]    
    CK = [k for k in CONTROL_KEYS if getattr(sweep_param, k)]

    print(f'FK={FK}')

    # Parse model parameter
    model_parser = UndulationExperiment.parameter_parser()
    model_param = model_parser.parse_known_args(argv)[0]

    # Customize parameter
    model_param.Ds_h = 0.01
    model_param.Ds_t = 0.01
    model_param.s0_h = 0.05
    model_param.s0_t = 0.95
    model_param.T = 5.0    
    model_param.use_c = False
    model_param.a_from_physical = True
    model_param.b_from_physical = True
        
    # Print all model parameter whose value has been
    # set via the command line
    cml_args = {k: v for k, v in vars(model_param).items() 
        if v != model_parser.get_default(k)}
    
    if len(cml_args) != 0: 
        print(cml_args)
    
    #===============================================================================
    # Init ParameterGrid 
    #===============================================================================
    
    mu_exp_min, mu_exp_max = sweep_param.mu[0], sweep_param.mu[1]
    mu_exp_step = sweep_param.mu[2]

    mu_exp_arr = np.arange(mu_exp_min, mu_exp_max + 0.1 * mu_exp_step, mu_exp_step)        
    mu_arr = 10**mu_exp_arr                

    lam_fit, f_fit, A_fit = fang_yen_fit()    

    f_arr = f_fit(mu_exp_arr)     
    T_c_arr = 1.0 / f_arr    
    lam_arr = lam_fit(mu_exp_arr)
    A_arr = A_fit(mu_exp_arr)
        
    mu0, T0 = mu_arr[0], T_c_arr[0]
    f0 = 1.0 / T0
        
    model_param.T_c = T0 * ureg.second
    model_param.mu = mu0 * ureg.pascal * ureg.second
                
    physical_to_dimless_parameters(model_param)
    
    a0, b0 = model_param.a, model_param.b     
    
    a_arr = f_arr / f0 * mu_arr / mu0 * a0.magnitude  
    b_arr = f_arr / f0 * b0.magnitude    
    
    a_param = {'v_arr': a_arr.tolist(), 'round': 4}    
    b_param = {'v_arr': b_arr.tolist(), 'round': 5}    
    lam_param = {'v_arr': lam_arr.tolist(), 'round': 3}
    A_param = {'v_arr': A_arr.tolist(), 'round': 3}    
            
    # T_c_param = {'v_arr': T_c_arr.tolist(), 'round': 3, 'quantity': 'second'}    
    # mu_param = {'v_arr': mu_arr.tolist(), 'round': 6, 'quantity': 'pascal*second'}
    
    grid_param = {  
        ('a', 'b', 'lam', 'A'): (a_param, b_param, lam_param. A_param) 
    }
    
    sweep_parser = default_sweep_parameter()    
        
    PG = ParameterGrid(vars(model_param), grid_param)

    if sweep_param.save_to_storage:
        log_dir, sim_dir, sweep_dir = create_storage_dir()     
    else:
        from minimal_worm.experiments.undulation import sweep_dir, log_dir, sim_dir

    #=======================================================================
    # Run experiments 
    #=======================================================================
        
    # Experiments are run using the Sweeper class for parallelization 
    if sweep_param.run:
        Sweeper.run_sweep(
            sweep_param.worker, 
            PG, 
            UndulationExperiment.stw_control_sequence, 
            FK,
            log_dir, 
            sim_dir, 
            sweep_param.overwrite, 
            sweep_param.debug,
            'UExp')

    PG_filepath = PG.save(log_dir)
    print(f'Finished sweep! Save ParameterGrid to {PG_filepath}')
        
    # Pool and save simulation results to hdf5
    filename = Path(
        f'raw_data_'
        f'mu_min={mu_exp_min}_mu_max={mu_exp_max}_mu_step={mu_exp_step}'        
        f'N={model_param.N}_dt={model_param.dt}_'                
        f'T={model_param.T}.h5')
    
    h5_filepath = sweep_dir / filename

    if sweep_param.pool:        
        Sweeper.save_sweep_to_h5(PG, h5_filepath, sim_dir, FK, CK)

    #===============================================================================
    # Post analysis 
    #===============================================================================
    if sweep_param.analyse:
        sweep_param.A = True
        sweep_param.lam = True
        sweep_param.psi = True        
        analyse(h5_filepath, what_to_calculate=sweep_param)    
        
    return
    
    
    
    
if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-sweep',  
        choices = ['mu_fang_yen', 'mu_fang_yen_test'], help='Sweep to run')
            
    # Run function passed via command line
    args = parser.parse_known_args(argv)[0]    
    globals()['sweep_' + args.sweep](argv)
    