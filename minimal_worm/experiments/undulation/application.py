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

#===============================================================================
# Fits
#===============================================================================

def fang_yen_fit(return_param = False):
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
    
    if not return_param:
        return lam_sig_fit, f_sig_fit, A_sig_fit
    return lam_sig_fit, f_sig_fit, A_sig_fit, popt_lam, popt_f, popt_A 

#===============================================================================
# Celegans
#===============================================================================

def sweep_mu_fang_yen(argv):
    '''
    Sweeps over
    
    - fluid viscosity mu         
    
    Frequency f, lam0 and A0 as a function of mu are determined from fit to Fang Yeng data                       
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

    # Set baseline parameter to lowest viscosity and highest frequency
    mu0, T0 = mu_arr[0], T_c_arr[0]    
    model_param.T_c = T0 * ureg.second
    model_param.mu = mu0 * ureg.pascal * ureg.second
    physical_to_dimless_parameters(model_param)
    
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
    
    - fluid viscosity mu         
    
    Frequency f, lam0 and A0 as a function of mu are determined from fit to Fang Yeng data                       
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
        
    # Set baseline parameter to lowest viscosity and highest frequency        
    model_param.T_c = T0 * ureg.second
    model_param.mu = mu0 * ureg.pascal * ureg.second
    physical_to_dimless_parameters(model_param)    
    # Turn off, because we calculate a and b per hand
    model_param.a_from_physical = False
    model_param.b_from_physical = False
            
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
        ('a', 'b', 'lam', 'A'): (a_param, b_param, lam_param, A_param) 
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
        f'T={model_param.T}_test.h5')
    
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

def sweep_a_b_water_fang_yen(argv):
        
    # Parse sweep parameter
    sweep_parser = default_sweep_parameter()    

    sweep_parser.add_argument('--a', 
        type=float, nargs=3, default = [-2, 3, 0.1])    
    sweep_parser.add_argument('--b', 
        type=float, nargs=3, default = [-3, 0, 0.1])    

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

    # Waveform from experimental data
    lam_fit, _, A_fit = fang_yen_fit()    
    # Water
    log_mu0 = -3    
    lam0, A0 = lam_fit(log_mu0), A_fit(log_mu0) 
    
    model_param.lam = lam0
    model_param.A = A0
                
    # Print all model parameter whose value has been
    # set via the command line
    cml_args = {k: v for k, v in vars(model_param).items() 
        if v != model_parser.get_default(k)}
    
    if len(cml_args) != 0: 
        print(cml_args)
    
    # Create the ParameterGrid over which we want to run
    # the undulation experiments
    a_min, a_max = sweep_param.a[0], sweep_param.a[1]
    a_step = sweep_param.a[2]

    b_min, b_max = sweep_param.b[0], sweep_param.b[1]
    b_step = sweep_param.b[2]

    a_param = {'v_min': a_min, 'v_max': a_max + 0.1*a_step, 
        'N': None, 'step': a_step, 'round': 4, 'log': True}    

    b_param = {'v_min': b_min, 'v_max': b_max + 0.1*b_step, 
        'N': None, 'step': b_step, 'round': 5, 'log': True}
    
    grid_param = {'a': a_param, 'b': b_param}
    
    sweep_parser = default_sweep_parameter()    
        
    PG = ParameterGrid(vars(model_param), grid_param)

    if sweep_param.save_to_storage:
        log_dir, sim_dir, sweep_dir = create_storage_dir()     
    else:
        from minimal_worm.experiments.undulation import sweep_dir, log_dir, sim_dir
        
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
        f'a_min={a_min}_a_max={a_max}_a_step={a_step}_'
        f'b_min={b_min}_b_max={b_max}_b_step={b_step}_'                
        f'A={np.round(model_param.A,2)}_lam={np.round(model_param.lam, 2)}_'
        f'N={model_param.N}_dt={model_param.dt}_'        
        f'T={model_param.T}_pic_on={model_param.pic_on}.h5')
    
    h5_filepath = sweep_dir / filename

    if sweep_param.pool:        
        Sweeper.save_sweep_to_h5(PG, h5_filepath, sim_dir, FK, CK)

    if sweep_param.analyse:
        sweep_param.A = True
        sweep_param.f = True
        sweep_param.lag = True                
        sweep_param.lam = True                        
        analyse(h5_filepath, what_to_calculate=sweep_param)    
    return
    
def sweep_E_mu_fang_yen(argv):
    '''
    Sweeps over
    
    - fluid viscosity mu         
    
    Frequency f, lam0 and A0 as a function of mu are determined from fit to Fang Yeng data                       
    '''

    # Parse sweep parameter
    sweep_parser = default_sweep_parameter()    

    sweep_parser.add_argument('--E', 
        type=float, nargs=3, default = [4, 6, 11])        
    sweep_parser.add_argument('--mu', 
        type=float, nargs=3, default = [-3, 1, 0.2])        
    sweep_parser.add_argument('--xi', 
        type=float, nargs=3, default = 0.01)        

        
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
    
    log_E_min = sweep_param.E_over_E0[0]
    log_E_max = sweep_param.E_over_E0[1]
    NE = sweep_param.E_over_E0[2]
    xi = sweep_param.xi
        
    # eta_min = model_param.eta * E_over_E0_min
    # eta_max = model_param.eta * E_over_E0_max
    # Neta = sweep_param.E_over_E0[2]
    
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
        
    # Set baseline parameter to lowest viscosity and highest frequency        
    model_param.T_c = T0 * ureg.second
    model_param.mu = mu0 * ureg.pascal * ureg.second
    physical_to_dimless_parameters(model_param)    
    
    E_param = {'v_min': log_E_min, 'v_max': log_E_max, 
        'N': NE, 'step': None, 'log': True, 'round': 0, 'quantity': 'pascal'}    

    eta_param = {'v_min': log_E_min, 'v_max': log_E_max, 
        'N': NE, 'step': None, 'log': True, 'round': 2, 
        'scale': xi,
        'quantity': 'pascal*second'}

    T_c_param = {'v_arr': T_c_arr.tolist(), 'round': 3, 'quantity': 'second'}    
    lam_param = {'v_arr': lam_arr.tolist(), 'round': 3}
    A_param = {'v_arr': A_arr.tolist(), 'round': 3}    
    mu_param = {'v_arr': mu_arr.tolist(), 'round': 6, 'quantity': 'pascal*second'}
    
    grid_param = {  
        ('E', 'eta'): (E_param, eta_param),
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
        f'E_min_{log_E_min}_E_over_E0_max_{log_E_max}_NE={NE}_'
        f'mu_min={mu_exp_min}_mu_max={mu_exp_max}_mu_step={mu_exp_step}'        
        f'N={model_param.N}_dt={model_param.dt}_'                
        f'T={model_param.T}_test.h5')
    
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

def sweep_xi_mu_fang_yen(argv):
    '''
    Sweeps over
    
    - fluid viscosity mu         
    
    Frequency f, lam0 and A0 as a function of mu are determined from fit to Fang Yeng data                       
    '''

    # Parse sweep parameter
    sweep_parser = default_sweep_parameter()    

    sweep_parser.add_argument('--xi', 
        type=float, nargs=3, default = [-3, -1, 0.2])        
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
        
    xi_exp_min = sweep_param.xi[0]
    xi_exp_max = sweep_param.xi[1]        
    xi_step = sweep_param.xi[2]
    
    xi_exp_arr = np.arange(xi_exp_min, xi_exp_max + 0.1 * xi_step, xi_step)
    eta_arr = 10**xi_exp_arr * model_param.E.magnitude
    
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
        
    # Set baseline parameter to lowest viscosity and highest frequency        
    model_param.T_c = T0 * ureg.second
    model_param.mu = mu0 * ureg.pascal * ureg.second
    physical_to_dimless_parameters(model_param)    
    
    eta_param = {'v_arr': eta_arr.tolist(), 'round': 2, 'quantity': 'pascal*second'}    
    T_c_param = {'v_arr': T_c_arr.tolist(), 'round': 3, 'quantity': 'second'}    
    lam_param = {'v_arr': lam_arr.tolist(), 'round': 3}
    A_param = {'v_arr': A_arr.tolist(), 'round': 3}    
    mu_param = {'v_arr': mu_arr.tolist(), 'round': 6, 'quantity': 'pascal*second'}
    
    grid_param = {  
        'eta': eta_param,
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
        f'x_min_{xi_exp_min}_xi_max_{xi_exp_max}_x_step={xi_step}_'
        f'mu_min={mu_exp_min}_mu_max={mu_exp_max}_mu_step={mu_exp_step}'        
        f'N={model_param.N}_dt={model_param.dt}_'                
        f'T={model_param.T}_test.h5')
    
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

def sweep_E_xi_mu_fang_yen(argv):
    '''
    Sweeps over
    
    - fluid viscosity mu         
    
    Frequency f, lam0 and A0 as a function of mu are determined from fit to Fang Yeng data                       
    '''

    # Parse sweep parameter
    sweep_parser = default_sweep_parameter()    

    sweep_parser.add_argument('--Es', 
        type=float, nargs=3, default = [4, 6, 0.2])        
    sweep_parser.add_argument('--xi', 
        type=float, nargs=3, default = [-3, -1, 0.2])        
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

    log_E_min, log_E_max= sweep_param.Es[0], sweep_param.Es[1] 
    log_E_step = sweep_param.Es[2] 

    log_xi_min = sweep_param.xi[0]
    log_xi_max = sweep_param.xi[1]        
    log_xi_step = sweep_param.xi[2]
    
    log_xi_arr = np.arange(log_xi_min, log_xi_max + 0.1 * log_xi_step, log_xi_step)

    mu_exp_min, mu_exp_max = sweep_param.mu[0], sweep_param.mu[1]
    mu_exp_step = sweep_param.mu[2]
    mu_exp_arr = np.arange(mu_exp_min, mu_exp_max + 0.1 * mu_exp_step, mu_exp_step)        
    mu_arr = 10**mu_exp_arr                

    lam_fit, f_fit, A_fit = fang_yen_fit()    

    f_arr = f_fit(mu_exp_arr)     
    T_c_arr = 1.0 / f_arr     
    lam_arr = lam_fit(mu_exp_arr)
    A_arr = A_fit(mu_exp_arr)

    E_param = {'v_min': log_E_min, 'v_max': log_E_max + 0.1*log_E_step, 
        'N': None, 'step': log_E_step, 'log': True, 'round': 0, 'quantity': 'pascal'}    
    T_c_param = {'v_arr': T_c_arr.tolist(), 'round': 3, 'quantity': 'second'}    
    lam_param = {'v_arr': lam_arr.tolist(), 'round': 3}
    A_param = {'v_arr': A_arr.tolist(), 'round': 3}    
    mu_param = {'v_arr': mu_arr.tolist(), 'round': 6, 'quantity': 'pascal*second'}
    mu0, T0 = mu_arr[0], T_c_arr[0]

    # Set baseline parameter to lowest viscosity and highest frequency        
    model_param.T_c = T0 * ureg.second
    model_param.mu = mu0 * ureg.pascal * ureg.second
    physical_to_dimless_parameters(model_param)    
    
    for log_xi in log_xi_arr:
    
        eta_param = {'v_min': log_E_min, 'v_max': log_E_max + 0.1*log_E_step, 
            'N': None, 'step': log_E_step, 'log': True, 'round': 2, 
            'scale': 10**log_xi,
            'quantity': 'pascal*second'}
                
        grid_param = {  
            ('E', 'eta'): (E_param, eta_param),
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
            
        filename = Path(
            f'raw_data_'
            f'E_min_{log_E_min}_E_over_E0_max_{log_E_max}_E_step={log_E_step}_'
            f'mu_min={mu_exp_min}_mu_max={mu_exp_max}_mu_step={mu_exp_step}_'
            f'xi={log_xi}_'            
            f'N={model_param.N}_dt={model_param.dt}_'                
            f'T={model_param.T}_test.h5')
        
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


def sweep_mu_lam_c_fang_yen(argv):
    '''
    Sweeps over
    
    - fluid viscosity mu         
    
    Frequency f, lam0 and A0 as a function of mu are determined from fit to Fang Yeng data                       
    '''
    
    # Parse sweep parameter
    sweep_parser = default_sweep_parameter()    

    sweep_parser.add_argument('--mu', 
        type=float, nargs=3, default = [-3, 1, 0.2])        
    sweep_parser.add_argument('--lam', 
        type=float, nargs=3, default = [0.5, 2.0, 0.1])            
    sweep_parser.add_argument('--c', 
        type=float, nargs=3, default = [0.4, 2.0, 0.1])        
        
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
    model_param.use_c = True
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
        
    _, f_sig_fit, _ = fang_yen_fit()    
    T_c_arr = 1.0 / f_sig_fit(mu_exp_arr)

    # Shape-factor and curvature amplitude     
    c_min, c_max = sweep_param.c[0], sweep_param.c[1]
    c_step = sweep_param.c[2]

    lam_min, lam_max = sweep_param.lam[0], sweep_param.lam[1]
    lam_step = sweep_param.lam[2]

    # Use E fit from analysis
    model_param.E = 1.73 * model_param.E.magnitude * model_param.E.units
    model_param.eta = 1.73 * model_param.eta.magnitude * model_param.eta.units
    
    # Set baseline parameter to lowest viscosity and highest frequency
    mu0, T0 = mu_arr[0], T_c_arr[0]    
    model_param.T_c = T0 * ureg.second
    model_param.mu = mu0 * ureg.pascal * ureg.second
    
    c_param = {'v_min': c_min, 'v_max': c_max + 0.1*c_step, 
        'N': None, 'step': c_step, 'round': 2}    

    lam_param = {'v_min': lam_min, 'v_max': lam_max + 0.1*lam_step, 
        'N': None, 'step': lam_step, 'round': 2}
         
    T_c_param = {'v_arr': T_c_arr.tolist(), 'round': 3, 'quantity': 'second'}    
    mu_param = {'v_arr': mu_arr.tolist(), 'round': 6, 'quantity': 'pascal*second'}
    
    grid_param = {  
        ('T_c', 'mu'): (T_c_param, mu_param),
        'c': c_param, 
        'lam': lam_param
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
        f'lam_min={lam_min}_A_max={lam_max}_A_step={lam_step}_'        
        f'c_min={c_min}_A_max={c_max}_A_step={c_step}_'                
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
        choices = ['mu_fang_yen', 'mu_fang_yen_test', 'a_b_water_fang_yen', 
        'mu_lam_c_fang_yen', 'E_mu_fang_yen', 'xi_mu_fang_yen',
        'E_xi_mu_fang_yen'], help='Sweep to run')
                                    
    # Run function passed via command line
    args = parser.parse_known_args(argv)[0]    
    globals()['sweep_' + args.sweep](argv)
    