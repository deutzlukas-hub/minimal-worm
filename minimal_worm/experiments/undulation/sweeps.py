'''
Created on 15 Jun 2023

@author: amoghasiddhi
'''
# Built-in
from pathlib import Path
from sys import argv
from argparse import ArgumentParser, BooleanOptionalAction
#from decimal import Decimal

# Third-party
from parameter_scan import ParameterGrid
import numpy as np
from scipy.optimize import curve_fit
import pint

# Local imports
from minimal_worm.experiments import Sweeper, Saver
from minimal_worm.experiments.undulation import UndulationExperiment
from minimal_worm.experiments.undulation import create_storage_dir
from minimal_worm.experiments.undulation.analyse_sweeps import analyse_a_b


ureg = pint.UnitRegistry()

def fang_yen_data():
    '''
    Undulation wavelength, amplitude, frequencies for different viscosities
    from Fang-Yen 2010 paper    
    '''
    # Experimental Fang Yeng 2010
    mu_arr = 10**(np.array([0.000, 0.966, 2.085, 2.482, 2.902, 3.142, 3.955, 4.448])-3) # Pa*s            
    lam_arr = np.array([1.516, 1.388, 1.328, 1.239, 1.032, 0.943, 0.856, 0.799])        
    f_arr = [1.761, 1.597, 1.383, 1.119, 0.790, 0.650, 0.257, 0.169] # Hz
    
    return mu_arr, lam_arr, f_arr
    
def fang_yen_fit():
    '''
    Fit sigmoids to fang yen data
    '''
    mu_arr, lam_arr, f_arr = fang_yen_data()

    log_mu_arr = np.log10(mu_arr)

    # Define the sigmoid function
    def sigmoid(x, a, b, c, d):
        y = a / (1 + np.exp(-c*(x-b))) + d
        return y

    # Fit the sigmoid function to the data
    popt_lam, _ = curve_fit(sigmoid, log_mu_arr,lam_arr)
    lam_sig_fit = lambda log_mu: sigmoid(log_mu, *popt_lam)

    # Fit the sigmoid function to the data
    popt_f, _ = curve_fit(sigmoid, log_mu_arr, f_arr)
    f_sig_fit = lambda log_mu: sigmoid(log_mu, *popt_f)

    return lam_sig_fit, f_sig_fit

def default_sweep_parameter():
    '''
    Default sweep hyper parameter
    '''            
    parser = ArgumentParser(description = 'sweep-parameter')

    parser.add_argument('--worker', type = int, default = 10,
        help = 'Number of processes')         
    parser.add_argument('--run', action=BooleanOptionalAction, default = True,
        help = 'If true, sweep is run. Set to false if sweep has already been run and cached.')     
    parser.add_argument('--pool', action=BooleanOptionalAction, default = True,
        help = 'If true, FrameSequences are pickled to disk') 
    parser.add_argument('--analyse', action=BooleanOptionalAction, default = True,
        help = 'If true, analyse pooled raw data')     
    parser.add_argument('--overwrite', action=BooleanOptionalAction, default = False,
        help = 'If true, already existing simulation results are overwritten')
    parser.add_argument('--debug', action=BooleanOptionalAction, default = False,
        help = 'If true, exception handling is turned off which is helpful for debugging')    
    parser.add_argument('--save_to_storage', action=BooleanOptionalAction, default = False,
        help = 'If true, results are saved to external storage filesystem specified in dirs.py')         

    return parser

def sweep_a_b(argv):
    '''
    Parameter sweep over time scale ratios a and b

    Show that swimming speed and energy are fully captured 
    by the system input time scale ratios.          
    '''    

    # parse sweep parameter
    sweep_parser = default_sweep_parameter()    

    sweep_parser.add_argument('--a', 
        type=float, nargs=3, default = [-2, 3, 1.0])    
    sweep_parser.add_argument('--b', 
        type=float, nargs=3, default = [-3, 0, 1.0])    
    sweep_parser.add_argument('--FK', nargs = '+', 
        default = [
            't', 'r', 'theta', 'd1', 'd2', 'd3', 'k', 'sig', 
            'k_norm', 'sig_norm', 'r_t', 'w', 'k_t', 'sig_t', 
            'W_dot', 'D_F_dot', 'D_I_dot', 'V_dot']
        )
    sweep_parser.add_argument('--FK_pool', nargs = '+', 
        default = [
            'r', 'k', 'sig', 'k_norm', 'sig_norm', 'r_t',
            'W_dot', 'D_F_dot', 'D_I_dot', 'V_dot']
        )

    sweep_param = sweep_parser.parse_known_args(argv)[0]    

    # parse model parameter and convert to dict
    model_parser = UndulationExperiment.parameter_parser()
    model_param = model_parser.parse_known_args(argv)[0]

    # print all command-line-arguments assuming that they
    # are different from the default option 
    cml_args = {k: v for k, v in vars(model_param).items() 
        if v != model_parser.get_default(k)}
    
    if len(cml_args) != 0: 
        print(cml_args)

    a_min, a_max = sweep_param.a[0], sweep_param.a[1]
    a_step = sweep_param.a[2]

    b_min, b_max = sweep_param.b[0], sweep_param.b[1]
    b_step = sweep_param.b[2]

    a_param = {'v_min': a_min, 'v_max': a_max + 0.1*a_step, 
        'N': None, 'step': a_step, 'round': 4, 'log': True}    

    b_param = {'v_min': b_min, 'v_max': b_max + 0.1*b_step, 
        'N': None, 'step': b_step, 'round': 5, 'log': True}

    grid_param = {'a': a_param, 'b': b_param}
    
    PG = ParameterGrid(vars(model_param), grid_param)

    if sweep_param.save_to_storage:
        log_dir, sim_dir, sweep_dir = create_storage_dir()     
    else:
        from minimal_worm.experiments.undulation import sweep_dir, log_dir, sim_dir
           
    if sweep_param.run:
        # Run sweep
        Sweeper.run_sweep(
            sweep_param.worker, 
            PG, 
            UndulationExperiment.stw_control_sequence, 
            sweep_param.FK,
            log_dir, 
            sim_dir, 
            sweep_param.overwrite, 
            sweep_param.debug,
            'UExp')

    PG_filepath = PG.save(log_dir)
    print(f'Finished sweep! Save ParameterGrid to {PG_filepath}')

    # dt's number of decimal places 
    # dp = len(str(Decimal(str(model_param.dt))).split('.')[1])  
        
    # Run sweep
    filename = Path(
        f'raw_data_'
        f'a_min={a_min}_a_max={a_max}_a_step={a_step}_'
        f'b_min={b_min}_b_max={b_max}_b_step={b_step}_'
        f'A={model_param.A}_lam={model_param.lam}_'
        f'T={model_param.T}_'        
        f'N={model_param.N}_dt={model_param.dt}.h5')
    
    h5_filepath = sweep_dir / filename

    if sweep_param.pool:        
        Sweeper.save_sweep_to_h5(PG, h5_filepath, sim_dir, sweep_param.FK_pool)

    if sweep_param.analyse:
        analyse_a_b(h5_filepath)
                
    return


def sweep_A_lam_a_b(argv):
    '''
    Parameter sweep undulation parameter A, lam
    and the time scale ratios a and b

    Show if swimming speed and energy are fully captured 
    by the system input time scale ratios.          
    '''    

    # parse sweep parameter
    sweep_parser = default_sweep_parameter()    

    sweep_parser.add_argument('--A', 
        type=float, nargs=3, default = [2.0, 10.0, 2.0])    
    sweep_parser.add_argument('--lam', 
        type=float, nargs=3, default = [0.5, 2.0, 0.5])    
    sweep_parser.add_argument('--a', 
        type=float, nargs=3, default = [-2, 3, 1.0])    
    sweep_parser.add_argument('--b', 
        type=float, nargs=3, default = [-3, 0, 1.0])    
    sweep_parser.add_argument('--FK', nargs = '+', 
        default = [
            't', 'r', 'theta', 'd1', 'd2', 'd3', 'k', 'sig', 
            'k_norm', 'sig_norm', 'r_t', 'w', 'k_t', 'sig_t', 
            'W_dot', 'D_F_dot', 'D_I_dot', 'V_dot']
        )
    sweep_parser.add_argument('--FK_pool', nargs = '+', 
        default = [
            'r', 'k', 'sig', 'k_norm', 'sig_norm', 'r_t',
            'W_dot', 'D_F_dot', 'D_I_dot', 'V_dot']
        )


    sweep_param = sweep_parser.parse_known_args(argv)[0]    

    # parse model parameter and convert to dict
    model_parser = UndulationExperiment.parameter_parser()
    model_param = model_parser.parse_known_args(argv)[0]

    # print all command-line-arguments assuming that they
    # are different from the default option 
    cml_args = {k: v for k, v in vars(model_param).items() 
        if v != model_parser.get_default(k)}
    
    if len(cml_args) != 0: 
        print(cml_args)

    A_min, A_max = sweep_param.A[0], sweep_param.A[1]
    A_step = sweep_param.A[2]

    lam_min, lam_max = sweep_param.lam[0], sweep_param.lam[1]
    lam_step = sweep_param.lam[2]

    a_min, a_max = sweep_param.a[0], sweep_param.a[1]
    a_step = sweep_param.a[2]

    b_min, b_max = sweep_param.b[0], sweep_param.b[1]
    b_step = sweep_param.b[2]

    A_param = {'v_min': A_min, 'v_max': A_max + 0.1*a_step, 
        'N': None, 'step': A_step, 'round': 1}    

    lam_param = {'v_min': lam_min, 'v_max': lam_max + 0.1*lam_step, 
        'N': None, 'step': lam_step, 'round': 2}    

    a_param = {'v_min': a_min, 'v_max': a_max + 0.1*a_step, 
        'N': None, 'step': a_step, 'round': 4, 'log': True}    

    b_param = {'v_min': b_min, 'v_max': b_max + 0.1*b_step, 
        'N': None, 'step': b_step, 'round': 5, 'log': True}

    grid_param = {'A': A_param, 'lam': lam_param, 'a': a_param, 'b': b_param}
    
    PG = ParameterGrid(vars(model_param), grid_param)

    if sweep_param.save_to_storage:
        log_dir, sim_dir, sweep_dir = create_storage_dir()     
    else:
        from minimal_worm.experiments.undulation import sweep_dir, log_dir, sim_dir
           
    if sweep_param.run:
        # Run sweep
        Sweeper.run_sweep(
            sweep_param.worker, 
            PG, 
            UndulationExperiment.stw_control_sequence, 
            sweep_param.FK,
            log_dir, 
            sim_dir, 
            sweep_param.overwrite, 
            sweep_param.debug,
            'UExp')

    PG_filepath = PG.save(log_dir)
    print(f'Finished sweep! Save ParameterGrid to {PG_filepath}')

    # dt's number of decimal places 
    # dp = len(str(Decimal(str(model_param.dt))).split('.')[1])  
        
    # Run sweep
    filename = Path(
        f'raw_data_'
        f'A_min={lam_min}_A_max={lam_max}_A_step={lam_step}_'        
        f'lam_min={lam_min}_lam_max={lam_max}_lam_step={lam_step}_'
        f'a_min={a_min}_a_max={a_max}_a_step={a_step}_'
        f'b_min={b_min}_b_max={b_max}_b_step={b_step}_'
        f'T={model_param.T}_'        
        f'N={model_param.N}_dt={model_param.dt}.h5')
    
    h5_filepath = sweep_dir / filename

    if sweep_param.pool:        
        Sweeper.save_sweep_to_h5(PG, h5_filepath, sim_dir, sweep_param.FK_pool)

    if sweep_param.analyse:
        analyse_a_b(h5_filepath)
                
    return

def sweep_lam_a_b(argv):
    '''
    Sweeps over
        - wave length lambda
        - time scale ratio a
        - time scale rario b
    
    Why? 
    
    How does the position of the transition band characterized by the 
    contour U/U_max = 0.5 changes with the wavelength of the undulation?
    '''    

    # parse sweep parameter
    sweep_parser = default_sweep_parameter()    

    sweep_parser.add_argument('--lam', 
        type=float, nargs=3, default = [0.6, 2.0, 0.2])    
    sweep_parser.add_argument('--a', 
        type=float, nargs=3, default = [0.0, 4, 0.2])    
    sweep_parser.add_argument('--b', 
        type=float, nargs=3, default = [-3, 0, 0.2])    
    sweep_parser.add_argument('--FK', nargs = '+', 
        default = [
            't', 'r', 'theta', 'd1', 'd2', 'd3', 'k', 'sig', 
            'k_norm', 'sig_norm', 'r_t', 'w', 'k_t', 'sig_t', 
            'W_dot', 'D_F_dot', 'D_I_dot', 'V_dot']
        )
    sweep_parser.add_argument('--FK_pool', nargs = '+', 
        default = [
            'r', 'k', 'sig', 'k_norm', 'sig_norm', 'r_t',
            'W_dot', 'D_F_dot', 'D_I_dot', 'V_dot']
        )

    sweep_param = sweep_parser.parse_known_args(argv)[0]    

    # parse model parameter and convert to dict
    model_parser = UndulationExperiment.parameter_parser()
    model_param = model_parser.parse_known_args(argv)[0]
    model_param.use_c = True

    # print all command-line-arguments assuming that they
    # are different from the default option 
    cml_args = {k: v for k, v in vars(model_param).items() 
        if v != model_parser.get_default(k)}
    
    if len(cml_args) != 0: 
        print(cml_args)

    lam_min, lam_max = sweep_param.lam[0], sweep_param.lam[1]
    lam_step = sweep_param.lam[2]

    a_min, a_max = sweep_param.a[0], sweep_param.a[1]
    a_step = sweep_param.a[2]

    b_min, b_max = sweep_param.b[0], sweep_param.b[1]
    b_step = sweep_param.b[2]

    lam_param = {'v_min': lam_min, 'v_max': lam_max + 0.1*lam_step, 
        'N': None, 'step': lam_step, 'round': 2}    

    a_param = {'v_min': a_min, 'v_max': a_max + 0.1*a_step, 
        'N': None, 'step': a_step, 'round': 4, 'log': True}    

    b_param = {'v_min': b_min, 'v_max': b_max + 0.1*b_step, 
        'N': None, 'step': b_step, 'round': 5, 'log': True}

    grid_param = {'lam': lam_param, 'a': a_param, 'b': b_param}
    
    PG = ParameterGrid(vars(model_param), grid_param)

    if sweep_param.save_to_storage:
        log_dir, sim_dir, sweep_dir = create_storage_dir()     
    else:
        from minimal_worm.experiments.undulation import sweep_dir, log_dir, sim_dir
           
    if sweep_param.run:
        # Run sweep
        Sweeper.run_sweep(
            sweep_param.worker, 
            PG, 
            UndulationExperiment.stw_control_sequence, 
            sweep_param.FK,
            log_dir, 
            sim_dir, 
            sweep_param.overwrite, 
            sweep_param.debug,
            'UExp')

    PG_filepath = PG.save(log_dir)
    print(f'Finished sweep! Save ParameterGrid to {PG_filepath}')

    # dt's number of decimal places 
    # dp = len(str(Decimal(str(model_param.dt))).split('.')[1])  
        
    # Run sweep
    filename = Path(
        f'raw_data_'
        f'lam_min={lam_min}_lam_max={lam_max}_lam_step={lam_step}_'
        f'a_min={a_min}_a_max={a_max}_a_step={a_step}_'
        f'b_min={b_min}_b_max={b_max}_b_step={b_step}_'
        f'c={model_param.c}_T={model_param.T}_'        
        f'N={model_param.N}_dt={model_param.dt}.h5')
    
    h5_filepath = sweep_dir / filename

    if sweep_param.pool:        
        Sweeper.save_sweep_to_h5(PG, h5_filepath, sim_dir, sweep_param.FK_pool)

    if sweep_param.analyse:
        analyse_a_b(h5_filepath)
                
    return

def sweep_c_a_b(argv):
    '''
    Sweeps over
        - Amplitude wavenumber ratio c 
        - time scale ratio a    
        - time scale rario b
    
    Why? 
    
    How does the position of the transition band characterized by the 
    contour U/U_max = 0.5 changes with c?
    '''    

    # parse sweep parameter
    sweep_parser = default_sweep_parameter()    

    sweep_parser.add_argument('--c', 
        type=float, nargs=3, default = [0.4, 1.4, 0.2])    
    sweep_parser.add_argument('--a', 
        type=float, nargs=3, default = [0.0, 4, 0.2])    
    sweep_parser.add_argument('--b', 
        type=float, nargs=3, default = [-3, 0, 0.2])    
    sweep_parser.add_argument('--FK', nargs = '+', 
        default = [
            't', 'r', 'theta', 'd1', 'd2', 'd3', 'k', 'sig', 
            'k_norm', 'sig_norm', 'r_t', 'w', 'k_t', 'sig_t', 
            'W_dot', 'D_F_dot', 'D_I_dot', 'V_dot']
        )
    sweep_parser.add_argument('--FK_pool', nargs = '+', 
        default = [
            'r', 'k', 'sig', 'k_norm', 'sig_norm', 'r_t',
            'W_dot', 'D_F_dot', 'D_I_dot', 'V_dot']
        )

    sweep_param = sweep_parser.parse_known_args(argv)[0]    

    # parse model parameter and convert to dict
    model_parser = UndulationExperiment.parameter_parser()
    model_param = model_parser.parse_known_args(argv)[0]
    model_param.use_c = True

    # print all command-line-arguments assuming that they
    # are different from the default option 
    cml_args = {k: v for k, v in vars(model_param).items() 
        if v != model_parser.get_default(k)}
    
    if len(cml_args) != 0: 
        print(cml_args)

    c_min, c_max = sweep_param.c[0], sweep_param.c[1]
    c_step = sweep_param.c[2]


    c_param = {'v_min': c_min, 'v_max': c_max + 0.1*c_step, 
        'N': None, 'step': c_step, 'round': 1}    

    a_min, a_max = sweep_param.a[0], sweep_param.a[1]
    a_step = sweep_param.a[2]

    b_min, b_max = sweep_param.b[0], sweep_param.b[1]
    b_step = sweep_param.b[2]

    a_param = {'v_min': a_min, 'v_max': a_max + 0.1*a_step, 
        'N': None, 'step': a_step, 'round': 4, 'log': True}    

    b_param = {'v_min': b_min, 'v_max': b_max + 0.1*b_step, 
        'N': None, 'step': b_step, 'round': 5, 'log': True}

    grid_param = {'c': c_param, 'a': a_param, 'b': b_param}
    
    PG = ParameterGrid(vars(model_param), grid_param)

    if sweep_param.save_to_storage:
        log_dir, sim_dir, sweep_dir = create_storage_dir()     
    else:
        from minimal_worm.experiments.undulation import sweep_dir, log_dir, sim_dir
           
    if sweep_param.run:
        # Run sweep
        Sweeper.run_sweep(
            sweep_param.worker, 
            PG, 
            UndulationExperiment.stw_control_sequence, 
            sweep_param.FK,
            log_dir, 
            sim_dir, 
            sweep_param.overwrite, 
            sweep_param.debug,
            'UExp')

    PG_filepath = PG.save(log_dir)
    print(f'Finished sweep! Save ParameterGrid to {PG_filepath}')
        
    # Run sweep
    filename = Path(
        f'raw_data_'
        f'c_min={c_min}_c_max={c_max}_c_step={c_step}_'        
        f'a_min={a_min}_a_max={a_max}_a_step={a_step}_'
        f'b_min={b_min}_b_max={b_max}_b_step={b_step}_'
        f'lam={model_param.lam}_T={model_param.T}_'        
        f'N={model_param.N}_dt={model_param.dt}.h5')
    
    h5_filepath = sweep_dir / filename

    if sweep_param.pool:        
        Sweeper.save_sweep_to_h5(PG, h5_filepath, sim_dir, sweep_param.FK_pool)

    if sweep_param.analyse:
        analyse_a_b(h5_filepath)
                
    return


def sweep_c_lam_a_b(argv):
    '''
    Parameter sweep over time scale ratios a and b

    Show that swimming speed and energy are fully captured 
    by the system input time scale ratios.          
    '''    

    # parse sweep parameter
    sweep_parser = default_sweep_parameter()    

    sweep_parser.add_argument('--c', 
        type=float, nargs=3, default = [0.4, 1.4, 0.2])    
    sweep_parser.add_argument('--lam', 
        type=float, nargs=3, default = [0.5, 2.0, 0.5])    
    sweep_parser.add_argument('--a', 
        type=float, nargs=3, default = [-2, 3, 1.0])    
    sweep_parser.add_argument('--b', 
        type=float, nargs=3, default = [-3, 0, 1.0])    
    sweep_parser.add_argument('--FK', nargs = '+', 
        default = [
            't', 'r', 'theta', 'd1', 'd2', 'd3', 'k', 'sig', 
            'k_norm', 'sig_norm', 'r_t', 'w', 'k_t', 'sig_t', 
            'W_dot', 'D_F_dot', 'D_I_dot', 'V_dot']
        )
    sweep_parser.add_argument('--FK_pool', nargs = '+', 
        default = [
            'r', 'k', 'sig', 'k_norm', 'sig_norm', 'r_t',
            'W_dot', 'D_F_dot', 'D_I_dot', 'V_dot']
        )

    sweep_param = sweep_parser.parse_known_args(argv)[0]    

    # parse model parameter and convert to dict
    model_parser = UndulationExperiment.parameter_parser()
    model_param = model_parser.parse_known_args(argv)[0]
    model_param.use_c = True

    # print all command-line-arguments assuming that they
    # are different from the default option 
    cml_args = {k: v for k, v in vars(model_param).items() 
        if v != model_parser.get_default(k)}
    
    if len(cml_args) != 0: 
        print(cml_args)

    c_min, c_max = sweep_param.c[0], sweep_param.c[1]
    c_step = sweep_param.c[2]

    lam_min, lam_max = sweep_param.lam[0], sweep_param.lam[1]
    lam_step = sweep_param.lam[2]

    a_min, a_max = sweep_param.a[0], sweep_param.a[1]
    a_step = sweep_param.a[2]

    b_min, b_max = sweep_param.b[0], sweep_param.b[1]
    b_step = sweep_param.b[2]

    c_param = {'v_min': c_min, 'v_max': c_max + 0.1*c_step, 
        'N': None, 'step': c_step, 'round': 1}    

    lam_param = {'v_min': lam_min, 'v_max': lam_max + 0.1*lam_step, 
        'N': None, 'step': lam_step, 'round': 2}    

    a_param = {'v_min': a_min, 'v_max': a_max + 0.1*a_step, 
        'N': None, 'step': a_step, 'round': 4, 'log': True}    

    b_param = {'v_min': b_min, 'v_max': b_max + 0.1*b_step, 
        'N': None, 'step': b_step, 'round': 5, 'log': True}

    grid_param = {'c': c_param, 'lam': lam_param, 'a': a_param, 'b': b_param}
    
    PG = ParameterGrid(vars(model_param), grid_param)

    if sweep_param.save_to_storage:
        log_dir, sim_dir, sweep_dir = create_storage_dir()     
    else:
        from minimal_worm.experiments.undulation import sweep_dir, log_dir, sim_dir
           
    if sweep_param.run:
        # Run sweep
        Sweeper.run_sweep(
            sweep_param.worker, 
            PG, 
            UndulationExperiment.stw_control_sequence, 
            sweep_param.FK,
            log_dir, 
            sim_dir, 
            sweep_param.overwrite, 
            sweep_param.debug,
            'UExp')

    PG_filepath = PG.save(log_dir)
    print(f'Finished sweep! Save ParameterGrid to {PG_filepath}')

    # dt's number of decimal places 
    # dp = len(str(Decimal(str(model_param.dt))).split('.')[1])  
        
    # Run sweep
    filename = Path(
        f'raw_data_'
        f'c_min={c_min}_c_max={c_max}_c_step={c_step}_'        
        f'lam_min={lam_min}_lam_max={lam_max}_lam_step={lam_step}_'
        f'a_min={a_min}_a_max={a_max}_a_step={a_step}_'
        f'b_min={b_min}_b_max={b_max}_b_step={b_step}_'
        f'T={model_param.T}_'        
        f'N={model_param.N}_dt={model_param.dt}.h5')
    
    h5_filepath = sweep_dir / filename

    if sweep_param.pool:        
        Sweeper.save_sweep_to_h5(PG, h5_filepath, sim_dir, sweep_param.FK_pool)

    if sweep_param.analyse:
        analyse_a_b(h5_filepath)
                
    return

def sweep_C_a_b(argv):
    '''
    Sweeps over
        - drag coefficient ratio C
        - time scale ratio a
        - time scale rario b
    
    Why? 
    
    Find out if the region which marks the transition from internal 
    to external dissipation dominated regime changes as function of C
    '''    

    # parse sweep parameter
    sweep_parser = default_sweep_parameter()    

    sweep_parser.add_argument('--C', 
        type=float, nargs=3, default = [2.0, 10.0, 1.0])    
        
    sweep_parser.add_argument('--a', 
        type=float, nargs=3, default = [-2, 3, 1.0])    
    sweep_parser.add_argument('--b', 
        type=float, nargs=3, default = [-3, 0, 1.0])    
    sweep_parser.add_argument('--FK', nargs = '+', 
        default = [
            't', 'r', 'theta', 'd1', 'd2', 'd3', 'k', 'sig', 
            'k_norm', 'sig_norm', 'r_t', 'w', 'k_t', 'sig_t', 
            'W_dot', 'D_F_dot', 'D_I_dot', 'V_dot']
        )
    sweep_parser.add_argument('--FK_pool', nargs = '+', 
        default = [
            'r', 'k', 'sig', 'k_norm', 'sig_norm', 'r_t',
            'W_dot', 'D_F_dot', 'D_I_dot', 'V_dot']
        )

    sweep_param = sweep_parser.parse_known_args(argv)[0]    

    # parse model parameter and convert to dict
    model_parser = UndulationExperiment.parameter_parser()
    model_param = model_parser.parse_known_args(argv)[0]

    # print all command-line-arguments assuming that they
    # are different from the default option 
    cml_args = {k: v for k, v in vars(model_param).items() 
        if v != model_parser.get_default(k)}
    
    if len(cml_args) != 0: 
        print(cml_args)

    C_min, C_max = sweep_param.C[0], sweep_param.C[1]
    C_step = sweep_param.C[2]

    a_min, a_max = sweep_param.a[0], sweep_param.a[1]
    a_step = sweep_param.a[2]

    b_min, b_max = sweep_param.b[0], sweep_param.b[1]
    b_step = sweep_param.b[2]


    C_param = {'v_min': C_min, 'v_max': C_max + 0.1*C_step, 
        'N': None, 'step': C_step, 'round': 1}    

    a_param = {'v_min': a_min, 'v_max': a_max + 0.1*a_step, 
        'N': None, 'step': a_step, 'round': 4, 'log': True}    

    b_param = {'v_min': b_min, 'v_max': b_max + 0.1*b_step, 
        'N': None, 'step': b_step, 'round': 5, 'log': True}

    grid_param = {'C': C_param, 'a': a_param, 'b': b_param}
    
    PG = ParameterGrid(vars(model_param), grid_param)

    if sweep_param.save_to_storage:
        log_dir, sim_dir, sweep_dir = create_storage_dir()     
    else:
        from minimal_worm.experiments.undulation import sweep_dir, log_dir, sim_dir
           
    if sweep_param.run:
        # Run sweep
        Sweeper.run_sweep(
            sweep_param.worker, 
            PG, 
            UndulationExperiment.stw_control_sequence, 
            sweep_param.FK,
            log_dir, 
            sim_dir, 
            sweep_param.overwrite, 
            sweep_param.debug,
            'UExp')

    PG_filepath = PG.save(log_dir)
    print(f'Finished sweep! Save ParameterGrid to {PG_filepath}')

    # dt's number of decimal places 
    # dp = len(str(Decimal(str(model_param.dt))).split('.')[1])  
        
    # Run sweep
    filename = Path(
        f'raw_data_'
        f'C_min={C_min}_C_max={C_max}_C_step={C_step}_'
        f'a_min={a_min}_a_max={a_max}_a_step={a_step}_'
        f'b_min={b_min}_b_max={b_max}_b_step={b_step}_'
        f'A={model_param.A}_lam={model_param.lam}_T={model_param.T}_'        
        f'N={model_param.N}_dt={model_param.dt}.h5')
    
    h5_filepath = sweep_dir / filename

    if sweep_param.pool:        
        Sweeper.save_sweep_to_h5(PG, h5_filepath, sim_dir, sweep_param.FK_pool)

    if sweep_param.analyse:
        analyse_a_b(h5_filepath)
                
    return

def sweep_C_c_lam(argv):
    '''
    Sweeps over
        - drag coefficient ratio C
        - time scale ratio a
        - time scale rario b
    
    Why? 
    
    Find out if the region which marks the transition from internal 
    to external dissipation dominated regime changes as function of C
    
    Find out if the fluid dissipation energy surface as a function of
    c and lambda changes.
    '''    

    # parse sweep parameter
    sweep_parser = default_sweep_parameter()    

    sweep_parser.add_argument('--C', 
        type=float, nargs=3, default = [2.0, 10.0, 2.0])        
    sweep_parser.add_argument('--c', 
        type=float, nargs=3, default = [0.4, 1.4, 0.1])    
    sweep_parser.add_argument('--lam', 
        type=float, nargs=3, default = [0.5, 2.0, 0.1])    
        
    sweep_parser.add_argument('--FK', nargs = '+', 
        default = [
            't', 'r', 'theta', 'd1', 'd2', 'd3', 'k', 'sig', 
            'k_norm', 'sig_norm', 'r_t', 'w', 'k_t', 'sig_t', 
            'W_dot', 'D_F_dot', 'D_I_dot', 'V_dot']
        )
    sweep_parser.add_argument('--FK_pool', nargs = '+', 
        default = [
            'r', 'k', 'sig', 'k_norm', 'sig_norm', 'r_t',
            'W_dot', 'D_F_dot', 'D_I_dot', 'V_dot']
        )

    sweep_param = sweep_parser.parse_known_args(argv)[0]    

    # parse model parameter and convert to dict
    model_parser = UndulationExperiment.parameter_parser()
    model_param = model_parser.parse_known_args(argv)[0]

    # print all command-line-arguments assuming that they
    # are different from the default option 
    cml_args = {k: v for k, v in vars(model_param).items() 
        if v != model_parser.get_default(k)}
    
    if len(cml_args) != 0: 
        print(cml_args)

    mu = model_param.mu                 
    f_mu = fang_yen_fit()[1]    
    T_c = 1.0 / f_mu(np.log10(mu.magnitude)) * ureg.second
    model_param.T_c = T_c

    C_min, C_max = sweep_param.C[0], sweep_param.C[1]
    C_step = sweep_param.C[2]

    lam_min, lam_max = sweep_param.lam[0], sweep_param.lam[1]
    lam_step = sweep_param.lam[2]

    c_min, c_max = sweep_param.c[0], sweep_param.c[1]
    c_step = sweep_param.c[2]

    C_param = {'v_min': C_min, 'v_max': C_max + 0.1*C_step, 
        'N': None, 'step': C_step, 'round': 1}    

    lam_param = {'v_min': lam_min, 'v_max': lam_max + 0.1*lam_step, 
        'N': None, 'step': lam_step, 'round': 2}    

    c_param = {'v_min': c_min, 'v_max': c_max + 0.1*c_step, 
        'N': None, 'step': c_step, 'round': 2}

    grid_param = {'C': C_param, 'lam': lam_param, 'c': c_param}
    
    PG = ParameterGrid(vars(model_param), grid_param)

    if sweep_param.save_to_storage:
        log_dir, sim_dir, sweep_dir = create_storage_dir()     
    else:
        from minimal_worm.experiments.undulation import sweep_dir, log_dir, sim_dir
           
    if sweep_param.run:
        # Run sweep
        Sweeper.run_sweep(
            sweep_param.worker, 
            PG, 
            UndulationExperiment.stw_control_sequence, 
            sweep_param.FK,
            log_dir, 
            sim_dir, 
            sweep_param.overwrite, 
            sweep_param.debug,
            'UExp')

    PG_filepath = PG.save(log_dir)
    print(f'Finished sweep! Save ParameterGrid to {PG_filepath}')

    # dt's number of decimal places 
    # dp = len(str(Decimal(str(model_param.dt))).split('.')[1])  
        
    # Run sweep
    filename = Path(
        f'raw_data_'
        f'C_min={C_min}_C_max={C_max}_C_step={C_step}_'
        f'lam_min={lam_min}_lam_max={lam_max}_lam_step={lam_step}_'
        f'c_min={c_min}_c_max={c_max}_c_step={c_step}_'
        f'T={model_param.T}_N={model_param.N}_dt={model_param.dt}.h5')
    
    h5_filepath = sweep_dir / filename

    if sweep_param.pool:        
        Sweeper.save_sweep_to_h5(PG, h5_filepath, sim_dir, sweep_param.FK_pool)

    if sweep_param.analyse:
        analyse_a_b(h5_filepath)
                
    return


def sweep_mu_c_lam_fang_yen(argv):
    '''
    Sweeps over
        - mu: Fluid viscosity
        - c = A/q where A is the undulation amplitude and q the wavenumber
        - lam undulation wavelength
    
    Fit frequency f over log of fluid viscosity mu to Fang Yeng data               
        
    Sweep over c lam grid for every (mu, f) pair.        
    '''    

    # parse sweep parameter
    sweep_parser = default_sweep_parameter()    

    # 
    sweep_parser.add_argument('--mu', 
        type=float, nargs=3, default = [-3, 1, 1.0])        
    sweep_parser.add_argument('--c', 
        type=float, nargs=3, default = [0.4, 1.4, 0.2])    
    sweep_parser.add_argument('--lam', 
        type=float, nargs=3, default = [0.5, 2.0, 0.5])    
    sweep_parser.add_argument('--xi', 
        type=float, )
                
    sweep_parser.add_argument('--FK', nargs = '+', 
        default = [
            't', 'r', 'theta', 'd1', 'd2', 'd3', 'k', 'sig', 
            'k_norm', 'sig_norm', 'r_t', 'w', 'k_t', 'sig_t', 
            'W_dot', 'D_F_dot', 'D_I_dot', 'V_dot']
        )
    sweep_parser.add_argument('--FK_pool', nargs = '+', 
        default = [
            'r', 'k', 'sig', 'k_norm', 'sig_norm', 'r_t',
            'W_dot', 'D_F_dot', 'D_I_dot', 'V_dot']
        )
                
    sweep_param = sweep_parser.parse_known_args(argv)[0]    

    # parse model parameter and convert to dict
    model_parser = UndulationExperiment.parameter_parser()
    model_param = model_parser.parse_known_args(argv)[0]
    
    model_param.use_c = True

    model_param.a_from_physical = True
    model_param.b_from_physical = True

    # print all command-line-arguments assuming that they
    # are different from the default option 
    cml_args = {k: v for k, v in vars(model_param).items() 
        if v != model_parser.get_default(k)}
    
    if len(cml_args) != 0: 
        print(cml_args)

    mu_exp_min, mu_exp_max = sweep_param.mu[0], sweep_param.mu[1]
    mu_exp_step = sweep_param.mu[2]
             
    mu_exp_arr = np.arange(mu_exp_min, mu_exp_max + 0.1 * mu_exp_step, mu_exp_step)        
    mu_arr = 10**mu_exp_arr                
    
    f_mu = fang_yen_fit()[1]    
    T_c_arr = 1.0 / f_mu(mu_exp_arr)

    T_c_param = {'v_arr': T_c_arr.tolist(), 'round': 2, 'quantity': 'second'}
    mu_param = {'v_arr': mu_arr.tolist(), 'round': 5, 'quantity': 'pascal*second'}
    
    c_min, c_max = sweep_param.c[0], sweep_param.c[1]
    c_step = sweep_param.c[2]

    lam_min, lam_max = sweep_param.lam[0], sweep_param.lam[1]
    lam_step = sweep_param.lam[2]

    c_param = {'v_min': c_min, 'v_max': c_max + 0.1*c_step, 
        'N': None, 'step': c_step, 'round': 2}    

    lam_param = {'v_min': lam_min, 'v_max': lam_max + 0.1*lam_step, 
        'N': None, 'step': lam_step, 'round': 2}

    grid_param = {('T_c', 'mu'): (T_c_param, mu_param), 
        'c': c_param, 'lam': lam_param}

    PG = ParameterGrid(vars(model_param), grid_param)

    if sweep_param.save_to_storage:
        log_dir, sim_dir, sweep_dir = create_storage_dir()     
    else:
        from minimal_worm.experiments.undulation import sweep_dir, log_dir, sim_dir
           
    if sweep_param.run:
        # Run sweep
        Sweeper.run_sweep(
            sweep_param.worker, 
            PG, 
            UndulationExperiment.stw_control_sequence, 
            sweep_param.FK,
            log_dir, 
            sim_dir, 
            sweep_param.overwrite, 
            sweep_param.debug,
            'UExp')

    PG_filepath = PG.save(log_dir)
    print(f'Finished sweep! Save ParameterGrid to {PG_filepath}')

    # dt's number of decimal places 
    # dp = len(str(Decimal(str(model_param.dt))).split('.')[1])  
        
    # Run sweep
    filename = Path(
        f'raw_data_fang_yeng_'
        f'mu_min={mu_exp_min}_mu_max={mu_exp_max}_mu_step={mu_exp_step}_'        
        f'c_min={c_min}_c_max={c_max}_c_step={c_step}_'
        f'lam_min={lam_min}_lam_max={lam_max}_lam_step={lam_step}_'
        f'T={model_param.T}_'
        f'N={model_param.N}_dt={model_param.dt}.h5')
    
    h5_filepath = sweep_dir / filename

    if sweep_param.pool:        
        Sweeper.save_sweep_to_h5(PG, h5_filepath, sim_dir, sweep_param.FK_pool)

    if sweep_param.analyse:
        analyse_a_b(h5_filepath)
                
    return

def sweep_eta_mu_c_lam_fang_yen(argv):
    '''
    Sweeps over
        - mu: Fluid viscosity
        - c = A/q where A is the undulation amplitude and q the wavenumber
        - lam undulation wavelength
    
    Fit frequency f over log of fluid viscosity mu to Fang Yeng data               
        
    Sweep over c lam grid for every (mu, f) pair.        
    '''    

    # parse sweep parameter
    sweep_parser = default_sweep_parameter()    

    #
    sweep_parser.add_argument('--eta', 
        type=float, nargs=3, default = [-3, -1, 1.0])             
    sweep_parser.add_argument('--mu', 
        type=float, nargs=3, default = [-3, 1, 1.0])        
    sweep_parser.add_argument('--c', 
        type=float, nargs=3, default = [0.4, 1.4, 0.2])    
    sweep_parser.add_argument('--lam', 
        type=float, nargs=3, default = [0.5, 2.0, 0.5])    
    sweep_parser.add_argument('--xi', 
        type=float, )
                
    sweep_parser.add_argument('--FK', nargs = '+', 
        default = [
            't', 'r', 'theta', 'd1', 'd2', 'd3', 'k', 'sig', 
            'k_norm', 'sig_norm', 'r_t', 'w', 'k_t', 'sig_t', 
            'W_dot', 'D_F_dot', 'D_I_dot', 'V_dot']
        )
    sweep_parser.add_argument('--FK_pool', nargs = '+', 
        default = [
            'r', 'k', 'sig', 'k_norm', 'sig_norm', 'r_t',
            'W_dot', 'D_F_dot', 'D_I_dot', 'V_dot']
        )
                
    sweep_param = sweep_parser.parse_known_args(argv)[0]    

    # parse model parameter and convert to dict
    model_parser = UndulationExperiment.parameter_parser()
    model_param = model_parser.parse_known_args(argv)[0]
    
    model_param.use_c = True

    model_param.a_from_physical = True
    model_param.b_from_physical = True

    # print all command-line-arguments assuming that they
    # are different from the default option 
    cml_args = {k: v for k, v in vars(model_param).items() 
        if v != model_parser.get_default(k)}
    
    if len(cml_args) != 0: 
        print(cml_args)

    eta_min, eta_max = sweep_param.eta[0], sweep_param.eta[1]
    eta_step = sweep_param.eta[2]

    eta_param = {'v_min': eta_min, 'v_max': eta_max + 0.1*eta_step, 
        'N': None, 'step': eta_step, 'round': 0, 'log': True, 
        'scale': model_param.E.magnitude, 'quantity': 'pascal*second'}    
    
    mu_exp_min, mu_exp_max = sweep_param.mu[0], sweep_param.mu[1]
    mu_exp_step = sweep_param.mu[2]
             
    mu_exp_arr = np.arange(mu_exp_min, mu_exp_max + 0.1 * mu_exp_step, mu_exp_step)        
    mu_arr = 10**mu_exp_arr                
    
    f_mu = fang_yen_fit()[1]    
    T_c_arr = 1.0 / f_mu(mu_exp_arr)

    T_c_param = {'v_arr': T_c_arr.tolist(), 'round': 2, 'quantity': 'second'}
    mu_param = {'v_arr': mu_arr.tolist(), 'round': 5, 'quantity': 'pascal*second'}
    
    c_min, c_max = sweep_param.c[0], sweep_param.c[1]
    c_step = sweep_param.c[2]

    lam_min, lam_max = sweep_param.lam[0], sweep_param.lam[1]
    lam_step = sweep_param.lam[2]

    c_param = {'v_min': c_min, 'v_max': c_max + 0.1*c_step, 
        'N': None, 'step': c_step, 'round': 2}    

    lam_param = {'v_min': lam_min, 'v_max': lam_max + 0.1*lam_step, 
        'N': None, 'step': lam_step, 'round': 2}

    grid_param = {
        'eta': eta_param,
        ('T_c', 'mu'): (T_c_param, mu_param), 
        'c': c_param, 'lam': lam_param
    }

    PG = ParameterGrid(vars(model_param), grid_param)

    if sweep_param.save_to_storage:
        log_dir, sim_dir, sweep_dir = create_storage_dir()     
    else:
        from minimal_worm.experiments.undulation import sweep_dir, log_dir, sim_dir
           
    if sweep_param.run:
        # Run sweep
        Sweeper.run_sweep(
            sweep_param.worker, 
            PG, 
            UndulationExperiment.stw_control_sequence, 
            sweep_param.FK,
            log_dir, 
            sim_dir, 
            sweep_param.overwrite, 
            sweep_param.debug,
            'UExp')

    PG_filepath = PG.save(log_dir)
    print(f'Finished sweep! Save ParameterGrid to {PG_filepath}')

    # dt's number of decimal places 
    # dp = len(str(Decimal(str(model_param.dt))).split('.')[1])  
        
    # Run sweep
    filename = Path(
        f'raw_data_fang_yeng_'
        f'mu_min={mu_exp_min}_mu_max={mu_exp_max}_mu_step={mu_exp_step}_'        
        f'c_min={c_min}_c_max={c_max}_c_step={c_step}_'
        f'lam_min={lam_min}_lam_max={lam_max}_lam_step={lam_step}_'
        f'T={model_param.T}_'
        f'N={model_param.N}_dt={model_param.dt}.h5')
    
    h5_filepath = sweep_dir / filename

    if sweep_param.pool:        
        Sweeper.save_sweep_to_h5(PG, h5_filepath, sim_dir, sweep_param.FK_pool)

    if sweep_param.analyse:
        analyse_a_b(h5_filepath)
                
    return


if __name__ == '__main__':
        
    parser = ArgumentParser()
    parser.add_argument('-sweep',  
        choices = ['a_b', 'A_lam_a_b', 'c_lam_a_b', 'mu_c_lam_fang_yen', 
            'eta_mu_c_lam_fang_yen', 'C_c_lam', 'c_lam_a_b', 'C_a_b', 'c_lam', 
            'lam_a_b', 'c_a_b'], help='Sweep to run')
            
    # Run function passed via command line
    args = parser.parse_known_args(argv)[0]    
    globals()['sweep_' + args.sweep](argv)





