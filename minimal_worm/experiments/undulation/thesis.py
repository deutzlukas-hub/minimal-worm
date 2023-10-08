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
import pint

# Local imports
from minimal_worm import FRAME_KEYS
from minimal_worm.worm import CONTROL_KEYS
from minimal_worm.experiments import Sweeper
from minimal_worm.experiments.undulation import UndulationExperiment
from minimal_worm.experiments.undulation import create_storage_dir
from minimal_worm.experiments.undulation.analyse_sweeps import analyse

ureg = pint.UnitRegistry()

def default_sweep_parameter():
    '''
    Default sweep hyper parameter
    '''            
    parser = ArgumentParser(description = 'sweep-parameter', allow_abbrev=False)

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
    
    # Frame keys
    parser.add_argument('--t', action=BooleanOptionalAction, default = True,
        help = 'If true, save time stamps')
    parser.add_argument('--r', action=BooleanOptionalAction, default = True,
        help = 'If true, save centreline coordinates')
    parser.add_argument('--theta', action=BooleanOptionalAction, default = True,
        help = 'If true, save Euler angles')    
    parser.add_argument('--d1', action=BooleanOptionalAction, default = False,
        help = 'If true, save director 1')
    parser.add_argument('--d2', action=BooleanOptionalAction, default = False,
        help = 'If true, save director 2')
    parser.add_argument('--d3', action=BooleanOptionalAction, default = False,
        help = 'If true, save director 3')
    parser.add_argument('--k', action=BooleanOptionalAction, default = True,
        help = 'If true, save curvature')
    parser.add_argument('--sig', action=BooleanOptionalAction, default = True,
        help = 'If true, saved shear-stretch')
    parser.add_argument('--k_norm', action=BooleanOptionalAction, default = True,
        help = 'If true, save L2 norm of the real and preferred curvature difference')
    parser.add_argument('--sig_norm', action=BooleanOptionalAction, default = True,
        help = 'If true, save L2 norm of the real and preferred shear-stretch difference')

    # Velocity keys
    parser.add_argument('--r_t', action=BooleanOptionalAction, default = False,
        help = 'If trues, save centreline velocity')
    parser.add_argument('--w', action=BooleanOptionalAction, default = False,
        help = 'If true, save angular velocity')
    parser.add_argument('--k_t', action=BooleanOptionalAction, default = False,
        help = 'If true, save curvature strain rate')
    parser.add_argument('--sig_t', action=BooleanOptionalAction, default = False,
        help = 'If true, save shear-stretch strain rate')
    
    # Forces and torque keys
    parser.add_argument('--f_F', action=BooleanOptionalAction, default = False,
        help = 'If true, save fluid drag force density')
    parser.add_argument('--l_F', action=BooleanOptionalAction, default = False,
        help = 'If true, save fluid drag torque density')
    parser.add_argument('--f_M', action=BooleanOptionalAction, default = False,
        help = 'If true, save muscle force density')
    parser.add_argument('--l_M', action=BooleanOptionalAction, default = False,
        help = 'If true, save muscle torque density')
    parser.add_argument('--N_force', action=BooleanOptionalAction, default = False,
        dest = 'N', help = 'If true, save internal force resultant')
    parser.add_argument('--M_torque', action=BooleanOptionalAction, default = False,
        dest = 'M', help = 'If true, save internal torque resultant')

    # Power keys
    parser.add_argument('--D_F_dot', action=BooleanOptionalAction, default = True,
        help = 'If true, save fluid dissipation rate')
    parser.add_argument('--D_I_dot', action=BooleanOptionalAction, default = True,
        help = 'If true, save internal dissipation rate')
    parser.add_argument('--W_dot', action=BooleanOptionalAction, default = True,
        help = 'If true, save mechanical muscle power')
    parser.add_argument('--V_dot', action=BooleanOptionalAction, default = True,
        help = 'If true, save potential rate')
    parser.add_argument('--V', action=BooleanOptionalAction, default = False,
        help = 'If true, save potential')
    
    # Control key
    parser.add_argument('--k0', action=BooleanOptionalAction, default = True,
        help = 'If true, save controls')
    parser.add_argument('--sig0', action=BooleanOptionalAction, default = True,
        help = 'If true, save controls')

    # Analyse  
    parser.add_argument('--R', action=BooleanOptionalAction, default = False,
        dest = 'R', help = 'If true, calculate final position of centroid')        
    parser.add_argument('--U', action=BooleanOptionalAction, default = True,
        help = 'If true, calculate swimming speed U')
    parser.add_argument('--E', action=BooleanOptionalAction, default = True,
        help = 'If true, calculate L2 norm between real and preferred curvature')
    parser.add_argument('--A', action=BooleanOptionalAction, default = False,
        dest = 'A', help = 'If true, calculate real curvature amplitude')
 
    return parser

def sweep_N_dt_k(argv):
    '''
    Parameter sweep over time scale ratios a and b

    Show that swimming speed and energy are fully captured 
    by the system input time scale ratios.          
    '''    

    # Parse sweep parameter
    sweep_parser = default_sweep_parameter()    

    sweep_parser.add_argument('--dt_arr', 
        type=float, nargs='*', default = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4])    
    sweep_parser.add_argument('--N_arr', 
        type=int, nargs='*', default = [125, 250, 500, 750, 1000])    
    sweep_parser.add_argument('--k_arr', 
        type=int, nargs='*', default = [1, 2, 3, 4, 5])    

    sweep_param = sweep_parser.parse_known_args(argv)[0]    
    
    # The argumentparser for the sweep parameter has a boolean argument 
    # for ever frame key and control key which can be set to true
    # if it should be saved 
    FK = [k for k in FRAME_KEYS if getattr(sweep_param, k)]    
    CK = [k for k in CONTROL_KEYS if getattr(sweep_param, k)]

    # Parse model parameter
    model_parser = UndulationExperiment.parameter_parser()
    model_param = model_parser.parse_known_args(argv)[0]

    # Customize parameter
    model_param.pic_on = True
    model_param.Ds_h = 0.01
    model_param.Ds_t = 0.01
    model_param.s0_h = 0.05
    model_param.s0_t = 0.95
    model_param.use_c = True
    model_param.c = 1.0 
    model_param.T = 5.0
    model_param.dt_report = 0.01
    model_param.N_report = 125

    # Print all model parameter whose value has been
    # set via the command line
    cml_args = {k: v for k, v in vars(model_param).items() 
        if v != model_parser.get_default(k)}
    
    if len(cml_args) != 0: 
        print(cml_args)

    # Create the ParameterGrid over which we want to run
    # the undulation experiments
    dt_arr = sweep_param.dt_arr
    N_arr = sweep_param.N_arr
    k_arr = sweep_param.k_arr

    # dt_arr = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
    # N_arr = [125, 250, 500, 1000, 2000]
    
    dt_param = {'v_arr': dt_arr, 'round': 5}    
    N_param = {'v_arr': N_arr, 'round': None, 'int': True}
    k_param = {'v_arr': k_arr, 'round': None, 'int': True}

    grid_param = {'dt': dt_param, 'N': N_param, 'fdo': k_param}
    
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
        f'c={model_param.c}_lam={model_param.lam}_'
        f'dt_arr={dt_arr}_N_arr={N_arr}_k_arr_{k_arr}_'                
        f'T={model_param.T}_pic_on={model_param.pic_on}.h5')
    
    h5_filepath = sweep_dir / filename

    if sweep_param.pool:        
        Sweeper.save_sweep_to_h5(PG, h5_filepath, sim_dir, FK, CK)

    if sweep_param.analyse:
        sweep_param.R = True
        analyse(h5_filepath, what_to_calculate=sweep_param)    
    return

def sweep_dt_k(argv):
    '''
    Sweep over finite difference order 
    '''    

    # Parse sweep parameter
    sweep_parser = default_sweep_parameter()    

    sweep_parser.add_argument('--dt_arr', 
        type=int, nargs='*', default = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4])
    sweep_parser.add_argument('--k_arr', 
        type=int, nargs='*', default = [1, 2, 3, 4, 5])

    sweep_param = sweep_parser.parse_known_args(argv)[0]    
    
    # The argumentparser for the sweep parameter has a boolean argument 
    # for ever frame key and control key which can be set to true
    # if it should be saved 
    FK = [k for k in FRAME_KEYS if getattr(sweep_param, k)]    
    CK = [k for k in CONTROL_KEYS if getattr(sweep_param, k)]

    # Parse model paramete5
    model_parser = UndulationExperiment.parameter_parser()
    model_param = model_parser.parse_known_args(argv)[0]

    # Customize parameter
    model_param.pic_on = True
    model_param.Ds_h = 0.01
    model_param.Ds_t = 0.01
    model_param.s0_h = 0.05
    model_param.s0_t = 0.95
    model_param.use_c = True
    model_param.c = 1.0 
    model_param.T = 5.0
    model_param.N = 500
    model_param.dt_report = 0.01
    model_param.N_report = 125
    
    # Print all model parameter whose value has been
    # set via the command line
    cml_args = {k: v for k, v in vars(model_param).items() 
        if v != model_parser.get_default(k)}
    
    if len(cml_args) != 0: 
        print(cml_args)

    # Create the ParameterGrid over which we want to run
    # the undulation experiments
    dt_arr = sweep_param.dt_arr
    k_arr = sweep_param.k_arr
    
    dt_param = {'v_arr': dt_arr, 'round': 5}    
    k_param = {'v_arr': k_arr, 'round': None, 'int': True}    

    grid_param = {'dt': dt_param, 'fdo': k_param}
    
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
        f'c={model_param.c}_lam={model_param.lam}_'
        f'k_arr={k_arr}_'                
        f'T={model_param.T}_pic_on={model_param.pic_on}.h5')
    
    h5_filepath = sweep_dir / filename

    if sweep_param.pool:        
        Sweeper.save_sweep_to_h5(PG, h5_filepath, sim_dir, FK, CK)

    if sweep_param.analyse:
        sweep_param.R = True
        analyse(h5_filepath, what_to_calculate=sweep_param)    
    return

    
    pass    
    
def sweep_a_b(argv):
    '''
    Parameter sweep over time scale ratios a and b

    Show that swimming speed and energy are fully captured 
    by the system input time scale ratios.          
    '''    

    # Parse sweep parameter
    sweep_parser = default_sweep_parameter()    

    sweep_parser.add_argument('--a', 
        type=float, nargs=3, default = [-2, 3, 1.0])    
    sweep_parser.add_argument('--b', 
        type=float, nargs=3, default = [-3, 0, 1.0])    

    sweep_param = sweep_parser.parse_known_args(argv)[0]    
    
    # The argumentparser for the sweep parameter has a boolean argument 
    # for ever frame key and control key which can be set to true
    # if it should be saved 
    FK = [k for k in FRAME_KEYS if getattr(sweep_param, k)]    
    CK = [k for k in CONTROL_KEYS if getattr(sweep_param, k)]

    # Parse model parameter
    model_parser = UndulationExperiment.parameter_parser()
    model_param = model_parser.parse_known_args(argv)[0]

    # Customize parameter
    model_param.pic_on = True
    model_param.Ds_h = 0.01
    model_param.Ds_t = 0.01
    model_param.s0_h = 0.05
    model_param.s0_t = 0.95
    model_param.use_c = True
    model_param.c = 1.0 
    model_param.dt = 0.01
    model_param.N = 250
    model_param.T = 5.0
    model_param.dt_report = 0.01
    model_param.N_report = 125
    
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
    
    PG = ParameterGrid(vars(model_param), grid_param)

    grid_param = {'a': a_param, 'b': b_param} #'fdo': k_param}
    
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
        f'c={model_param.c}_lam={model_param.lam}_'
        f'T={model_param.T}_pic_on={model_param.pic_on}.h5')
    
    h5_filepath = sweep_dir / filename

    if sweep_param.pool:        
        Sweeper.save_sweep_to_h5(PG, h5_filepath, sim_dir, FK, CK)

    if sweep_param.analyse:
        sweep_param.R = True
        analyse(h5_filepath, what_to_calculate=sweep_param)    
    return

if __name__ == '__main__':
        
    parser = ArgumentParser()
    parser.add_argument('-sweep',  
        choices = ['N_dt_k', 'k'], help='Sweep to run')
            
    # Run function passed via command line
    args = parser.parse_known_args(argv)[0]    
    globals()['sweep_' + args.sweep](argv)



