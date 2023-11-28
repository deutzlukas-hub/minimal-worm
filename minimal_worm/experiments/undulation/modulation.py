'''
Created on 25 Nov 2023

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

# Local imports
from minimal_worm import FRAME_KEYS
from minimal_worm.worm import CONTROL_KEYS
from minimal_worm.experiments import Sweeper
from minimal_worm.experiments.undulation import UndulationExperiment
from minimal_worm.experiments.undulation import create_storage_dir
from minimal_worm.experiments.undulation.analyse_sweeps import analyse
from minimal_worm.experiments.undulation.thesis import default_sweep_parameter

ureg = pint.UnitRegistry()

def sweep_f_c_lam(argv):
    '''
    Parameter sweep over time scale ratios a and b

    Show that swimming speed and energy are fully captured 
    by the system input time scale ratios.          
    '''    
    # Parse sweep parameter
    sweep_parser = default_sweep_parameter()    

    sweep_parser.add_argument('--c', 
        type=float, nargs=3, default = [0.5, 1.6, 0.1])    
    sweep_parser.add_argument('--lam', 
        type=float, nargs=3, default = [0.5, 2.0, 0.1])    

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
    model_param.Ds_h = 0.01
    model_param.Ds_t = 0.01
    model_param.s0_h = 0.05
    model_param.s0_t = 0.95
    model_param.T = 5.0
    model_param.use_c = True
    
    # Print all model parameter whose value has been
    # set via the command line
    cml_args = {k: v for k, v in vars(model_param).items() 
        if v != model_parser.get_default(k)}
    
    if len(cml_args) != 0: 
        print(cml_args)
    
    #===============================================================================
    # Init ParameterGrid 
    #===============================================================================
    
    # Undulation frequency 
    
    # Operating-point at f_min
    a_min = 1.0
    b_min = 10**(-2.5)    
    
    log_f_over_f0_range = 2.5
    a_f_arr = a_min*np.logspace(0, log_f_over_f0_range, int(2e1))
    b_f_arr = b_min*np.logspace(0, log_f_over_f0_range, int(2e1))
        
    a_param = {'v_arr': a_f_arr.tolist(), 'round': 1}
    b_param = {'v_arr': b_f_arr.tolist(), 'round': 4}

    # Shape-factor and curvature amplitude     
    c_min, c_max = sweep_param.c[0], sweep_param.c[1]
    c_step = sweep_param.c[2]

    lam_min, lam_max = sweep_param.lam[0], sweep_param.lam[1]
    lam_step = sweep_param.lam[2]

    c_param = {'v_min': c_min, 'v_max': c_max + 0.1*c_step, 
        'N': None, 'step': c_step, 'round': 2}    

    lam_param = {'v_min': lam_min, 'v_max': lam_max + 0.1*lam_step, 
        'N': None, 'step': lam_step, 'round': 2}
    
    grid_param = {
        ('a', 'b'): (a_param, b_param),
        'c': c_param, 
        'lam': lam_param}
    
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
        f'a_min={a_min}_b_min={b_min}_f_range={log_f_over_f0_range}_'                
        f'c_min={c_min}_c_max={c_max}_c_step={c_step}_'
        f'lam_min={lam_min}_lam_max={lam_max}_lam_step={lam_step}_'
        f'N={model_param.N}_dt={model_param.dt}_'        
        f'T={model_param.T}_pic_on={model_param.pic_on}.h5')
    
    h5_filepath = sweep_dir / filename

    if sweep_param.pool:        
        Sweeper.save_sweep_to_h5(PG, h5_filepath, sim_dir, FK, CK)

    if sweep_param.analyse:
        sweep_param.A = False
        sweep_param.lam = False
        sweep_param.f = False
        sweep_param.lag = False                
        sweep_param.psi = False
        sweep_param.Y = True        
        analyse(h5_filepath, what_to_calculate=sweep_param)    
    return

if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument('-sweep',  
        choices = ['f_c_lam'], help='Sweep to run')
            
    # Run function passed via command line
    args = parser.parse_known_args(argv)[0]    
    globals()['sweep_' + args.sweep](argv)

