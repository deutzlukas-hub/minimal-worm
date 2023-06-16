'''
Created on 15 Jun 2023

@author: amoghasiddhi
'''
# Built-in
from sys import argv
from argparse import ArgumentParser, BooleanOptionalAction

# Third-party
from parameter_scan import ParameterGrid

# Local imports
from minimal_worm.experiments import Sweeper
from minimal_worm.experiments.undulation import UndulationExperiment
from minimal_worm.experiments.undulation import sweep_dir, log_dir, sim_dir

def default_sweep_parameter():
    '''
    Default sweep hyper parameter
    '''            
    parser = ArgumentParser(description = 'sweep-parameter')

    parser.add_argument('--worker', type = int, default = 10,
        help = 'Number of processes') 
    parser.add_argument('--simulate', action=BooleanOptionalAction, default = True,
        help = 'If true, simulations are run from scratch') 
    parser.add_argument('--save_raw_data', action=BooleanOptionalAction, default = True,
        help = 'If true, FrameSequences are pickled to disk') 
    parser.add_argument('--overwrite', action=BooleanOptionalAction, default = False,
        help = 'If true, already existing simulation results are overwritten')
    parser.add_argument('--debug', action=BooleanOptionalAction, default = False,
        help = 'If true, exception handling is turned off which is helpful for debugging')    

    return parser

def compute_swimming_speed():
    
    pass

def compute_energies():
    
    pass

def sweep_a_b(argv):
    '''
    Parameter sweep over time scale ratios a and b

    Show that swimming speed and energy are fully captured 
    by the system input time scale ratios.          
    '''    

    # parse sweep parameter
    sweep_parser = default_sweep_parameter()    

    sweep_parser.add_argument('--a', 
        type=float, nargs=3, default = [-2, 3, 0.2])    
    sweep_parser.add_argument('--b', 
        type=float, nargs=3, default = [-3, 0, 0.2])    

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

    a_param = {'v_min': a_min, 'v_max': a_max, 
        'N': None, 'step': a_step, 'round': 4, 'log': True}    

    b_param = {'v_min': b_min, 'v_max': b_max, 
        'N': None, 'step': b_step, 'round': 5, 'log': True}

    grid_param = {'a': a_param, 'b': b_param}

    PG = ParameterGrid(vars(model_param), grid_param)

    # Run sweep
    Sweeper.run_sweep(
        sweep_param.worker, 
        PG, 
        UndulationExperiment.stw_control_sequence, 
        log_dir, 
        sim_dir, 
        sweep_param.overwrite, 
        sweep_param.debug,
        'UExp')

    # # Run sweep
    # filename = (
    #     f'raw_data_'
    #     f'a_min={a_min}_a_max={a_max}_a_step={a_step}_'
    #     f'b_min={b_min}_b_max={b_max}_b_step={b_step}_'        
    #     f'A_{model_param.A}_lam_{model_param.lam}_'
    #     f'N={model_param.N}_dt={model_param.dt}.h5')        
    #
    # filename = ()
    #
    # h5_filepath = sweep_dir / filename 
    # exper_spec = 'UE'
    #
    # h5_raw_data = sim_exp_wrapper(sweep_param.simulate, 
    #     sweep_param.save_raw_data,       
    #     ['x', 'Omega', 
    #     'V_dot_k', 'V_dot_sig', 'D_k', 'D_sig', 
    #     'dot_W_F_lin', 'dot_W_F_rot', 
    #     'dot_W_M_lin', 'dot_W_M_rot'],
    #     ['Omega'],                        
    #     sweep_param.worker, 
    #     PG, 
    #     UndulationExperiment.sinusoidal_traveling_wave_control_sequence,
    #     h5_filepath,
    #     log_dir,
    #     sim_dir,
    #     exper_spec,
    #     sweep_param.overwrite,
    #     sweep_param.debug)
    #
    # h5_filepath = sweep_dir / h5_filename
    # h5_results = h5py.File(h5_filepath, 'w')
    # h5_results.attrs['grid_filename'] = PG.filename + '.json' 
    #
    # compute_swimming_speed(h5_raw_data, h5_results, PG)
    # compute_undulation_kinematics(h5_raw_data, h5_results, PG)
    # compute_energies(h5_raw_data, h5_results, PG)
    #
    # h5_results.close()    
    # h5_raw_data.close()
    #
    # print(f'Saved swimming speed, kinematics and energies {h5_filepath}')

    return

if __name__ == '__main__':
        
    parser = ArgumentParser()
    parser.add_argument('-sweep',  
        choices = ['a_b'], 
        help='Sweep to run')
    # Run function passed via command line
    args = parser.parse_known_args(argv)[0]    
    globals()['sweep_' + args.sweep](argv)





