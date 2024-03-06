'''
Created on 5 Mar 2024

@author: amoghasiddhi
'''
from sys import argv
from types import SimpleNamespace
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
import pickle

from parameter_scan import ParameterGrid

# Local imports
from minimal_worm import FRAME_KEYS
from minimal_worm.worm import CONTROL_KEYS
from minimal_worm.worm import Worm
from minimal_worm.experiments.experiment import simulate_experiment
from minimal_worm.experiments.sweeper import Sweeper
from minimal_worm.experiments.undulation.thesis import default_sweep_parameter
from minimal_worm.experiments.actuation_relaxation.actuation_relaxation import ActuationRelaxationExperiment
from minimal_worm.experiments.actuation_relaxation.dirs import sim_dir, log_dir, sweep_dir, create_storage_dir


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
    parser.add_argument('--theta', action=BooleanOptionalAction, default = False,
        help = 'If true, save Euler angles')    
    parser.add_argument('--d1', action=BooleanOptionalAction, default = True,
        help = 'If true, save director 1')
    parser.add_argument('--d2', action=BooleanOptionalAction, default = True,
        help = 'If true, save director 2')
    parser.add_argument('--d3', action=BooleanOptionalAction, default = True,
        help = 'If true, save director 3')
    parser.add_argument('--k', action=BooleanOptionalAction, default = True,
        help = 'If true, save curvature')
    parser.add_argument('--sig', action=BooleanOptionalAction, default = False,
        help = 'If true, saved shear-stretch')
    parser.add_argument('--k_norm', action=BooleanOptionalAction, default = False,
        help = 'If true, save L2 norm of the real and preferred curvature difference')
    parser.add_argument('--sig_norm', action=BooleanOptionalAction, default = False,
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
    parser.add_argument('--D_F_dot', action=BooleanOptionalAction, default = False,
        help = 'If true, save fluid dissipation rate')
    parser.add_argument('--D_I_dot', action=BooleanOptionalAction, default = False,
        help = 'If true, save internal dissipation rate')
    parser.add_argument('--W_dot', action=BooleanOptionalAction, default = False,
        help = 'If true, save mechanical muscle power')
    parser.add_argument('--V_dot', action=BooleanOptionalAction, default = False,
        help = 'If true, save potential rate')
    parser.add_argument('--V', action=BooleanOptionalAction, default = False,
        help = 'If true, save potential')
    
    # Control key
    parser.add_argument('--k0', action=BooleanOptionalAction, default = True,
        help = 'If true, save controls')
    parser.add_argument('--sig0', action=BooleanOptionalAction, default = False,
        help = 'If true, save controls')
    
    return parser


def run_actuation_relaxation_experiment_example(argv):
    
    # Parse model parameter
    model_parser = ActuationRelaxationExperiment.parameter_parser()
    model_param = model_parser.parse_known_args(argv)[0]

    # Resolution
    model_param.dt = 0.01
    model_param.N = 750

    # Gradual muscle onset at head and tail
    model_param.Ds_h = 0.01
    model_param.Ds_t = 0.01
    model_param.s0_h = 0.05
    model_param.s0_t = 0.95
    model_param.T = 5.0
    
    model_param.tau_on = 0.05
    model_param.tau_off = 0.05
    model_param.t_on = 0.25
    model_param.t_off = 2.5
        
    # Set waveform
    model_param.use_c = True
    model_param.c = 1.0
    model_param.lam = 1.0
                
    # Control sequence
    CS = ActuationRelaxationExperiment.actuation_relaxation_control_sequence(model_param)

    # Run experiment             
    worm = Worm(model_param.N, model_param.dt, fdo = model_param.fdo, quiet=False)
    
    FS, CS, _, e, _  = simulate_experiment(worm, model_param, CS)    

    if e is not None:
        assert False
        
    # Save data
    FS_dict = SimpleNamespace(**{
        'r': FS.r,
        'd1': FS.d1,
        'd2': FS.d2,
        'd3': FS.d3,
        't': FS.t,  
        'k': FS.k
    })
        
    filepath= sim_dir / Path(
        f'raw_data_'
        f'c={model_param.c}_lam={model_param.lam}_'
        f'tau_off_{model_param.tau_off}_'                        
        f'N={model_param.N}_dt={model_param.dt}_'        
        f'T={model_param.T}.h5')
    

    pickle.dump(FS_dict, open(filepath, 'wb'))
        
            
    print(f'Saved file to {filepath}')
    
    return    

def sweep_c_lam(argv):
    '''
    Parameter sweep over c and lam for fixed a and b
    '''    

    # Parse sweep parameter
    sweep_parser = default_sweep_parameter()    

    sweep_parser.add_argument('--c', 
        type=float, nargs=3, default = [0.5, 1.5, 0.1])    
    sweep_parser.add_argument('--lam', 
        type=float, nargs=3, default = [0.5, 2.0, 0.1])    

    sweep_param = sweep_parser.parse_known_args(argv)[0]    
    
    # The argumentparser for the sweep parameter has a boolean argument 
    # for ever frame key and control key which can be set to true
    # if it should be saved 
    FK = [k for k in FRAME_KEYS if getattr(sweep_param, k)]    
    CK = [k for k in CONTROL_KEYS if getattr(sweep_param, k)]

    print(f'FK={FK}')

    # Parse model parameter
    model_parser = ActuationRelaxationExperiment.parameter_parser()
    model_param = model_parser.parse_known_args(argv)[0]

    # Resolution
    model_param.dt = 0.01
    model_param.N = 750

    # Gradual muscle onset at head and tail
    model_param.Ds_h = 0.01
    model_param.Ds_t = 0.01
    model_param.s0_h = 0.05
    model_param.s0_t = 0.95
    model_param.T = 5.0
    
    model_param.tau_on = 0.05
    model_param.tau_off = 0.05
    model_param.t_on = 0.25
    model_param.t_off = 2.5
        
    # Set waveform
    model_param.use_c = True
    model_param.c = 1.0
    model_param.lam = 1.0
    
    # Print all model parameter whose value has been
    # set via the command line
    cml_args = {k: v for k, v in vars(model_param).items() 
        if v != model_parser.get_default(k)}
    
    if len(cml_args) != 0: 
        print(cml_args)
    
    #===============================================================================
    # Init ParameterGrid 
    #===============================================================================
        
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
        'c': c_param, 
        'lam': lam_param}
    
    sweep_parser = default_sweep_parameter()    
        
    PG = ParameterGrid(vars(model_param), grid_param)

    a, b = model_param.a.magnitude, model_param.b.magnitude
    
    if sweep_param.save_to_storage:
        log_dir, sim_dir, sweep_dir = create_storage_dir()     
        
    # Experiments are run using the Sweeper class for parallelization 
    if sweep_param.run:
        Sweeper.run_sweep(
            sweep_param.worker, 
            PG, 
            ActuationRelaxationExperiment.actuation_relaxation_control_sequence, 
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
        f'a={round(a, 3)}_b={round(b, 3)}_'                
        f'c_min={c_min}_c_max={c_max}_c_step={c_step}_'
        f'lam_min={lam_min}_lam_max={lam_max}_lam_step={lam_step}_'
        f'N={model_param.N}_dt={model_param.dt}_'        
        f'T={model_param.T}.h5')
    
    h5_filepath = sweep_dir / filename

    if sweep_param.pool:        
        Sweeper.save_sweep_to_h5(PG, h5_filepath, sim_dir, FK, CK)
    
    print(f'Finished sweep! Save ParameterGrid to {PG_filepath}')
    
    
    
    return

if __name__ == '__main__':
       
    #run_actuation_relaxation_experiments(argv)
    
    # Make video

    parser = ArgumentParser()
    parser.add_argument('-sweep',  
        choices = ['c_lam'], help='Sweep to run')
            
    # Run function passed via command line
    args = parser.parse_known_args(argv)[0]    
    globals()['sweep_' + args.sweep](argv)


    print('Finished!')
        
        


    
    



    





