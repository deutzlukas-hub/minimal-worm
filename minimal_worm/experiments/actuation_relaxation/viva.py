'''
Created on 5 Mar 2024

@author: amoghasiddhi
'''
from sys import argv
from pathlib import Path
from fenics import Function
import pickle

from minimal_worm.experiments.sweeper import Sweeper

# Local imports
from minimal_worm.experiments.actuation_relaxation.actuation_relaxation import ActuationRelaxationExperiment
from minimal_worm.experiments.experiment import simulate_experiment
from minimal_worm.experiments.actuation_relaxation.dirs import sim_dir, log_dir, video_dir
from minimal_worm.worm import Worm

def run_actuation_relaxation_experiments(argv):
    
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
    
    FS, CS, MP, e, sim_t  = simulate_experiment(worm, model_param, CS)    

    if e is not None:
        assert False
        
    # Save data

    filepath= sim_dir / Path(
        f'raw_data_'
        f'c={model_param.c}_lam={model_param.lam}_'
        f'tau_off_{model_param.tau_off}_'                        
        f'N={model_param.N}_dt={model_param.dt}_'        
        f'T={model_param.T}.h5')
    
    Sweeper.save_output(filepath, FS, CS, MP, model_param, e, sim_t)
        
    print(f'Saved file to {filepath}')
    
    return    

# def generate_worm_video(filepath, output_path = None):
#
#     with open(filepath, 'rb') as f
#         data = pickle.load(f)
#         FS = data['FS'] 
#
#     if output_path is None:
#         output_path = filepath.stem
#
#     WormStudio.generate_clip(
#         output_path, 
#         add_trajectory = False, 
#         add_centreline = True, 
#         add_frame_vectors = False, 
#         add_surface = True, 
#         draw_e1 = False, 
#         draw_e2 = False, 
#         draw_e3 = False) 
#
#         #n_arrows, 
#         #fig_width, 
#         #fig_height, 
#         #rel_camera_distance, 
#         #azim_offset, 
#         #revolution_rate, 
#         #T_max)
    
    
    return
    
if __name__ == '__main__':
    
        
    run_actuation_relaxation_experiments(argv)
    
    # Make video
    filename = 'raw_data_c=1.0_lam=1.0_tau_off_0.05_N=750_dt=0.01_T=5.0.h5'
    
    print('Finished!')
        
        


    
    



    





