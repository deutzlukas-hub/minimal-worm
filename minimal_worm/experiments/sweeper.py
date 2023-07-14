'''
Created on 15 Jun 2023

@author: amoghasiddhi
'''
# Built-in
from os.path import isfile, join
from typing import Callable, Dict, List
from pathlib import Path
from argparse import Namespace
import pickle

# Third-party
import numpy as np

# Local imports
from .saver import Saver
from minimal_worm.experiments import simulate_experiment
from minimal_worm import Worm, FrameSequence, ModelParameter
from parameter_scan import ParameterGrid
from mp_progress_logger import FWProgressLogger, FWException

class Sweeper():
    '''
    Sweeps parameter space and runs simulations
    '''
        
    @staticmethod
    def save_output(
        filepath: Path, 
        FS: FrameSequence, 
        CS: Namespace, 
        MP: ModelParameter, 
        param: Dict, 
        exit_status: int = 1):
        
        '''
        Save simulation results
        
        :param filepath (str):
        :param filename (str):
        :param FS (FrameSequenceNumpy):
        :param CS (ControlSequenceNumpy):
        :param MP (...):
        :param param (dict):
        :param exit_status (int): if 0, then the simulation finished succesfully, if 1, error occured  
        '''
    
        output = {}
    
        output['param'] = param
        output['MP'] = MP
        output['exit_status'] = exit_status
            
        output['FS'] = FS        
        output['CS'] = CS
                                 
        with open(filepath, 'wb') as file:                                                                                
            pickle.dump(output, file)
        
        return
    
    @staticmethod
    def wrap_simulate_experiment(_input, 
                                 pbar,
                                 logger,
                                 task_number,                             
                                 create_CS,
                                 FK,
                                 sim_dir,  
                                 overwrite  = False, 
                                 save_keys = None,                              
                                 ):
        '''
        Wrapes simulate_experiment function to make it compatible with parameter_scan module. 
            
        :param _input (tuple): Parameter dictionary and hash
        :param create_CS (function): Creates control sequence from param
        :param pbar (tqdm.tqdm): Progressbar
        :param logger (logging.Logger): Progress logger
        :param task_number (int): Number of tasks
        :param sim_dir (str): Result directory
        :param overwrite (bool): If true, exisiting files are overwritten
        :param save_keys (list): List of attributes which will be saved to the result file. 
            If None, then all attributes get saved.        
        '''
    
        param, param_hash = _input[0], _input[1]
        
        filepath = join(sim_dir, (param_hash) + '.dat')
                
        if not overwrite:    
            if isfile(filepath):
                logger.info(f'Task {task_number}: File already exists')                                    
                output = pickle.load(open(filepath, 'rb'))
                FS = output['FS']
                            
                exit_status = output['exit_status'] 
                            
                if exit_status:
                    raise FWException(FS.pic, param['T'], param['dt'], FS.times[-1])
                
                result = {}
                result['pic'] = None
                
                return result
             
        worm = Worm(param['N'], param['dt'], fdo = param['fdo'], quiet=True)
        
        # Experiment 
        param_ns = Namespace()
        param_ns.__dict__.update(param)
        
        CS = create_CS(param)
    
        FS, CS, MP, e = simulate_experiment(worm, param_ns, CS, 
            FK = FK, pbar = pbar, logger = logger)
                            
        if e is not None:
            exit_status = 1
        else:
            exit_status = 0
                    
        # Regardless if the simulation has finished or failed, simulation results
        # up to this point are saved to file         
        Sweeper.save_output(filepath, FS, CS, MP, param, exit_status)                        
        logger.info(f'Task {task_number}: Saved file to {filepath}.')         
                    
        # If the simulation has failed then we reraise the exception
        # which has been passed upstream        
        if e is not None:         
                
            raise FWException(None, 
                              param['T'], 
                              param['dt'], 
                              FS.t[-1]) from e
            
        # If simulation has finished succesfully then we return the relevant results 
        # for the logger
        result = {}    
        result['pic'] = None
        
        return result


    @staticmethod
    def run_sweep(
            N_worker: int, 
            PG: ParameterGrid, 
            create_CS: Callable,                    
            FK: List[str],            
            log_dir: Path,
            sim_dir: Path,
            overwrite = False,
            debug = False,
            exper_spec = ''):
        
        '''
        Runs the experiment defined by the task function for all parameters in 
        ParameterGrid 
            
        :param N_worker (int): Number of processes
        :param PG (ParameterGrid): Parameter grid
        :param task (funtion): function
        :param log_dir (str): log file directory
        :param sim_dir (str): output directory
        :param overwrite (boolean): If true, existing files are overwritten
        :param exper_spec (str): experiment descriptor
        :param debug (boolean): Set to true, to debug 
        '''
            
        # Creater status logger for experiment
        # The logger will log and display
        # the progress and outcome of the simulations
        PGL = FWProgressLogger(PG, 
            str(log_dir), 
            pbar_to_file = False,                        
            pbar_path = './pbar/pbar.txt', 
            exper_spec = exper_spec,
            debug = debug)
    
        # Start experiment pool
        PGL.run_pool(N_worker, 
            Sweeper.wrap_simulate_experiment, 
            create_CS,
            FK,
            str(sim_dir),                 
            overwrite = overwrite)

        PGL.close()
        
        return 
                
    @staticmethod
    def save_sweep_to_h5(
            PG: ParameterGrid,                
            h5_filepath: Path,
            sim_dir: Path,
            FS_keys = ['r', 'theta', 'sig','k'], 
            CS_keys = None):    
        
        '''
        Pools experiment results and saves them to single HDF5
        
        :param PG (ParameterGrid): Parameter grid object
        :param h5_filepath (str): HDF5 filepath  
        :param sim_dir (str): Output directory FrameSequence
        :param log_dir (str): Log file directory    
        :param FS_keys (list): List of frame variables which are saved to h5
        :param CS_keys (list): List of vontrol variables which are saved to h5
        '''    
        
        # Save results to HDF5            
                                    
        h5 = Saver.save_data(h5_filepath, PG, sim_dir, FS_keys, CS_keys)            
    
        exit_status = h5['exit_status'][:]
    
        print(f'Finished simulations: {np.sum(exit_status)/len(exit_status)*100}% failed')    
        print(f'Saved parameter scan simulations results to {h5_filepath}')
        
        return h5



