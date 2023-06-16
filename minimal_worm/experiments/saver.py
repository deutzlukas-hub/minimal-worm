'''
Created on 15 Jun 2023

@author: amoghasiddhi
'''
#Built-in
from typing import List, Tuple
from pathlib import Path
from abc import ABC

# Third-partys
import numpy as np
import pickle
import h5py 

# Local 
from parameter_scan import ParameterGrid

class Saver(ABC):
    '''
    Saves sweeps to HDF%
    '''
                                    
#------------------------------------------------------------------------------ 
# pool sweep and save results to hdf5
    
    @staticmethod
    def pad_time(
            n: int,
            t_list,
            exit_status_arr: List):
        '''
        Pads time arrays with nans for failed simulation
        '''
                
        if np.all(np.logical_not(exit_status_arr)):
            return t_list
        
        pad_t_list = []
        
        for t, exit_status in zip(t_list, exit_status_arr):
        
            if not exit_status:
                continue
                            
            pad_t= np.zeros(n)
            pad_t[:len(t)] = t
            pad_t_list.append(pad_t)
        
        return np.array(pad_t_list)
        
    @staticmethod
    def pad_arrays(
            n: int,
            arr_list: List, 
            exit_status_arr: List):
        '''
        Pads failed simulation results with nans.        
        '''       
                
        pad_arr_list = []

        # Desired shape (n x 3 x N) where N is the number of body points  
        shape = (n,) + arr_list[0].shape[1:]                         

        for arr, exit_status in zip(arr_list, exit_status_arr):
            
            if not exit_status:
                continue
                                
                pad_arr = np.zeros(shape)[:] = np.nan                                
                pad_arr[:arr.size(0)] = arr
                pad_arr_list.append(pad_arr)
            
        return pad_arr_list

    @staticmethod
    def pool_data(
            PG: ParameterGrid,
            sim_dir: Path,
            FS_keys: List, 
            CS_keys: Tuple[List, None]):                            
        
        '''        
        Loads data from pickled FrameSequence and ControlSequence
        specified by given keys. 
        '''          
        
        # List of simulation file paths associated with given ParameterGrid
        sim_filepaths = [sim_dir / (h + '.dat') for h in PG.hash_arr]        

        # Simulation output are pooled into dictionary                
        output = {}                                
        output['FS'] = {key: [] for key in FS_keys}        
        
        if CS_keys is not None:
            output['CS'] = {key: [] for key in CS_keys}                                                            
        
        output['exit_status'] = []
        output['t'] = []
                        
        for filepath in sim_filepaths: 
                    
            data = pickle.load(open(filepath, 'rb'))

            for key in FS_keys:                
                output['FS'][key].append(getattr(data['FS'], key))
            
            if CS_keys is not None:
                for key in CS_keys: 
                    output['CS'][key].append(getattr(data['CS'], key))

            output['t'].append(data['FS'].times)            
            output['exit_status'].append(data['exit_status'])

        # If all simulation succeeded we only
        # need to save timestamps once        
        if np.all(output['exit_status']):
            output['t'] = data['FS'].times
                                                                
        return output        
    
    @staticmethod    
    def save_data(
            filepath: Path, 
            PG: ParameterGrid,
            sim_dir: Path, 
            FS_keys: List, 
            CS_keys: Tuple[List, None], 
            overwrite = True):
        '''
        Pools simulation and saves simulation output over a given ParameterGrid
        into single HDF5 file.
        '''
        
        assert not PG.has_key('T'), ('ParameterGrid sweeps over T, but simulations times ' 
            'are expected to be identical for all simulations')
                
        if filepath.exists():
            if not overwrite:
                assert (f'HDF5 file {filepath} already exists.' 
                    'Set overwrite=True to overwrite existing file.')
                return
                                                                                                    
        # Create
        h5 = h5py.File(filepath, 'w')
        h5.attrs['grid_filename'] = PG.filename + '.json'       

        # Load output from pickled simulation files        
        output = Saver.pool_data(PG, sim_dir, FS_keys, CS_keys)
            
        exit_status_arr = output['exit_status']
        h5.create_dataset('exit_status', data = exit_status_arr) 
                
        FS_grp = h5.create_group('FS')

        # If a simulation has failed than we need to pad its 
        # data arrays with nans so that it has the same shape
        # as the other simulations
        if np.all(np.logical_not(exit_status_arr)):
            
            needs_padding = True
                    
            if PG.base_parameter['dt_report'] is not None: 
                dt = PG.base_parameter['dt_report']
            else: 
                dt = PG.base_parameter['dt']
            
            # Desired number of time steps for padding
            n = int(round(PG.base_parameter['T'] / dt) )                
                
        for key, arr_list in output['FS'].items():            
            
            if needs_padding:
                arr_list = Saver.pad_arrays(n, arr_list, exit_status_arr)            
            
            FS_grp.create_dataset(key, data = np.array(arr_list))
            
        CS_grp = h5.create_group('CS')
            
        for key, arr in output['CS'].items():            
            
            CS_grp.create_dataset(key, data = np.array(arr))

        if needs_padding: 
            t = Saver.pad_time(n, output['t'], exit_status_arr)
        
        h5.create_dataset('t', data = t)
        
        return h5
 