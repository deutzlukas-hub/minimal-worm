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
    Saves sweeps to HDF5
    '''
                                        
    @staticmethod    
    def save_data(
            filepath: Path, 
            PG: ParameterGrid,
            sim_dir: Path, 
            FS_keys: List, 
            CS_keys: Tuple[List, None], 
            overwrite = True):
        '''
        Saves simulation output over a given ParameterGrid
        into single HDF5 file.
        '''
        
        assert not PG.has_key('T'), ('ParameterGrid sweeps over T, but simulations times ' 
            'are expected to be identical for all simulations')
                
        if filepath.exists():
            if not overwrite:
                assert (f'HDF5 file {filepath} already exists.' 
                    'Set overwrite=True to overwrite existing file.')
                return

        if PG.base_parameter['dt_report'] is not None: 
            dt = PG.base_parameter['dt_report']
        else: 
            dt = PG.base_parameter['dt']
        
        # Number of time steps 
        n = int(round(PG.base_parameter['T'] / dt) )                
                                                                                                                                                                  
        sim_filepaths = [sim_dir / (h + '.dat') for h in PG.hash_arr]        

        # Find first simulation which succeeded  
        for sim_filepath in sim_filepaths:  
            with open(sim_filepath, 'rb') as f:                      
                data = pickle.load(f)
                if data['exit_status'] == 0:
                    break
            
        FS = data['FS']
        CS = data['CS']
                                                            
        # HDF5 data file
        h5 = h5py.File(filepath, 'w')
        h5.attrs['grid_filename'] = PG.filename + '.json'       
        h5.attrs['shape'] = PG.shape        
        h5.create_dataset('exit_status', shape = len(PG), dtype = float)
        h5.create_dataset('t', data = FS.t)

        # Allocate arrays for frame attributes                                
        FS_grp = h5.create_group('FS')
        
        for key in FS_keys:                        
            shape = (len(PG), ) + getattr(FS, key).shape            
            FS_grp.create_dataset(key, shape = shape, dtype = float)

        if CS_keys is not None:        
            CS_grp = h5.create_group('CS')
            for key in CS_keys:                
                shape = (len(PG), ) + getattr(CS, key).shape
                CS_grp.create_dataset(key, shape = shape, dtype = float)
                                                
        # Load output from pickled simulation files        
        Saver._populate_array(h5, sim_filepaths, n, FS_keys, CS_keys)

        return h5
    
    @staticmethod
    def _pad_array(n, arr):
        '''
        Pads missing time steps for failed simulations with nans
        '''                
        shape = (n,) + arr.shape[1:]                                                                                 
        pad_arr = np.zeros(shape)[:] = np.nan                                
        pad_arr[:arr.size(0)] = arr

        return pad_arr
        
    @staticmethod
    def _populate_array(
            h5: h5py.File,
            sim_filepaths: List[str],
            n: float, 
            FS_keys: List, 
            CS_keys: Tuple[List, None]):                                    
        '''        
        Loads data from pickled FrameSequence and ControlSequence
        specified by given keys. 
        '''                                                 
                                                                                
        for i, filepath in enumerate(sim_filepaths): 

            with open(filepath, 'rb') as f:                
                data = pickle.load(f)
                for key in FS_keys:                
                    arr = getattr(data['FS'], key)
                    if data['exit_status'] == 1:
                        arr = Saver._pad_array(n, arr)
                    h5['FS'][key][i, :] = arr
                                                            
                if CS_keys is not None:
                    for key in CS_keys: 
                        arr = getattr(data['CS'], key)
                        if data['exit_status'] == 1:                                        
                            arr = Saver._pad_array(n, arr, )                    
                        h5['CS'][key][i, :] = arr
    
                h5['exit_status'][i] = data['exit_status']
                                                                
        return 
    
 