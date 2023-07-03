'''
Created on 17 Jun 2023

@author: amoghasiddhi
'''
# Built-in
from sys import argv
from typing import Tuple
from argparse import ArgumentParser
from pathlib import Path
    
# Third-party
import numpy as np
import h5py 

# Local imports
from minimal_worm.experiments import PostProcessor
from minimal_worm import POWER_KEYS
from minimal_worm.experiments.undulation import log_dir, sweep_dir

def compute_average_curvature_norm(h5: h5py, Delta_t: float = 2.0):
    '''
    Computes time averaged L2 norm of curvature minus preferred curvature
    for every simulation in h5
    '''

    t = h5['t'][:]
    idx_arr = t >= Delta_t
        
    k_norm_arr = h5['FS']['k_norm'][:, idx_arr]
    k_avg_norm = k_norm_arr.mean(axis = 1)
    
    return k_avg_norm.reshape(h5.attrs['shape'])

def compute_average_sig_norm(h5: h5py, Delta_t: float = 2.0):

    t = h5['t'][:]
    idx_arr = t >= Delta_t
        
    sig_norm_arr = h5['FS']['sig_norm'][:, idx_arr]
    sig_avg_norm = sig_norm_arr.mean(axis = 1)
        
    return sig_avg_norm.reshape(h5.attrs['shape'])

def compute_swimming_speed(h5: h5py, Delta_t: float):
    '''
    Computes swimming speed for every simulation in h5
    '''
    
    U_arr = np.zeros(h5['FS']['r'].shape[0])
    t = h5['t'][:]
        
    for i, r in enumerate(h5['FS']['r']):

        U_arr[i] = PostProcessor.comp_mean_swimming_speed(r, t, Delta_t)[0]
        
    return U_arr.reshape(h5.attrs['shape'])
                
def compute_energies(h5: h5py, Delta_t: float = 2.0):
    '''
    Computes energy cost  and mechanical work per undulation period 
    '''
    
    t = h5['t'][:]    
    N_undu = h5.attrs['T'] - Delta_t
    
    E_dict = {}
    
    # Iterate over powers
    for P_key in POWER_KEYS:                        
        E = np.zeros(h5['FS']['r'].shape[0])        
        # Iterate over all simulations 
        for i, P in enumerate(h5['FS'][P_key]):            
            E[i] = PostProcessor.comp_energy_from_power(P, t, Delta_t) / N_undu
            
        E_key = PostProcessor.engery_names_from_power[P_key]            
        E_dict[E_key] = E.reshape(h5.attrs['shape'])
        
    return E_dict

def analyse_a_b(
        raw_data_filepath: Path,
        analysis_filepath: Tuple[Path, None] = None):

    if analysis_filepath is None:
        assert raw_data_filepath.name.startswith('raw_data') 
        analysis_filepath = ( raw_data_filepath.parent
            / raw_data_filepath.name.replace('raw_data', 'analysis'))
                
    h5_raw_data = h5py.File(raw_data_filepath, 'r')
    h5_analysis = h5py.File(analysis_filepath, 'w')

    h5_analysis.attrs.update(h5_raw_data.attrs)
    
    # Compute energies from last period
    T = h5_raw_data.attrs['T']
    Delta_t = T - 1
                
    U = compute_swimming_speed(h5_raw_data, Delta_t)
    E_dict = compute_energies(h5_raw_data, Delta_t)
    k_norm = compute_average_curvature_norm(h5_raw_data, Delta_t)
    sig_norm = compute_average_sig_norm(h5_raw_data, Delta_t)
    
    h5_analysis.create_dataset('U', data = U)
    h5_analysis.create_dataset('k_norm', data = k_norm)
    h5_analysis.create_dataset('sig_norm', data = sig_norm)
   
    grp = h5_analysis.create_group('energies')
    
    for k, E in E_dict.items():

        grp.create_dataset(k, data = E) 

    print(f'Saved Analysis to {analysis_filepath}')
         
if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument('-analyse',  
        choices = ['a_b'], help='Sweep to run')
    parser.add_argument('-input',  help='HDF5 raw data filepath')
    parser.add_argument('-output', help='HDF5 output filepath',
        default = None)

    args = parser.parse_args(argv)[0]    
    globals()['analyse_' + args.sweep](args.input, args.output)

                    
    


        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    
