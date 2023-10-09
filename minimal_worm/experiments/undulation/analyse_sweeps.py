'''
Created on 17 Jun 2023

@author: amoghasiddhi
'''
# Built-in
from sys import argv
from typing import Tuple, List
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace
        
# Third-party
import numpy as np
import h5py 
from scipy.integrate import trapz

# Local imports
from minimal_worm.experiments import PostProcessor
from minimal_worm import POWER_KEYS

    
def analyse(
        raw_data_filepath: Path,
        analysis_filepath: Tuple[Path, None] = None,
        what_to_calculate: List[str] = None
):
            
    if analysis_filepath is None:
        assert raw_data_filepath.name.startswith('raw_data') 
        analysis_filepath = ( raw_data_filepath.parent
            / raw_data_filepath.name.replace('raw_data', 'analysis'))
                
    h5_raw_data = h5py.File(raw_data_filepath, 'r')
    h5_analysis = h5py.File(analysis_filepath, 'w')

    h5_analysis.create_dataset('sim_t', data = h5_raw_data['sim_t'][:].reshape(h5_raw_data.attrs['shape']))
    h5_analysis.attrs.update(h5_raw_data.attrs)
    
    # Compute energies from last period
    T = h5_raw_data.attrs['T']
    Delta_t = T - 1
    
    if what_to_calculate.R:
        R = compute_final_centroid_destination(h5_raw_data)
        h5_analysis.create_dataset('R', data = R)            
    
    if what_to_calculate.U:                    
        U = compute_swimming_speed(h5_raw_data, Delta_t)
        h5_analysis.create_dataset('U', data = U)
    
    if what_to_calculate.E:    
        E_dict = compute_energies(h5_raw_data) 
        grp = h5_analysis.create_group('energies')
        for k, E in E_dict.items():
            grp.create_dataset(k, data = E)         
    
    if what_to_calculate.k_norm:
        k_norm = compute_average_curvature_norm(h5_raw_data, Delta_t)        
        h5_analysis.create_dataset('k_norm', data = k_norm)
    
    if what_to_calculate.sig_norm:
        sig_norm = compute_average_sig_norm(h5_raw_data, Delta_t)
        h5_analysis.create_dataset('sig_norm', data = sig_norm)            
    
    if what_to_calculate.A:    
        A_max, A_min = compute_average_curvature_amplitude(h5_raw_data, Delta_t)        
        h5_analysis.create_dataset('A_max', data = A_max)
        h5_analysis.create_dataset('A_min', data = A_min)
           
    print(f'Saved Analysis to {analysis_filepath}')    
    
    return

def compute_final_centroid_destination(h5: h5py):
    '''
    Computes the final centroid destination
    :param h5:
    '''

    r = h5['FS']['r'][:, -1, :, :]        
    R = r.mean(axis = 2)
    
    return R.reshape(np.append(h5.attrs['shape'], 3))

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

def compute_average_curvature_amplitude(h5: h5py, Delta_t: float = 2.0):
    '''
    Computes curvature amplitude    
    '''    
    t = h5['t'][:]    
    T = h5.attrs['T']
    
    # Only look at last period
    idx_arr = t >= (T - 1)

    # For planar undulation, we only need to consider 
    # the first element of the curvature vector                    
    k_arr = h5['FS']['k'][:, idx_arr, 0, :]
    
    # Max and min along time dimension
    k_max_arr = k_arr.max(axis = 1)     
    k_min_arr = k_arr.min(axis = 1) 

    # Choose midpoint along body 
    idx_mp = int(k_arr.shape[-1] / 2)    
    A_max = k_max_arr[:, idx_mp]
    A_min = k_min_arr[:, idx_mp]
    
    return A_max.reshape(h5.attrs['shape']), A_min.reshape(h5.attrs['shape'])

def compute_undulation_wavelength(h5: h5py, Delta_t: float = 2.0):
    '''
    Compute undulation wavelength
    '''    
    
    # 1.Fit straight to centre 80 % body length
    # 2.This gives me undulation speed c. given the undulation 
    # frequency 
    # 3.Calculate wavelength from undulation speed
    
    # TODO: Let's do this tomorrow!
        
    t = h5['t'][:]    
    T = h5.attrs['T']
    
    # Only look at last period
    idx_arr = t >= (T - 1)

    # For planar undulation, we only need to consider 
    # the first element of the curvature vector                    
    k_arr = h5['FS']['k'][:, idx_arr, 0, :]
    
    # Max and min along time dimension
    k_max_arr = k_arr.max(axis = 1)     
    k_min_arr = k_arr.min(axis = 1) 

    # Choose midpoint along body 
    idx_mp = int(k_arr.shape[-1] / 2)    
    A_max = k_max_arr[:, idx_mp]
    A_min = k_min_arr[:, idx_mp]
    
    return A_max.reshape(h5.attrs['shape']), A_min.reshape(h5.attrs['shape'])

def compute_swimming_speed(h5: h5py, Delta_t: float):
    '''
    Computes swimming speed for every simulation in h5
    '''
    
    U_arr = np.zeros(h5['FS']['r'].shape[0])
    t = h5['t'][:]
        
    for i, r in enumerate(h5['FS']['r']):

        U_arr[i] = PostProcessor.comp_mean_swimming_speed(r, t, Delta_t)[0]
        
    return U_arr.reshape(h5.attrs['shape'])
                
def compute_energies(h5: h5py): #Delta_t: float = 2.0):
    '''
    Computes energy cost  and mechanical work per undulation period 
    '''
    
    t = h5['t'][:]    
    dt = t[1] - t[0]
    t_start = h5.attrs['T'] - 1.0
    idx_arr = t >= t_start
    
    E_dict = {}
    
    # Iterate over powers
    for P_key in POWER_KEYS:                        
        # Iterate over all simulations 
        P = h5['FS'][P_key][:, idx_arr]
        E = trapz(P, dx = dt, axis = 1) 
                                    
        E_key = PostProcessor.engery_names_from_power[P_key]            
        E_dict[E_key] = E.reshape(h5.attrs['shape'])
        
    return E_dict

if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument('-analyse',  
        choices = ['a_b'], help='Sweep to run')
    parser.add_argument('-input',  help='HDF5 raw data filepath')
    parser.add_argument('-output', help='HDF5 output filepath',
        default = None)

    args = parser.parse_args(argv)[0]    
    globals()['analyse_' + args.sweep](args.input, args.output)

                    
    


        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    
