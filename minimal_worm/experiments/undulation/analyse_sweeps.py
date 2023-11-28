'''
Created on 17 Jun 2023

@author: amoghasiddhi
'''
# Built-in
from sys import argv
from typing import Tuple, List
from argparse import ArgumentParser
from pathlib import Path
        
# Third-party
import numpy as np
import h5py 
from scipy.integrate import trapz
from scipy.signal import correlate
from scipy.interpolate import interp1d
from scipy.interpolate import splrep, splev
from sklearn.cluster import DBSCAN

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
    
    if what_to_calculate.R:
        R = compute_final_centroid_destination(h5_raw_data)
        h5_analysis.create_dataset('R', data = R)            
    
    if what_to_calculate.U:                    
        U = compute_swimming_speed(h5_raw_data)
        h5_analysis.create_dataset('U', data = U)
    
    if what_to_calculate.E:    
        E_dict = compute_energies(h5_raw_data) 
        grp = h5_analysis.create_group('energies')
        for k, E in E_dict.items():
            grp.create_dataset(k, data = E)         
    
    if what_to_calculate.k_norm:
        k_norm = compute_average_curvature_norm(h5_raw_data)        
        h5_analysis.create_dataset('k_norm', data = k_norm)
    
    if what_to_calculate.sig_norm:
        sig_norm = compute_average_sig_norm(h5_raw_data)
        h5_analysis.create_dataset('sig_norm', data = sig_norm)            
    
    if what_to_calculate.f:
        f_avg, f_std = compute_undulation_frequency(h5_raw_data, Delta_t = 1.0)                
        h5_analysis.create_dataset('f', data = f_avg)
        h5_analysis.create_dataset('f_std', data = f_std)

    if what_to_calculate.lag:
        lag_avg, lag_std = compute_time_lag(h5_raw_data, Delta_t = 2.0)                
        h5_analysis.create_dataset('lag', data = lag_avg)
        h5_analysis.create_dataset('lag_std', data = lag_std)
    
    if what_to_calculate.A:    
        A_avg, A_std = compute_curvature_amplitude(h5_raw_data)        
        h5_analysis.create_dataset('A', data = A_avg)
        h5_analysis.create_dataset('A_std', data = A_std)
                
    if what_to_calculate.lam:
        lam_avg, lam_std = compute_undulation_wavelength(h5_raw_data)                
        h5_analysis.create_dataset('lam', data = lam_avg)
        h5_analysis.create_dataset('lam_std', data = lam_std)

    if what_to_calculate.psi:
        psi_avg, psi_std = compute_angle_attack(h5_raw_data)                
        h5_analysis.create_dataset('psi', data = psi_avg)
        h5_analysis.create_dataset('psi_std', data = psi_std)

    if what_to_calculate.Y:
        Y_avg, Y_max = compute_wobbling_speed(h5_raw_data)
        h5_analysis.create_dataset('Y_avg', data = Y_avg)
        h5_analysis.create_dataset('Y_max', data = Y_max)
                   
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

def compute_average_curvature_norm(h5: h5py):
    '''
    Computes time averaged L2 norm of curvature minus preferred curvature
    for every simulation in h5
    '''

    t = h5['t'][:]
    T = h5.attrs['T']

    t_idx_arr = t >= (T - 1)        
        
    k_norm_arr = h5['FS']['k_norm'][:, t_idx_arr]
    k_avg_norm = k_norm_arr.mean(axis = 1)
    
    return k_avg_norm.reshape(h5.attrs['shape'])

def compute_average_sig_norm(h5: h5py):

    t = h5['t'][:]
    T = h5.attrs['T']

    t_idx_arr = t >= (T - 1)        
        
    sig_norm_arr = h5['FS']['sig_norm'][:, t_idx_arr]
    sig_avg_norm = sig_norm_arr.mean(axis = 1)
        
    return sig_avg_norm.reshape(h5.attrs['shape'])


def compute_undulation_frequency(h5: h5py, Delta_t = 1.0):
    '''
    Compute average frequency of curvature wave
    '''

    # time
    t_arr = h5['t'][:]
    dt = t_arr[1] - t_arr[0]     
    # mesh
    N = h5['FS']['k'].shape[-1]
    s_arr = np.linspace(0, 1, N)
        
    # Crop first undulation cycle
    t_idx_arr = t_arr >= Delta_t
    t_arr = t_arr[t_idx_arr]
    # Only consider body centre    
    s0 = 0.1
    s1 = 0.9
    s_idx_arr = np.logical_and(s0 <= s_arr,  s_arr <= s1)     
    s_arr = s_arr[s_idx_arr]
    s_int_idx_arr = np.arange(len(s_idx_arr))[s_idx_arr]

    f_arr = np.fft.fftfreq(len(t_arr), dt)    
    f_idx_arr = np.argsort(f_arr)
    f_arr = f_arr[f_idx_arr]
    
    # Calculate frequency for all parameters a and b         
    f_avg_undu_arr = np.zeros(h5['FS']['k'].shape[0])
    f_std_undu_arr = np.zeros(h5['FS']['k'].shape[0])

    # Iterate of simulation
    for i in range(h5['FS']['k'].shape[0]):
        
        f_undu_s_arr = np.zeros(len(s_arr))

        # Iterate over body points        
        for j, idx in enumerate(s_int_idx_arr):
            
            k_arr = h5['FS']['k'][i, :, 0, idx]                            
            k_arr = k_arr[t_idx_arr]
            
            # Calculate Fourier power spectrum
            fft = np.fft.fft(k_arr)[f_idx_arr]
            ps = np.abs(fft)**2
            # Frequency = PS maximum
            idx = ps.argmax()            
            f_undu_s_arr[j] = np.abs(f_arr[idx])
        
        f_avg_undu_arr[i] = f_undu_s_arr.mean()        
        f_std_undu_arr[i] = f_undu_s_arr.std()        
           
    f_avg_undu_mat = f_avg_undu_arr.reshape(h5.attrs['shape'])
    f_std_undu_mat = f_std_undu_arr.reshape(h5.attrs['shape'])
                            
    return f_avg_undu_mat, f_std_undu_mat

def compute_time_lag(h5: h5py, Delta_t = 1.0):
    '''
    Compute average frequency of curvature wave
    '''
    
    # time
    t_arr = h5['t'][:]
    dt = t_arr[1] - t_arr[0]     
    # mesh
    N = h5['FS']['k'].shape[-1]
    s_arr = np.linspace(0, 1, N)
        
    # Crop first undulation cycle
    t_idx_arr = t_arr >= Delta_t
    t_arr = t_arr[t_idx_arr]
    # Only consider body centre    
    s0 = 0.1
    s1 = 0.9
    s_idx_arr = np.logical_and(s0 <= s_arr,  s_arr <= s1)     
    s_arr = s_arr[s_idx_arr]
    s_int_idx_arr = np.arange(len(s_idx_arr))[s_idx_arr]
            
    # Calculate frequency for all parameters a and b                 
    lag_avg_arr = np.zeros(h5['FS']['k'].shape[0])
    lag_std_arr = np.zeros(h5['FS']['k'].shape[0])
    
    lags_arr = dt * np.arange(-len(t_arr) + 1, len(t_arr))
        
    for i in range(h5['FS']['k'].shape[0]):
        
        lags_along_body_arr = np.zeros(len(s_arr))
        
        for j, idx in enumerate(s_int_idx_arr):
                        
            k_arr = h5['FS']['k'][i, :, 0, idx]
            k_arr = k_arr[t_idx_arr]

            k0_arr = h5['CS']['k0'][i, :, 0, idx]
            k0_arr = k0_arr[t_idx_arr]
                                                                           
            cc = correlate(k_arr, k0_arr, mode='full') 
            # max idx 
            max_idx = np.argmax(cc)
            # window size 
            ws = 6
                        
            # Extract data around cc maximum
            start_idx = max_idx - ws // 2
            end_idx = max_idx + ws // 2 

            # Create an quadratic interpolation function 
            interp_function = interp1d(
                lags_arr[start_idx:end_idx + 1],
                cc[start_idx:end_idx + 1],
                kind='quadratic'
            )
            
            lags_refined = np.linspace(lags_arr[start_idx], lags_arr[end_idx], 100)            
            cc_refined = interp_function(lags_refined)                                    
            lags_along_body_arr[j] = lags_refined[cc_refined.argmax()]
                
        lag_avg_arr[i] = lags_along_body_arr.mean()  
        lag_std_arr[i] = lags_along_body_arr.std()  
                               
    lag_avg_mat = lag_avg_arr.reshape(h5.attrs['shape'])
    lag_std_mat = lag_std_arr.reshape(h5.attrs['shape'])
    
    return lag_avg_mat, lag_std_mat

def compute_curvature_amplitude(h5: h5py):
    '''
    Computes curvature amplitude    
    '''    
    t = h5['t'][:]    
    T = h5.attrs['T']
    
    # For planar undulation, we only need to consider 
    # the first element of the curvature vector                    
        
    # mesh
    N = h5['FS']['k'].shape[-1]
    s_arr = np.linspace(0, 1, N)

    # Only look at last period
    t_idx_arr = t >= (T - 1)        
    # Only consider body centre    
    s0 = 0.1
    s1 = 0.9
    s_idx_arr = np.logical_and(s0 <= s_arr,  s_arr <= s1)     
    s_arr = s_arr[s_idx_arr]
    
    k_centre_arr = h5['FS']['k'][:, t_idx_arr, 0, :]
    k_centre_arr = k_centre_arr[:, :, s_idx_arr]
        
    # Max and min along time dimension
    k_max_arr = k_centre_arr.max(axis = 1)     

    # Mean and std along body dimnension
    A_avg_arr = k_max_arr.mean(axis = 1)
    A_std_arr = k_max_arr.std(axis = 1)
    
    A_avg_mat = A_avg_arr.reshape(h5.attrs['shape'])
    A_std_mat = A_std_arr.reshape(h5.attrs['shape'])
    
    return A_avg_mat, A_std_mat

def cluster_curvature_zero_crossings(k_all_mat, t_arr, s_arr):
    
    ds = s_arr[1] - s_arr[0]
        
    lam_mat = np.zeros((k_all_mat.shape[0], len(s_arr)))
    lam_avg_arr = np.zeros(k_all_mat.shape[0])
    lam_std_arr = np.zeros(k_all_mat.shape[0])
    
    tck_list = []

    zc_raw_list = []          
    zc_aligned_list = []
                      
    # Iterate over all simulations
    for i, k_mat in enumerate(k_all_mat):

        t_zc = []
        s_zc = []
                
        # Iterate over body points
        for s, k_arr in zip(s_arr, k_mat.T):

            # Find zero-crossings of curvature at current body point                         
            idx_arr = np.where(np.diff(np.sign(k_arr)) != 0)[0]                                
            
            # Refine zero-crossings
            for idx in idx_arr:
             
                k1,k2 = k_arr[idx], k_arr[idx+1]                 
                t1,t2 = t_arr[idx], t_arr[idx+1]
                
                m = (k2 - k1) / (t2 - t1)
                
                t_zc.append(-(k1 - m*t1) /m)
                s_zc.append(s)
                                                                        
        zc_raw_data = np.hstack((np.array(t_zc)[:, None], np.array(s_zc)[:, None]))
        zc_raw_list.append(zc_raw_data)
                
        dbscan = DBSCAN(eps=0.1, min_samples=3)
        clusters = dbscan.fit_predict(zc_raw_data)
        cluster_labels = set(clusters)

        cluster_list = []
        t_zc_cluster_mean = np.zeros(len(cluster_labels))

        # Get data in each cluster
        for j, label in enumerate(cluster_labels):            
            if j == -1:
                continue
            
            idx_arr = clusters == label
            cluster_data = zc_raw_data[idx_arr, :]                                     
            idx_arr = np.argsort(cluster_data[:, 0])            
            cluster_list.append(cluster_data[idx_arr])            
            
            t_zc_cluster_mean[j] = cluster_data[:, 0].mean()

        idx_arr = np.argsort(t_zc_cluster_mean)
        cluster_list = [cluster_list[idx] for idx in idx_arr]
        
        # Align clusters                                
        for j in range(len(cluster_list)-1):
            
            current_cluster = cluster_list[j]
            next_cluster = cluster_list[j+1]            

            dt_arr = []

            for l, s in enumerate(current_cluster[:, 1]):
                
                idx = np.where(s == next_cluster[:, 1])[0]                
                if len(idx) > 0:
                    dt_arr.append(next_cluster[idx, 0] - current_cluster[l, 0])
                else:
                    pass
                            
            dt_avg = np.mean(dt_arr)
            next_cluster[:, 0] -= dt_avg 
            
        
        zc_aligned = np.vstack(cluster_list)
        # Sort in ascending body coordinates
        idx_arr = np.argsort(zc_aligned[:, 1])
        zc_aligned = zc_aligned[idx_arr, :]   
        zc_aligned_list.append(zc_aligned)
               
        # Shift back in time to avoid confusion        
        dt = zc_raw_data[:, 0].min() - zc_aligned[:, 0].min()
        zc_aligned[:, 0] = zc_aligned[:, 0] + dt
                                                                                      
        # Fit bspline to zero-crossings 
        t_zc_arr = zc_aligned[:, 0]
        s_zc_arr = zc_aligned[:, 1]        

        # Thin duplicates 
        s_zc_arr_thin = [s_zc_arr[0]]
        t_zc_arr_thin = [t_zc_arr[0]]
    
        th = 0.5*ds
        
        for s_zc, t_zc in zip(s_zc_arr, t_zc_arr):    
            if np.abs(s_zc_arr_thin[-1] - s_zc) > th:     
                s_zc_arr_thin.append(s_zc)
                t_zc_arr_thin.append(t_zc)
                        
        tck  = splrep(s_zc_arr_thin, t_zc_arr_thin, k = 3, s=0.001)
                    
        tck_list.append(tck)
                     
        lam_arr = 1.0 / splev(s_arr, tck, der=1)
                
        #lam_arr[np.logical_or(lam_arr >= 1.6, lam_arr <= 0.0)] = np.nan                                
        lam_mat[i, :] = lam_arr
        anterior_lam_arr = lam_arr[s_arr <= 0.7]        
        lam_avg_arr[i] = anterior_lam_arr.mean()
        lam_std_arr[i] = anterior_lam_arr.std()

                                                                                                                                                           
    return tck_list, lam_mat, lam_avg_arr, lam_std_arr, zc_aligned_list, zc_raw_list #cut_off_idx_arr


def compute_undulation_wavelength(h5: h5py):
    '''
    Compute undulation wavelength
    '''    
      
    # Time
    t_arr = h5['t'][:]        
    T = h5.attrs['T']
    
    # mesh
    N = h5['FS']['k'].shape[-1]
    s_arr = np.linspace(0, 1, N)
        
    # Curvature
    k_mat = h5['FS']['k'][:, :, 0, :] 

    # Only analyse the last undulation cycle
    t_idx_arr = t_arr >= T - 1
    t_arr = t_arr[t_idx_arr]
    
    # Only analyse body centre    
    s0 = 0.1
    s1 = 0.9
    s_idx_arr = np.logical_and(s0 <= s_arr,  s_arr <= s1)     
    s_arr = s_arr[s_idx_arr]
        
    # Crop Curvature 
    k_centre_mat = k_mat[:, t_idx_arr, :]
    k_centre_mat = k_centre_mat[:, :, s_idx_arr]

    _, _, lam_avg_arr, lam_std_arr, _, _ = cluster_curvature_zero_crossings(k_centre_mat, t_arr, s_arr)

    lam_avg_mat = lam_avg_arr.reshape(h5.attrs['shape'])
    lam_std_mat = lam_std_arr.reshape(h5.attrs['shape'])
    
    return lam_avg_mat, lam_std_mat

def compute_swimming_speed(h5: h5py):
    '''
    Computes swimming speed for every simulation in h5
    '''
    
    T = h5.attrs['T']    
    U_arr = np.zeros(h5['FS']['r'].shape[0])    
    t = h5['t'][:]
        
    for i, r in enumerate(h5['FS']['r']):

        U_arr[i] = PostProcessor.comp_mean_swimming_speed(r, t, T-1)[0]
        
    return U_arr.reshape(h5.attrs['shape'])

def compute_wobbling_speed(h5: h5py):
    '''    
    :param h5:
    '''

    T = h5.attrs['T']    
    t = h5['t'][:]

    Y_avg_arr = np.zeros(h5['FS']['r'].shape[0])    
    Y_max_arr = np.zeros(h5['FS']['r'].shape[0])    
        
    for i, r in enumerate(h5['FS']['r']):

        Y_avg, Y_max, _ = PostProcessor.comp_amplitude_wobbling_speed(r, t, T-1)

        Y_avg_arr[i] = Y_avg 
        Y_max_arr[i] = Y_max
        
    return Y_avg_arr.reshape(h5.attrs['shape']), Y_max_arr.reshape(h5.attrs['shape'])
         
def compute_angle_attack(h5: h5py):
    '''
    Computes swimming speed for every simulation in h5
    '''
    
    T = h5.attrs['T']    
    t = h5['t'][:]

    psi_avg, psi_std, _ = PostProcessor.comp_angle_of_attack(h5['FS']['r'][0, :], t, T-1)
        
    psi_arr = np.zeros((h5['FS']['r'].shape[0], len(psi_avg)))    
    psi_std_arr = np.zeros_like(psi_arr)
        
    print(f'Shape={psi_arr.shape}')
        
    for i, r in enumerate(h5['FS']['r']):

        psi_avg, psi_std, _ = PostProcessor.comp_angle_of_attack(r, t, T-1)

        psi_arr[i, :] = psi_avg
        psi_std_arr[i, :] = psi_std
        
    print(f'PG shape type = {type(h5.attrs["shape"])}')
    print(f'PG shape={h5.attrs["shape"]}')
    print(f'Desired shape={h5.attrs["shape"] + (len(psi_avg),)}')
            
    return psi_arr.reshape(h5.attrs['shape'] + (len(psi_avg),)), psi_std_arr.reshape(h5.attrs['shape'] + (len(psi_avg),))
                         
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

                    
    


        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    
