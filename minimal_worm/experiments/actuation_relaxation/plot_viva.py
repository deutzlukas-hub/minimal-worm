'''
Created on 7 Mar 2024

@author: amoghasiddhi
'''
# Built-in
from pathlib import Path
# Third-party
import h5py
import numpy as np
import matplotlib.pyplot as plt
# My own packages
from parameter_scan import ParameterGrid
# minimal-worm
from dirs import create_storage_dir
from minimal_worm.plot import plot_scalar_field, plot_multiple_scalar_fields

# Get data from external storage harddrive
if True:
    log_dir, sim_dir, sweep_dir = create_storage_dir()

video_storage = Path('../../../results/videos/actuation_relaxation')
video_storage.mkdir(parents=True, exist_ok=True)

def load_data(filename):
    '''
    Loads hdf5 simulation file
    '''

    h5_filepath = sweep_dir / filename
    h5 = h5py.File(str(h5_filepath), 'r')    
    PG_filepath = log_dir / h5.attrs['grid_filename']
    PG = ParameterGrid.init_pg_from_filepath(str(PG_filepath))

    return h5, PG

def actuation_relaxation_curvature_videos():

    h5_filename = Path('raw_data_a=0.034_b=0.01_c_min=0.5_c_max=1.5_c_step=0.25_lam_min=0.5_lam_max=2.0_lam_step=0.25_N=750_dt=0.01_T=5.0.h5')    

    h5, PG = load_data(h5_filename) 

    c_arr = PG.v_from_key('c')
    lam_arr = PG.v_from_key('lam')
    
    t = h5['t'][:] 
    shape = h5['FS']['r'].shape

    video_dir = video_storage / 'actuation_relaxation' / h5_filename.stem
    video_dir.mkdir(parents=True, exist_ok = True)
    
    for i, c in enumerate(c_arr):
        for j, lam in enumerate(lam_arr):
            
            k = np.ravel_multi_index((i,j), (len(c_arr), len(lam_arr)))
            
            k = h5['FS']['k'][k, :, 0, :]
            k0 = h5['CS']['k0'][k, :, 0, :]
            
            gs = plt.GridSpec(2, 1)
            ax00 = plt.subplot(gs[0])
            ax01 = plt.subplot(gs[1])
            
            plot_scalar_field(ax00, k0, extent = [0.0, 5.0, 0.0, 1.0], cmap = 'seismic')
            plot_scalar_field(ax01, k, extent = [0.0, 5.0, 0.0, 1.0], cmap = 'seismic')
            
            plt.savefig(video_dir / f'c={c}_lam={lam}.png')
            
    return

if __name__ == '__main__':
    
    pass
    

