'''
Created on 8 Mar 2024

@author: amoghasiddhi
'''

# Built-in
from pathlib import Path
# Third-party
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# My own packages
from parameter_scan import ParameterGrid
# minimal-worm
from dirs import create_storage_dir
from minimal_worm.plot import plot_scalar_field, plot_multiple_scalar_fields

# Get data from external storage harddrive
if True:
    log_dir, sim_dir, sweep_dir = create_storage_dir()

video_storage = Path('../../results/videos/undulation')
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

def undulation_curvature_videos():

    h5_filename = Path('raw_data_a=0.034_b=0.01_c_min=0.5_c_max=1.5_c_step=0.25_lam_min=0.5_lam_max=2.0_lam_step=0.25_N=750_dt=0.01_T=5.0.h5')    

    h5, PG = load_data(h5_filename) 

    c0_arr = PG.v_from_key('c0')
    lam0_arr = PG.v_from_key('lam0')
    
    t = h5['t'][:] 

    video_dir = video_storage / 'undulation' / h5_filename.stem
    video_dir.mkdir(parents=True, exist_ok = True)
    
    for i, c0 in enumerate(c0_arr):
        for j, lam0 in enumerate(lam0_arr):
            
            k = np.ravel_multi_index((i,j), (len(c0_arr), len(lam0_arr)))
            
            k = h5['FS']['k'][k, :, 0, :]
            k0 = h5['CS']['k0'][k, :, 0, :]
            
            times = h5['t']
            
            fig = plt.figure()
            gs = plt.GridSpec(2, 1)
            ax00 = plt.subplot(gs[0])
            ax01 = plt.subplot(gs[1])
            
            plot_scalar_field(ax00, k0, extent = [0.0, 5.0, 0.0, 1.0], cmap = 'seismic')
            plot_scalar_field(ax01, k, extent = [0.0, 5.0, 0.0, 1.0], cmap = 'seismic')

            vertical_line_1, = ax00.plot([], [], color='red', linestyle='--')
            vertical_line_2, = ax01.plot([], [], color='red', linestyle='--')
            
            plt.savefig(video_dir / f'c0={c0}_lam0={lam0}.png')

            # Animation function
            def animate(t):
                # Set the position of the vertical line
                vertical_line_1.set_data([t, t], [-1, 1])
                vertical_line_2.set_data([t, t], [-1, 1])
                
                
                return [vertical_line_1, vertical_line_2]
            
            # Create the animation
            ani = FuncAnimation(fig, animate, frames=times, interval=100)
            
            plt.show()

            
    return

