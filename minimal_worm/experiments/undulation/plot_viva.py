'''
Created on 8 Mar 2024

@author: amoghasiddhi
'''

# Built-in
from pathlib import Path
from typing import Optional, Tuple
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

video_storage = Path('../results/videos/undulation')
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

def make_preferred_curvature_video(
        filename: str, 
        video_dir: Path,
        t: np.ndarray,
        k0: np.ndarray, 
        k: np.ndarray,
        vlim: Optional[Tuple[float, float]]):
    
    fig = plt.figure()
    gs = plt.GridSpec(2, 1)
    ax00 = plt.subplot(gs[0])
    ax01 = plt.subplot(gs[1])
        
    dt = t[1] - t[0]
    fps = 1 // dt
          
    cbar1, _ = plot_scalar_field(ax00, k0, v_lim = vlim, extent = [0.0, 5.0, 0.0, 1.0], cmap = 'seismic')
    cbar2, _ = plot_scalar_field(ax01, k, v_lim = vlim, extent = [0.0, 5.0, 0.0, 1.0], cmap = 'seismic')

    cbar1.set_label(r'$\kappa_0$', fontsize = 12)
    cbar2.set_label(r'$\kappa$', fontsize = 12)
        
    ax00.set_ylabel('s', fontsize = 12)
    ax01.set_ylabel('s', fontsize = 12)
        
    ax00.set_yticks([0.0, 0.5, 1.0])
    ax01.set_yticks([0.0, 0.5, 1.0])
                
    ax01.set_xlabel('t', fontsize = 12)
        #ax00.set_xticks([0.0, 2.5, T])
        #ax01.set_xticks([0.0, 2.5, T])

    vertical_line_1, = ax00.plot([], [], color='k', linestyle='--')
    vertical_line_2, = ax01.plot([], [], color='k', linestyle='--')
        
    plt.savefig(filename + '.pdf')

    # Animation function
    def animate(t):
        # Set the position of the vertical line
        vertical_line_1.set_data([t, t], [0, 1])
        vertical_line_2.set_data([t, t], [0, 1])
        
        return [vertical_line_1, vertical_line_2]
    
    # Create the animation
    ani = FuncAnimation(fig, animate, frames=t, interval=1000/fps)    
    ani.save(video_dir / (filename + '.mp4'), fps=fps)
    plt.close()

    return
    
def animate_curvature_chemograms():

    h5_filename = Path('raw_data_a=0.034_b=0.01_c_min=0.5_c_max=2.0_c_step=0.25_lam_min=0.5_lam_max=2.0_lam_step=0.25_N=750_dt=0.01_T=10.0.h5')    

    h5, PG = load_data(h5_filename) 

    c0_arr = PG.v_from_key('c0')
    lam0_arr = PG.v_from_key('lam0')
    
    t = h5['t'][:] 
    T = PG.base_parameter['T'].magnitude
    
    video_dir = video_storage / 'undulation' / h5_filename.stem
    video_dir.mkdir(parents=True, exist_ok = True)
    
    print(video_dir.resolve())
    
    for i, c0 in enumerate(c0_arr):
        for j, lam0 in enumerate(lam0_arr):
            
            l = np.ravel_multi_index((i,j), (len(c0_arr), len(lam0_arr)))

            A0 = 2*np.pi*c0/lam0 
            vlim = (-A0, A0)
            
            k = h5['FS']['k'][l, :, 0, :]
            k0 = h5['CS']['k0'][l, :, 0, :]
            
            vlim = (-A0, A0) 
            
            print(l)
    
            fig = plt.figure()
            gs = plt.GridSpec(2, 1)
            ax00 = plt.subplot(gs[0])
            ax01 = plt.subplot(gs[1])
        
            plot_scalar_field(ax00, k0, extent = [0.0, T, 0.0, 1.0], cmap = 'seismic')
            plot_scalar_field(ax01, k, extent = [0.0, T, 0.0, 1.0], cmap = 'seismic')

            vertical_line_1, = ax00.plot([], [], color='k', linestyle='--')
            vertical_line_2, = ax01.plot([], [], color='k', linestyle='--')

            cbar1, _ = plot_scalar_field(ax00, 
                k0, 
                v_lim = vlim, 
                extent = [0.0, T, 0.0, 1.0], 
                cmap = 'seismic')
            cbar2, _ = plot_scalar_field(ax01, k, extent = [0.0, T, 0.0, 1.0], cmap = 'seismic')

            cbar1.set_label(r'$\kappa_0$', fontsize = 12)
            cbar2.set_label(r'$\kappa$', fontsize = 12)
            
            ax00.set_ylabel('s', fontsize = 12)
            ax01.set_ylabel('s', fontsize = 12)
            
            ax00.set_yticks([0.0, 0.5, 1.0])
            ax01.set_yticks([0.0, 0.5, 1.0])
                    
            ax01.set_xlabel('t', fontsize = 12)


            plt.savefig(video_dir / f'c0={c0}_lam0={lam0}.png')

            # Animation function
            def animate(t):
                # Set the position of the vertical line
                vertical_line_1.set_data([t, t], [0, 1])
                vertical_line_2.set_data([t, t], [0, 1])
                
                
                return [vertical_line_1, vertical_line_2]
            
            # Create the animation
            ani = FuncAnimation(fig, animate, frames=t, interval=1000/fps)    
            ani.save(video_dir / f'c0={c0}_lam0={lam0}.mp4', fps=fps)
            plt.close()
                
    return
    
def plot_showcase_regimes():
    
    h5_filename  = Path('raw_data_a_min=-2_a_max=3_a_step=0.2_'
        'b_min=-3_b_max=0_b_step=0.2_'
        'A=4.0_lam=1.0_T=5.0_N=750_dt=0.001.h5')

    h5, PG = load_data(h5_filename) 

    a_arr, b_arr = PG.v_from_key('a'), PG.v_from_key('b')
    
    t = h5['t'][:]
    T = PG.base_parameter['T']    
    fps = int(1.0 / PG.base_parameter['dt'])

    # Regimes
    a1, b1 = 0.4, 0.005
    a2, b2 = 45, 0.1
    a3, b3 = 2200, 0.4
                        
    video_dir = video_storage / h5_filename.stem
    video_dir.mkdir(parents=True, exist_ok = True)
    
    c0 = PG.base_parameter['c']
    lam0 = PG.base_parameter['lam']
    A0 = 2 * np.pi * c0 / lam0 
    
    vlim = (-A0, A0)

    print(video_dir.resolve())

    for n, (ai, bi) in enumerate(zip([a1, a2, a3], [b1, b2, b3])):
        
        print(f'{3 - n} more videos to go!')
        
        i = np.abs(a_arr - ai).argmin()
        j = np.abs(b_arr - bi).argmin()
                                
        l = np.ravel_multi_index((i,j), (len(a_arr), len(b_arr)))

        k = h5['FS']['k'][l, :, 0, :]
        k0 = h5['CS']['k0'][l, :, 0, :]
        
        filename = f'regime_{n+1}_a={ai}_b_{bi}' 
                
        make_preferred_curvature_video(filename, video_dir, t, k0, k, vlim)
                                
    return

        
if __name__ == '__main__':
    
    # undulation_curvature_videos()
    # plot_showcase_regimes()    
    
    animate_curvature_chemograms()
    
    
    

