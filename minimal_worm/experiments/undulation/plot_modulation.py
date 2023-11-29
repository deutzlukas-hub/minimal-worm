'''
Created on 28 Nov 2023

@author: amoghasiddhi
'''
import h5py 
import numpy as np
import matplotlib.pyplot as plt

# My packages
from parameter_scan import ParameterGrid

# Local import
from minimal_worm.experiments.undulation.dirs import create_storage_dir, fig_dir
from minimal_worm.experiments import PostProcessor

log_dir, sim_dir, sweep_dir = create_storage_dir()

def load_data(filename):
    '''
    Loads hdf5 simulation file
    '''

    h5_filepath = sweep_dir / filename
    h5 = h5py.File(str(h5_filepath), 'r')    
    PG_filepath = log_dir / h5.attrs['grid_filename']
    PG = ParameterGrid.init_pg_from_filepath(str(PG_filepath))


    return h5, PG

def plot_wobbling():

    h5_fn = ('raw_data_a=1.0_b=0.0032_'
        'c_min=0.4_c_max=1.6_c_step=0.2_'
        'lam_min=0.5_lam_max=2.0_lam_step=0.2_'
        'N=250_dt=0.01_T=5.0_pic_on=False.h5'
    )

        
    h5, PG = load_data(h5_fn)
    
    lam0_arr = PG.v_from_key('lam')
    c0_arr = PG.v_from_key('lam')
    
    r = h5['FS']['r']
    t = h5['t'][:]
        
    plot_dir = fig_dir / 'modulation' / 'wobbling'    
    plot_dir.mkdir(exist_ok = True, parents = True)       
    
    for k, r in enumerate(r):
    
        print(f'plot={k}')
    
        i, j = np.unravel_index(k, PG.shape)
        
        # Centre of mass trajectory
        r_com = r.mean(axis = -1)
        # Time average
        r_com_avg = r_com.mean(axis = 0)
                                  
        S = np.linalg.norm(r_com[-1, :] - r_com[0, :]) 
        eS, eW = PostProcessor.comp_swimming_direction(r_com)

        r_com_0 = r_com_avg - 0.5 * S * eS 
        r_com_1 = r_com_avg + 0.5 * S * eS 

        r_com_2 = r_com_avg - 0.1 * S * eW 
        r_com_3 = r_com_avg + 0.1 * S * eW 
                      
        plt.suptitle(f'lam0={lam0_arr[i]}, c0={c0_arr[j]}')              
        plt.plot(r_com[:, 1], r_com[:, 2], '-', c='k')
        plt.plot(r_com_avg[1], r_com_avg[2], 'o', c='r') 
        plt.plot([r_com_0[1], r_com_1[1]] , [r_com_0[2], r_com_1[2]], '--', c='r') 
        plt.plot([r_com_2[1], r_com_3[1]] , [r_com_2[2], r_com_3[2]], '--', c='r') 
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
                
        plt.savefig(plot_dir / f'{str(k).zfill(3)}.png')                
        plt.close(plt.gcf())
            
    return

def plot_propulsive_force():
        
    h5_fn = ('raw_data_a=1.0_b=0.0032_'
        'c_min=0.4_c_max=1.6_c_step=0.2_'
        'lam_min=0.5_lam_max=2.0_lam_step=0.2_'
        'N=250_dt=0.01_T=5.0_pic_on=False.h5'
    )
    
    h5, PG = load_data(h5_fn)
        
    lam0_arr = PG.v_from_key('lam')
    c0_arr = PG.v_from_key('c')
    
    r_arr = h5['FS']['r'][:]
    f_F_arr = h5['FS']['f_F'][:]
    t = h5['t'][:]
    dt = t[1] - t[0]
    T = h5.attrs['T']
                
    plot_dir = fig_dir / 'modulation' / 'wobbling'    
    plot_dir.mkdir(exist_ok = True, parents = True)
               
    Fp_arr = np.zeros(r_arr.shape[0])

    plot_dir = fig_dir / 'modulation' / 'propulsion_force'    
    plot_dir.mkdir(exist_ok = True, parents = True)
               
    for k, (r, f_F) in enumerate(zip(r_arr, f_F_arr)):
        
        print(f'plot={k}')
        
        i,j = np.unravel_index(k, PG.shape)
        lam0 = lam0_arr[j]
        c0 = c0_arr[i]
                 
        Fp, _, t_crop = PostProcessor.comp_propulsive_force(f_F, r, t, Delta_t = T-1)
        plt.plot(t_crop, Fp)
        plt.xlabel(r'$t$')
        plt.ylabel(r'$f_{\mathrm{p}}$')
        plt.suptitle(fr'$\lambda={lam0}$, $c={c0}$')
        
        plt.savefig(plot_dir / f'{str(k).zfill(3)}.png')                
        plt.close(plt.gcf())
          
        Fp_arr[k] = np.trapz(Fp, dx=dt)
                            
    Fp_arr = Fp_arr.reshape(PG.shape)
    
    CS = plt.contourf(lam0_arr, c0_arr, Fp_arr, cmap = 'inferno')    
    plt.contour(lam0_arr, c0_arr, Fp_arr, c='k')
    plt.colorbar(CS)    
    plt.show()
                         
    return


def plot_swimming_speed():
        
    h5_fn = ('raw_data_a=1.0_b=0.0032_'
        'c_min=0.4_c_max=1.6_c_step=0.2_'
        'lam_min=0.5_lam_max=2.0_lam_step=0.2_'
        'N=250_dt=0.01_T=5.0_pic_on=False.h5'
    )
    
    h5, PG = load_data(h5_fn)
        
    lam0_arr = PG.v_from_key('lam')
    c0_arr = PG.v_from_key('c')
    
    r_arr = h5['FS']['r'][:]
    f_F_arr = h5['FS']['f_F'][:]
    t = h5['t'][:]
    T = h5.attrs['T']
                
    plot_dir = fig_dir / 'modulation' / 'wobbling'    
    plot_dir.mkdir(exist_ok = True, parents = True)
               
    fp_arr = np.zeros(r_arr.shape[0])

    plot_dir = fig_dir / 'modulation' / 'propulsion_force'    
    plot_dir.mkdir(exist_ok = True, parents = True)
               
    for k, (r, f_F) in enumerate(zip(r_arr, f_F_arr)):
        
        print(f'plot={k}')
        
        i,j = np.unravel_index(k, PG.shape)
        lam0 = lam0_arr[j]
        c0 = c0_arr[i]
                 
        avg_fp, _, t_crop = PostProcessor.comp_propulsive_force(f_F, r, t, Delta_t = T-1)
        # plt.plot(t_crop, avg_fp)
        # plt.xlabel(r'$t$')
        # plt.ylabel(r'$f_{\mathrm{p}}$')
        # plt.suptitle(fr'$\lambda={lam0}$, $c={c0}$')
        #
        # plt.savefig(plot_dir / f'{str(k).zfill(3)}.png')                
        # plt.close(plt.gcf())
          
        fp_arr[k] = avg_fp.mean()
                            
    fp_arr = fp_arr.reshape(PG.shape)
    
    CS = plt.contourf(lam0_arr, c0_arr, fp_arr, cmap = 'inferno')    
    plt.contour(lam0_arr, c0_arr, fp_arr, c='k')
    plt.colorbar(CS)    
    plt.show()
                         
    return



if __name__ == '__main__':
    
    #plot_wobbling() 
    plot_propulsive_force()    
    
    

