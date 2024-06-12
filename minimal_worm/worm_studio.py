'''
Created on 18 Jan 2023

@author: lukas
'''

import time
from pathlib import PosixPath
from typing import Callable

import cv2
import ffmpeg
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid
from mayavi import mlab
from tvtk.tools import visual
import pickle
import multiprocessing as mp

from wormlab3d import logger
from wormlab3d.postures.natural_frame import NaturalFrame
from wormlab3d.postures.plot_utils import FrameArtistMLab
#from wormlab3d.trajectories.cache import get_trajectory
from wormlab3d.trajectories.util import smooth_trajectory

from simple_worm.frame import FrameSequenceNumpy

# Off-screen rendering
mlab.options.offscreen = True
plt.rcParams['font.family'] = 'Helvetica'

class WormStudio():
                            
    ARROW_OPT_DEFAULTS = {
        'opacity': 0.9,
        'radius_shaft': 0.025,
        'radius_cone': 0.05,
        'length_cone': 0.2}
    
    ARROW_SCALE = 0.1
                                           
    def __init__(self, 
            FS: FrameSequenceNumpy):
        
        self.X = np.swapaxes(FS.x, 1, 2)
        self.D1 = np.swapaxes(FS.e1, 1, 2)
        self.D2 = np.swapaxes(FS.e2, 1, 2)
        self.D3 = np.swapaxes(FS.e3, 1, 2)
        self.K = np.swapaxes(FS.Omega, 1, 2)

        self.n = self.X.shape[0] # number of time steps
        self.N = self.X.shape[1] # number of points along the centreline
        
        dt = FS.times[1] - FS.times[0] 
        self.fps = 1.0 / dt
        self.t = FS.times
        
        self.lengths = self._comp_worm_length_from_centreline()
        
    @staticmethod
    def _init_figure(width, height):
        # Set up mlab figure
        fig = mlab.figure(size=(width * 2, height * 2), bgcolor=(1, 1, 1))               
        
        # Depth peeling required for nice opacity, the rest don't seem to make any difference
        fig.scene.renderer.use_depth_peeling = True
        fig.scene.renderer.maximum_number_of_peels = 32
        fig.scene.render_window.point_smoothing = True
        fig.scene.render_window.line_smoothing = True
        fig.scene.render_window.polygon_smoothing = True
        fig.scene.render_window.multi_samples = 20
        fig.scene.anti_aliasing_frames = 20
        visual.set_viewer(fig)
                  
        return fig
    
    def _comp_centreline_midpoints(self):
        
        # Smooth the midpoints for nicer camera tracking
        mps = np.zeros((self.n, 3))
        for i, x in enumerate(self.X):
            mps[i] = x.min(axis=0) + np.ptp(x, axis=0) / 2
        #mps = smooth_trajectory(mps, window_len=51)

        return mps
    
    def _comp_worm_length_from_centreline(self
        ) -> np.ndarray:
           
        ds = 1.0 / (self.N - 1.0)                
        dX_ds = np.linalg.norm(np.gradient(self.X, ds, axis = 1), axis = 2)        
        lengths = trapezoid(dX_ds, dx = ds, axis = 1)
                
        return lengths
                        
    def _init_frame_artist(self, n_arrows 
        )-> FrameArtistMLab:
        
        # Set up the artist
        NF = NaturalFrame(self.X[0])
        
        NF.T, NF.M1, NF.M2 = self.D3[0], self.D1[0], self.D2[0]  
                    
        fa = FrameArtistMLab(NF,
            use_centred_midline=False,
            midline_opts={'opacity': 1, 'line_width': 8},
            surface_opts={'radius': 0.024 * self.lengths.mean()},
            arrow_opts= WormStudio.ARROW_OPT_DEFAULTS,
            arrow_scale = WormStudio.ARROW_SCALE,
            n_arrows = n_arrows)
        
        return fa
        
    def _make_3d_plot(self,
            add_trajectory,
            add_centreline,
            add_frame_vectors,
            add_surface,
            draw_e1: bool,
            draw_e2: bool,
            draw_e3: bool,
            n_arrows: int,
            fig_width: float, 
            fig_height: float,
            rel_camera_distance: float, 
            azim_offset: float,
            revolution_rate: float        
        ) -> Callable:
        
        # initialize figure
        fig = WormStudio._init_figure(fig_width, fig_height)        
        fa = self._init_frame_artist(n_arrows)

        mps = self._comp_centreline_midpoints()

        # Camera distance from focal point
        distance = rel_camera_distance * self.lengths.mean()

        if add_trajectory:
            pass
            # TODO: 
            # Render the trajectory with simple lines
            # path = mlab.plot3d(*X_trajectory.T, s, opacity=0.4, tube_radius=None, line_width=8)
            # path.module_manager.scalar_lut_manager.lut.table = cmaplist

        if add_centreline:
            fa.add_midline(fig)
        
        if add_frame_vectors:
            fa.add_component_vectors(fig, 
                draw_e1 = draw_e1,
                draw_e2 = draw_e2,
                draw_e3 = draw_e3)
        
        if add_surface:
            k = np.sqrt(np.sum(self.D1**2 + self.D2**2, axis = 2))             
            fa.add_surface(fig, v_min=-k.max(), v_max=k.max())    

        # Aspects
        n_revolutions = self.n / self.fps / 60 * revolution_rate
        azims = azim_offset + np.linspace(start=0, stop=360 * n_revolutions, num=self.n)
        mlab.view(figure=fig, azimuth=azims[0], distance=distance, focalpoint=mps[0])

        def update(frame_idx: int):
            fig.scene.disable_render = True
            NF = NaturalFrame(self.X[frame_idx])
            NF.T, NF.M1, NF.M2 = self.D3[frame_idx], self.D1[frame_idx], self.D2[frame_idx]                          
                                                    
            fa.update(NF)
            
            fig.scene.disable_render = False
            mlab.view(figure=fig, azimuth=azims[frame_idx], distance=distance, focalpoint=mps[frame_idx])
            fig.scene.render()
            
            plot_3d = mlab.screenshot(mode='rgb', antialiased=True, figure=fig)
            plot_3d = cv2.resize(plot_3d, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                                    
            return plot_3d

        return update
        
    def generate_clip(self,
            output_path: PosixPath,
            add_trajectory = True,
            add_centreline = True,
            add_frame_vectors = True,            
            add_surface = True,
            draw_e1 = True,
            draw_e2 = True,
            draw_e3 = True,
            n_arrows = 0,            
            fig_width = 1200,
            fig_height = 900,
            rel_camera_distance = 2.0,
            azim_offset = 0.0,
            revolution_rate = 1 / 3,
            T_max = None):
        """
        Generate a basic exemplar video showing a rotating 3D worm along a trajectory
        and camera images with overlaid 2D midline reprojections.
        """

        logger.info('Building 3D plot.')
                        
        update_fn = self._make_3d_plot(
            add_trajectory, 
            add_centreline, 
            add_frame_vectors,             
            add_surface, 
            draw_e1,
            draw_e2,
            draw_e3,
            n_arrows,
            fig_width, 
            fig_height, 
            rel_camera_distance, 
            azim_offset, 
            revolution_rate)
            
        # Initialise ffmpeg process
        output_args = {
            'pix_fmt': 'yuv444p',
            'vcodec': 'libx264',
            'r': self.fps,
            #'metadata:g:1': 'artist=Leeds Wormlab',
            'metadata:g:2': f'year={time.strftime("%Y")}'}
        
        process = (
            ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{fig_width}x{fig_height}')
                .output(str(output_path) + '.mp4', **output_args)
                .overwrite_output()
                .run_async(pipe_stdin=True)
        )
    
        logger.info('Rendering frames.')
                        
        if T_max is not None:
            n = np.sum(self.t <= T_max)
        else:
            n = self.n
        
        for i in range(n):
            if i > 0 and i % 50 == 0:
                logger.info(f'Rendering frame {i + 1}/{n}.')
    
            # Update the frame and write to stream
            frame = update_fn(i)
            process.stdin.write(frame.tobytes())
    
        # Flush video
        process.stdin.close()
        process.wait()
    
        logger.info(f'Generated video.')        
        
        return
    
    @staticmethod
    def _init_worker(#worker_counter, 
        global_data_dir,
        global_output_dir,
        global_overwrite, 
        #init_args, 
        init_kwargs):
    
        global data_dir
        data_dir = global_data_dir
        
        global output_dir
        output_dir = global_output_dir
        
        global overwrite
        overwrite = global_overwrite
        
        global kwargs
        kwargs = init_kwargs
    
        # global worker_number         
        # worker_number = worker_counter.value 
        # worker_counter.value += 1
    
        return
#------------------------------------------------------------------------------ 
# Static methods for parallelized clip generation

    @staticmethod
    def wrap_generate_clip(input_tup):
        '''
        Wrapes generate clip function so that it can be used with 
        the multiprocessing package. 
            
        :param _input (tuple): Parameter dictionary and hash
        :param create_CS (function): Creates control sequence from parameter
        :param pbar (tqdm.tqdm): Progressbar
        :param logger (logging.Logger): Progress logger
        :param task_number (int): Number of tasks
        :param output_dir (str): Result directory
        :param overwrite (bool): If true, exisiting files are overwritten
        :param save_keys (list): List of attributes which will be saved to the result file. 
            If None, then all attributes get saved.        
        '''    
        # Access the global variables which 
        # have been instantiated when the 
        # worker have initiated
        global data_dir
        global output_dir
        global overwrite                 
        global kwargs
                
        # Unpack input and load FS
        FS_name, video_name = input_tup
                
        sim_data = pickle.load(open(str(data_dir / 'simulations' / FS_name), 'rb'))                    
        video_path = output_dir / video_name  
        
        if not overwrite:    
            # If video file exists, abort task
            if video_path.exists():
                #logger.info(f'Task {task_number} aborted: video already exists')                                    
                return
    
        #logger.info(f'Task {task_number}: Start rendering video')                                                            
        WS = WormStudio(sim_data['FS'])
        WS.generate_clip(video_path, **kwargs)
                                
        #logger.info(f'Task {task_number}: Saved video to {video_path}.')         
                
        return 
    
    @staticmethod
    def generate_clips_in_parallel(N_worker, 
            FS_names, 
            video_names,
            data_dir, 
            output_dir, 
            overwrite = False,
            **kwargs):
                
        pool = mp.Pool(N_worker, WormStudio._init_worker, initargs = (data_dir, output_dir, overwrite, kwargs))
        
        pool.map(WormStudio.wrap_generate_clip, zip(FS_names, video_names))
        
        return
    
    
#------------------------------------------------------------------------------ 
# Static method to generate clips for every parameter grid
    
    @staticmethod    
    def generate_worm_clips_from_PG(
            PG,
            filenames, 
            output_dir,
            data_dir,
            **kwargs):
        '''
        Generates a clip for every FrameSequence in parameter grid
                
        :param PG (ParameterGrid): Parameter gird object
        :param filenames (List[str]): video filenames
        :param output_dir (PoxisPath): output directory
        :param data_dir (PoxisPath): data directory with simulation files
        :param kwargs (dict): Additional keyword arguments passed to 
            WormStudio.generate_clip
        '''
                                                           
        output_dir.mkdir(parents=True, exist_ok = True)
         
        for h, fn in zip(PG.hash_arr, filenames):
            
            FS_name = h + '.dat'              
            sim_data = pickle.load(open(str(data_dir / FS_name), 'rb'))                    

            WS = WormStudio(sim_data['FS'])                                    
            WS.generate_clip(output_dir / fn, **kwargs)
                             
        return    
    