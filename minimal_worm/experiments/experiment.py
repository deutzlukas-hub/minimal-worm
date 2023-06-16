'''
Created on 13 Jun 2023

@author: amoghasiddhi
'''

# Build-in imports
from os.path import isfile, join
from abc import ABC 
import pickle
from argparse import BooleanOptionalAction

#Third party
from fenics import Expression
import numpy as np

# Local imports
from minimal_worm import Worm 
from minimal_worm import ModelParameter, parameter_parser
from mp_progress_logger import FWException
            
class Experiment(ABC):      
    
    @staticmethod    
    def parameter_parser():
        
        parser = parameter_parser()
        
        # Muscle parameter 
        parser.add_argument('--gsmo', action = BooleanOptionalAction, default = True,
            help = 'If true, muscles have a gradual spatial onset at the head and tale')
        parser.add_argument('--Ds_h', type = bool, default = 0.01,
            help = 'Sigmoid slope at the head')
        parser.add_argument('--Ds_t', type = bool, default = 0.01,
            help = 'Sigmoid slope at the tale')    
        parser.add_argument('--s0_h', type = bool, default = 3*0.01,
            help = 'Sigmoid midpoint at the head')
        parser.add_argument('--s0_t', type = bool, default = 1-3*0.01,
            help = 'Sigmoid midpoint at the tale')

        # Muscle timescale
        parser.add_argument('--fmts', type=bool, default=True,
            help='If true, muscles switch on on a finite time scale')
        parser.add_argument('--tau_on', type=float, default = 0.1,
            help='Muscle time scale')
        parser.add_argument('--t_on', type=float, default = 5*0.1,
            help='Sigmoid midpoint')
      
        return parser
      
    @staticmethod   
    def sig_m_on(t0, tau):
        
        return lambda t: 1.0 / (1 + np.exp(-( t - t0 ) / tau))
    
    @staticmethod   
    def sig_m_off(t0, tau):

        return lambda t: 1.0 / (1 + np.exp( ( t - t0) / tau))

    @staticmethod
    def sig_m_on_expr(t, t0, tau):

        return Expression('1 / ( 1 + exp(- ( t - t0) / tau))', 
            degree = 1, 
            t = t, 
            t0 = t0, 
            tau = tau)

    @staticmethod
    def sig_m_off_expr(t, t0, tau):

        return Expression('1 / ( 1 + exp( ( t - t0) / tau))', 
            degree = 1, 
            t = t, 
            t0 = t0, 
            tau = tau)
        
    @staticmethod
    def sig_head_expr(Ds, s0):
        
        return Expression('1 / ( 1 + exp(- (x[0] - s0) / Ds ) )', 
            degree = 1, 
            Ds = Ds, 
            s0 = s0)
    
    @staticmethod
    def sig_tale_expr(Ds, s0):
            
        return Expression('1 / ( 1 + exp(  (x[0] - s0) / Ds) )', 
            degree = 1, 
            Ds = Ds, 
            s0 = s0) 

    @staticmethod        
    def spatial_gmo(param):        
        # Gradual onset of muscle activation at head and tale
        if param.gsmo:                            
            Ds_h, s0_h = param.Ds_h, param.s0_h
            Ds_t, s0_t = param.Ds_t, param.s0_t
            sh = Experiment.sig_head_expr(Ds_h, s0_h)
            st = Experiment.sig_tale_expr(Ds_t, s0_t)
        else: 
            sh = Expression('1', degree = 1)
            st = Expression('1', degree = 1)
        return sh, st
    
    @staticmethod
    def muscle_on_switch(t, param):        
        # Muscle switch on at finite timescale
        if param.fmts:        
            t_on, tau_on  = param.t_on, param.tau_on
            sm_on = Experiment.sig_m_on_expr(t, t_on, tau_on)
        else:
            sm_on = Expression('1', degree = 1)
        
        return sm_on

    @staticmethod
    def muscle_off_switch(t, param):
        # Muscle switch off on a finite timescale
        if param.fmts:        
            t_off, tau_off  = param.t_off, param.tau_off
            sm_off = Experiment.sig_m_off_expr(t, t_off, tau_off)
        else:
            sm_off = Expression('1', degree = 1)
        
        return sm_off

                                                  
def simulate_experiment(worm,
                        param,
                        CS,
                        F0 = None,
                        pbar = None,
                        logger = None,
                        ):            
    '''
    Simulate experiments defined by control sequence
            
    :param worm (simple_worm.CosseratRod): Worm
    :param param (dict): Parameter
    :param pbar (tqdm.tqdm): Progressbar
    :param logger (logging.Logger): Progress logger
    :param F0 (simple_worm.FrameSequence): Initial frame
    '''

    MP = ModelParameter(param)
                        
    FS, CS, e = worm.solve(param.T, MP, CS, F0, pbar=pbar, 
        logger=logger, dt_report=param.dt_report, N_report=param.N_report) 
                              
    return FS, CS, MP, e


        
        


