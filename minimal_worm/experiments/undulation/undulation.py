'''
Created on 14 Jun 2023

@author: amoghasiddhi
'''

#Third-party
from argparse import Namespace
import numpy as np
from fenics import Expression, Constant

#Local
from minimal_worm.experiments import Experiment
from argparse import BooleanOptionalAction

class UndulationExperiment(Experiment):
    '''
    Implements control sequences to model simple 2d undulation gait
    '''    
    
    @staticmethod
    def parameter_parser():
        
        parser = Experiment.parameter_parser()
            
        # Kinematic parameter
        parser.add_argument('--_A', type = bool, default = 4.0,
            help = 'Dimensionless curvature amplitude')
        parser.add_argument('--use_c', action = BooleanOptionalAction, default = False,
            help = 'If True, uses curvature amplitude wavenumber ratio c to determine amplitude _A')                       
        parser.add_argument('--c', type = float, default = 1.0,
            help= 'Curvature amplitude wavenumber ratio')                           
        parser.add_argument('--lam', type = float, default = 1.0,
            help = 'Dimensionless wavelength')
                                        
        return parser
    
    @staticmethod                                            
    def stw_control_sequence(param):
        '''
        Sinusoidal travelling wave control sequence

        :param worm (CosseratRod): worm object
        :param param (dict): param dictionary
        '''
        if isinstance(param, dict):
            param_dict = param
            param = Namespace()
            param.__dict__.update(param_dict)
           
        # Kinematic param
        lam = param.lam
        q = 2*np.pi / lam
                
        if not param.use_c:
            _A = param._A            
        else:
            c = param.c
            _A = c*q

        t = Constant(0.0)
                                    
        # Muscles switch on and off on a finite time scale                
        sm_on = UndulationExperiment.muscle_on_switch(t, param)
            
        # Gradual muscle activation onset at head and tale
        sh, st = UndulationExperiment.spatial_gmo(param)        
                                                                        
        k = Expression(("sm_on*sh*st*_A*sin(q*x[0] - 2*pi*t)", "0", "0"), 
            degree=1, t = t, _A = _A, q = q, sh = sh, st = st, sm_on = sm_on)   
                  
        sig = Constant((0, 0, 0))    
    
        return {'k': k, 'sig': sig, 't': t}
