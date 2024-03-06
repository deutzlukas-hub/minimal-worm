'''
Created on 5 Mar 2024

@author: amoghasiddhi
'''
from argparse import BooleanOptionalAction, Namespace


import numpy as np
from fenics import Constant, Expression

from minimal_worm.experiments import Experiment

class ActuationRelaxationExperiment(Experiment):
    
    
    @staticmethod
    def parameter_parser():
        
        parser = Experiment.parameter_parser()
            
        # Kinematic parameter
        parser.add_argument('--A', type = float, default = 4.0,
            help = 'Dimensionless curvature amplitude')
        parser.add_argument('--use_c', action = BooleanOptionalAction, default = False,
            help = 'If True, curvature amplitude A is determined from shape factor c')                       
        parser.add_argument('--c', type = float, default = 1.0,
            help= 'Shape factor c=A/q')                           
        parser.add_argument('--lam', type = float, default = 1.0,
            help = 'Dimensionless wavelength')
        parser.add_argument('--t_off', type = float, default = 2.5,
            help = 'Time when muscles is switched off')
        parser.add_argument('--tau_off', type = float, default = 0.1,
            help = 'Time scale on which muscles are switched off')
                                                                                                                                                                        
        return parser    

    @staticmethod
    def actuation_relaxation_control_sequence(param: dict):
        '''
        Initialize controls for contraction relation experiment
        
        :param parameter:
        '''                                                                                                                                    

        if isinstance(param, dict):
            param_dict = param
            param = Namespace()
            param.__dict__.update(param_dict)


        # Kinematic param
        lam = param.lam
        q = 2*np.pi / lam
                
        if not param.use_c:
            A = param.A            
        else:
            c = param.c
            A = c*q

        #kmax = Expression('A*sin(q*x[0])', degree=1, A=A, q=q)

        t = Constant(0.0)

        # On and off switch of muscles is modeled by sigmoids
        sm_on = ActuationRelaxationExperiment.muscle_on_switch(t, param)
        sm_off = ActuationRelaxationExperiment.muscle_off_switch(t, param)
                        
        # No muscles at head and tail, muscle torque gradually increases
        sh, st = Experiment.spatial_gmo(param)
                                                                       
        k = Expression(("sm_on*sm_off*sh*st*A*sin(q*x[0])", "0", "0"), 
            degree = 1, 
            t = t, 
            sh = sh, 
            st = st, 
            sm_on = sm_on, 
            sm_off = sm_off, 
            A = A,
            q = q)   
                  
        sig = Constant((0, 0, 0))    
    
        return {'k0': k, 'sig0': sig, 't': t}
