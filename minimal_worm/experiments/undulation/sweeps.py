'''
Created on 15 Jun 2023

@author: amoghasiddhi
'''
# Built-in
from pathlib import Path
from sys import argv
from argparse import ArgumentParser, BooleanOptionalAction
#from decimal import Decimal

# Third-party
from parameter_scan import ParameterGrid
import numpy as np
from scipy.optimize import curve_fit
import pint

# Local imports
from minimal_worm import FRAME_KEYS
from minimal_worm.worm import CONTROL_KEYS
from minimal_worm.experiments import Sweeper
from minimal_worm.experiments.undulation import UndulationExperiment
from minimal_worm.experiments.undulation import create_storage_dir
from minimal_worm.experiments.undulation.analyse_sweeps import analyse

ureg = pint.UnitRegistry()

def fang_yen_data():
    '''
    Undulation wavelength, amplitude, frequencies for different viscosities
    from Fang-Yen 2010 paper    
    '''
    # Experimental Fang Yeng 2010
    mu_arr = 10**(np.array([0.000, 0.966, 2.085, 2.482, 2.902, 3.142, 3.955, 4.448])-3) # Pa*s            
    lam_arr = np.array([1.516, 1.388, 1.328, 1.239, 1.032, 0.943, 0.856, 0.799])        
    f_arr = [1.761, 1.597, 1.383, 1.119, 0.790, 0.650, 0.257, 0.169] # Hz
    A_arr = [2.872, 3.126, 3.290, 3.535, 4.772, 4.817, 6.226, 6.735]
    
    return mu_arr, lam_arr, f_arr, A_arr

def david_gagnon():
    '''
    Undulation speed, frequncies for different viscosities from 
    '''

    mu_arr_U = np.array([1.02, 2.02, 3.68, 12.27, 151.89, 218.18, 309.68, 434.62, 645.85])
    U_arr = np.array([0.38, 0.3, 0.37, 0.31, 0.27, 0.25, 0.23, 0.22, 0.18])

    mu_arr_f = np.array([ 1., 1.98, 3.62, 12.08, 147.73, 216.56, 310.18, 434.09, 643.76])
    f_arr = np.array([1.84, 1.47, 2.08, 1.43, 1.4, 1.3, 1.39, 1.16, 1.05])

    mu_arr_c = np.array([1., 1.96, 3.58, 11.94, 146.03, 214.06, 306.6, 429.09, 643.76 ])
    c_arr = np.array([3.77, 3.45, 4.39, 3.05, 2.22, 2.06, 2.06, 1.86, 1.72])

    mu_arr_A = np.array([1.01, 2., 3.65, 12.26, 148.32, 217.17, 310.38,433.76, 642.42])    
    A_arr = np.array([0.26, 0.31, 0.33, 0.29, 0.24, 0.24, 0.22, 0.21,0.21])
    
    # average and convert to from mPa to Pa    
    mu_arr = 1e-3 * np.mean(np.vstack([mu_arr_U, mu_arr_f, mu_arr_c, mu_arr_A]), axis = 0)
    
    return mu_arr, U_arr, f_arr, c_arr, A_arr

def sznitzman():
    '''
    Sznitman 2010 (a) figure 4
    '''
    
    mu_arr_U = np.array([0.95, 1.21, 1.52, 1.91, 3.46, 6.04, 11.54])
    U_arr = np.array([0.37, 0.32, 0.35, 0.35, 0.33, 0.38,0.35])    
    
    mu_arr_f = np.array([0.99, 1.24, 1.57, 1.97, 3.59, 6.33, 12.01])                    
    f_arr = np.array([2.02, 1.97, 1.87, 1.77, 1.62, 1.69, 1.67])

    # average and convert to from mPa to Pa
    mu_arr = np.mean(np.vstack([mu_arr_U, mu_arr_f]), axis = 0) * 1e-3
    
    return mu_arr, U_arr, f_arr

def fang_yen_fit():
    '''
    Fit sigmoids to fang yen data
    '''
    mu_arr, lam_arr, f_arr, A_arr = fang_yen_data()

    log_mu_arr = np.log10(mu_arr)

    # Define the sigmoid function
    def sigmoid(x, a, b, c, d):
        y = a / (1 + np.exp(-c*(x-b))) + d
        return y

    # Fit the sigmoid to wavelength
    popt_lam, _ = curve_fit(sigmoid, log_mu_arr,lam_arr)
    lam_sig_fit = lambda log_mu: sigmoid(log_mu, *popt_lam)

    # Fit the sigmoid to frequency
    popt_f, _ = curve_fit(sigmoid, log_mu_arr, f_arr)
    f_sig_fit = lambda log_mu: sigmoid(log_mu, *popt_f)

    # Fit the sigmoid to amplitude
    a0 = 3.95
    b0 = 0.12    
    c0 = 2.13
    d0 = 2.94
    p0 = [a0, b0, c0, d0] 
    popt_A, _ = curve_fit(sigmoid, log_mu_arr, A_arr, p0=p0)
    A_sig_fit = lambda log_mu: sigmoid(log_mu, *popt_A)
    
    return lam_sig_fit, f_sig_fit, A_sig_fit

def fit_gagnon_sznitman():

    mu_arr_0, U_arr_0 = david_gagnon()[:2]    
    mu_arr_1, U_arr_1 =  sznitzman()[:2]

    U_arr = np.concatenate((U_arr_0, U_arr_1))
    mu_arr = np.concatenate((mu_arr_0, mu_arr_1))
    log_mu_arr = np.log10(mu_arr)

    # Fit 
    def sigmoid(x, a, b, c, d):
        y = a / (1 + np.exp(-c*(x-b))) + d
        return y

    popt_U, _ = curve_fit(sigmoid, log_mu_arr, U_arr)
    U_sig_fit = lambda log_mu: sigmoid(log_mu, *popt_U)
    
    return U_sig_fit

def rikmenspoel_1978():
    '''
    
    '''

    f = np.array([8.26, 10.08, 26.13, 31.82, 36.23, 42.79, 45.69, 51.91]) * ureg.hertz
    lam = np.array([49.11, 46.11, 34.15, 32.75, 32.28, 30.51, 30.81, 27.93]) * ureg.micrometer
    A_real = np.array([9.12, 8.86, 5.23, 4.61, 4.51, 4.65, 4.77, 3.81]) * ureg.micrometer
    
    s = np.array([1.0, 0.9, 1.1, 1.2, 1.2, 5.4, 5.5, 4.5, 5.6, 8.6, 
        10.0, 10.4, 11.1, 12.4, 19.7, 20.0, 20.7, 21.4, 22.0, 29.4, 
        30.3, 30.6, 31.5, 30.0, 39.2, 40.7, 41.0]) * ureg.micrometer    
    
    A = np.array([5244.0, 4052.0, 3826.0, 3456.0, 3158.0, 2705.0, 2523.0, 
        2350.0, 2095.0, 1635.0, 1635.0, 1889.0, 1707.0, 1721.0, 1630.0,
        1405.0, 1659.0, 1549.0, 1709.0, 1494.0, 1930.0, 1581.0, 1450.0, 
        1240.0, 1381.0, 1445.0, 1227.0]) / ureg.centimeter

    L0 = sperm_param_dict['L0']    

    A_real_star = A_real / L0
    A_real_star.ito_base_units()
    A_star = A * L0
    A_star.ito_base_units()
    s_star = s / L0
    s_star.ito_base_units()
    
    lam_star = lam / L0    
    lam_star.ito_base_units()
        
    f_avg = 35
      
    data = {                
        'L0': 43*ureg.micrometer,
        'f': f,
        'lam': lam,
        'A_real': A_real,
        's': s,
        'A': A,                
        'f_avg': f_avg,
        'lam_star': lam_star,
        'A_real_star': A_real_star,
        'A_star': A_star,
        's_star': s_star
    }
    
    return data
    
def fit_rikmenspoel_1978():
    '''
    Fit rikmenspoel data
    '''
    
    data = rikmenspoel_1978()
    
    f, s = data['f'], data['s_star'] 
    lam, A_real, A = data['lam_star'], data['A_real_star'], data['A_star']
    
    # # Fit the polynomial
    c = np.polyfit(f.magnitude, lam.magnitude, 2)    
    # Create a polynomial function using the c
    lam_fit = np.poly1d(c)

    #Fit the polynomial
    c = np.polyfit(f.magnitude, A_real.magnitude, 2)    
    # Create a polynomial function using the c    
    A_real_fit = np.poly1d(c)
    
    # def sigmoid(s, a, b, c, d):
    #     y = a / (1 + np.exp(-c*(s-b))) + d
    #     return y
    #
    # A_avg = A[s > 0.25].mean()
    #
    # # Fit the sigmoid to wavelength
    # popt_lam, _ = curve_fit(sigmoid, 
    #     s.magnitude, A.magnitude, p0=[10, 5, -10, A_avg])
    # A_fit = lambda s: sigmoid(s, *popt_lam)

    #Fit the polynomial
    
    #assign higher weight to maximum value
    w = np.ones(len(A.magnitude))
    w[A.magnitude.argmax()] = 2
       
    c = np.polyfit(s.magnitude, A.magnitude, 6, w = w)    
    # Create a polynomial function using the c    
    A_fit = np.poly1d(c)
        
    return lam_fit, A_real_fit, A_fit




def default_sweep_parameter():
    '''
    Default sweep hyper parameter
    '''            
    parser = ArgumentParser(description = 'sweep-parameter', allow_abbrev=False)

    parser.add_argument('--worker', type = int, default = 10,
        help = 'Number of processes')         
    parser.add_argument('--run', action=BooleanOptionalAction, default = True,
        help = 'If true, sweep is run. Set to false if sweep has already been run and cached.')     
    parser.add_argument('--pool', action=BooleanOptionalAction, default = True,
        help = 'If true, FrameSequences are pickled to disk') 
    parser.add_argument('--analyse', action=BooleanOptionalAction, default = True,
        help = 'If true, analyse pooled raw data')     
    parser.add_argument('--overwrite', action=BooleanOptionalAction, default = False,
        help = 'If true, already existing simulation results are overwritten')
    parser.add_argument('--debug', action=BooleanOptionalAction, default = False,
        help = 'If true, exception handling is turned off which is helpful for debugging')    
    parser.add_argument('--save_to_storage', action=BooleanOptionalAction, default = False,
        help = 'If true, results are saved to external storage filesystem specified in dirs.py')         
    
    # Frame keys
    parser.add_argument('--t', action=BooleanOptionalAction, default = True,
        help = 'If true, save time stamps')
    parser.add_argument('--r', action=BooleanOptionalAction, default = True,
        help = 'If true, save centreline coordinates')
    parser.add_argument('--theta', action=BooleanOptionalAction, default = True,
        help = 'If true, save Euler angles')    
    parser.add_argument('--d1', action=BooleanOptionalAction, default = False,
        help = 'If true, save director 1')
    parser.add_argument('--d2', action=BooleanOptionalAction, default = False,
        help = 'If true, save director 2')
    parser.add_argument('--d3', action=BooleanOptionalAction, default = False,
        help = 'If true, save director 3')
    parser.add_argument('--k', action=BooleanOptionalAction, default = True,
        help = 'If true, save curvature')
    parser.add_argument('--sig', action=BooleanOptionalAction, default = True,
        help = 'If true, saved shear-stretch')
    parser.add_argument('--k_norm', action=BooleanOptionalAction, default = True,
        help = 'If true, save L2 norm of the real and preferred curvature difference')
    parser.add_argument('--sig_norm', action=BooleanOptionalAction, default = True,
        help = 'If true, save L2 norm of the real and preferred shear-stretch difference')

    # Velocity keys
    parser.add_argument('--r_t', action=BooleanOptionalAction, default = False,
        help = 'If trues, save centreline velocity')
    parser.add_argument('--w', action=BooleanOptionalAction, default = False,
        help = 'If true, save angular velocity')
    parser.add_argument('--k_t', action=BooleanOptionalAction, default = False,
        help = 'If true, save curvature strain rate')
    parser.add_argument('--sig_t', action=BooleanOptionalAction, default = False,
        help = 'If true, save shear-stretch strain rate')
    
    # Forces and torque keys
    parser.add_argument('--f_F', action=BooleanOptionalAction, default = False,
        help = 'If true, save fluid drag force density')
    parser.add_argument('--l_F', action=BooleanOptionalAction, default = False,
        help = 'If true, save fluid drag torque density')
    parser.add_argument('--f_M', action=BooleanOptionalAction, default = False,
        help = 'If true, save muscle force density')
    parser.add_argument('--l_M', action=BooleanOptionalAction, default = False,
        help = 'If true, save muscle torque density')
    parser.add_argument('--N_force', action=BooleanOptionalAction, default = False,
        dest = 'N', help = 'If true, save internal force resultant')
    parser.add_argument('--M_torque', action=BooleanOptionalAction, default = False,
        dest = 'M', help = 'If true, save internal torque resultant')

    # Power keys
    parser.add_argument('--D_F_dot', action=BooleanOptionalAction, default = True,
        help = 'If true, save fluid dissipation rate')
    parser.add_argument('--D_I_dot', action=BooleanOptionalAction, default = True,
        help = 'If true, save internal dissipation rate')
    parser.add_argument('--W_dot', action=BooleanOptionalAction, default = True,
        help = 'If true, save mechanical muscle power')
    parser.add_argument('--V_dot', action=BooleanOptionalAction, default = True,
        help = 'If true, save potential rate')
    parser.add_argument('--V', action=BooleanOptionalAction, default = False,
        help = 'If true, save potential')
    
    # Control key
    parser.add_argument('--k0', action=BooleanOptionalAction, default = True,
        help = 'If true, save controls')
    parser.add_argument('--sig0', action=BooleanOptionalAction, default = True,
        help = 'If true, save controls')

    # Analyse  
    parser.add_argument('--U', action=BooleanOptionalAction, default = True,
        help = 'If true, calculate swimming speed U')
    parser.add_argument('--E', action=BooleanOptionalAction, default = True,
        help = 'If true, calculate L2 norm between real and preferred curvature')
    parser.add_argument('--A_real', action=BooleanOptionalAction, default = True,
        dest = 'A', help = 'If true, calculate real curvature amplitude')

    return parser

def sweep_a_b(argv):
    '''
    Parameter sweep over time scale ratios a and b

    Show that swimming speed and energy are fully captured 
    by the system input time scale ratios.          
    '''    

    # Parse sweep parameter
    sweep_parser = default_sweep_parameter()    

    sweep_parser.add_argument('--a', 
        type=float, nargs=3, default = [-2, 3, 1.0])    
    sweep_parser.add_argument('--b', 
        type=float, nargs=3, default = [-3, 0, 1.0])    

    sweep_param = sweep_parser.parse_known_args(argv)[0]    
    
    # The argumentparser for the sweep parameter has a boolean argument 
    # for ever frame key and control key which can be set to true
    # if it should be saved 
    FK = [k for k in FRAME_KEYS if getattr(sweep_param, k)]    
    CK = [k for k in CONTROL_KEYS if getattr(sweep_param, k)]

    # Parse model parameter
    model_parser = UndulationExperiment.parameter_parser()
    model_param = model_parser.parse_known_args(argv)[0]

    # Print all model parameter whose value has been
    # set via the command line
    cml_args = {k: v for k, v in vars(model_param).items() 
        if v != model_parser.get_default(k)}
    
    if len(cml_args) != 0: 
        print(cml_args)

    # Create the ParameterGrid over which we want to run
    # the undulation experiments
    a_min, a_max = sweep_param.a[0], sweep_param.a[1]
    a_step = sweep_param.a[2]

    b_min, b_max = sweep_param.b[0], sweep_param.b[1]
    b_step = sweep_param.b[2]

    a_param = {'v_min': a_min, 'v_max': a_max + 0.1*a_step, 
        'N': None, 'step': a_step, 'round': 4, 'log': True}    

    b_param = {'v_min': b_min, 'v_max': b_max + 0.1*b_step, 
        'N': None, 'step': b_step, 'round': 5, 'log': True}

    grid_param = {'a': a_param, 'b': b_param}
    
    PG = ParameterGrid(vars(model_param), grid_param)

    if sweep_param.save_to_storage:
        log_dir, sim_dir, sweep_dir = create_storage_dir()     
    else:
        from minimal_worm.experiments.undulation import sweep_dir, log_dir, sim_dir
        
    # Experiments are run using the Sweeper class for parallelization 
    if sweep_param.run:
        Sweeper.run_sweep(
            sweep_param.worker, 
            PG, 
            UndulationExperiment.stw_control_sequence, 
            FK,
            log_dir, 
            sim_dir, 
            sweep_param.overwrite, 
            sweep_param.debug,
            'UExp')

    PG_filepath = PG.save(log_dir)
    print(f'Finished sweep! Save ParameterGrid to {PG_filepath}')
        
    # Pool and save simulation results to hdf5
    filename = Path(
        f'raw_data_'
        f'a_min={a_min}_a_max={a_max}_a_step={a_step}_'
        f'b_min={b_min}_b_max={b_max}_b_step={b_step}_'
        f'A={model_param.A}_lam={model_param.lam}_'
        f'T={model_param.T}_'        
        f'N={model_param.N}_dt={model_param.dt}.h5')
    
    h5_filepath = sweep_dir / filename

    if sweep_param.pool:        
        Sweeper.save_sweep_to_h5(PG, h5_filepath, sim_dir, FK, CK)

    if sweep_param.analyse:
        analyse(h5_filepath, what_to_calculate=sweep_param)
                
    return

def sweep_A_lam_a_b(argv):
    '''
    Parameter sweep undulation parameter A, lam
    and the time scale ratios a and b

    Show if swimming speed and energy are fully captured 
    by the system input time scale ratios.          
    '''    

    # Parse sweep parameter
    sweep_parser = default_sweep_parameter()    

    sweep_parser.add_argument('--A', 
        type=float, nargs=3, default = [2.0, 10.0, 2.0])    
    sweep_parser.add_argument('--lam', 
        type=float, nargs=3, default = [0.5, 2.0, 0.5])    
    sweep_parser.add_argument('--a', 
        type=float, nargs=3, default = [-2, 3, 1.0])    
    sweep_parser.add_argument('--b', 
        type=float, nargs=3, default = [-3, 0, 1.0])    

    sweep_param = sweep_parser.parse_known_args(argv)[0]    

    # The argumentparser for the sweep parameter has a boolean argument 
    # for ever frame key and control key which can be set to true
    # if it should be saved 
    FK = [k for k in FRAME_KEYS if getattr(sweep_param, k)]    
    CK = [k for k in CONTROL_KEYS if getattr(sweep_param, k)]

    # Parse model parameter
    model_parser = UndulationExperiment.parameter_parser()
    model_param = model_parser.parse_known_args(argv)[0]

    # Print all model parameter whose value has been
    # set via the command line
    cml_args = {k: v for k, v in vars(model_param).items() 
        if v != model_parser.get_default(k)}
    
    if len(cml_args) != 0: 
        print(cml_args)
    
    # Create the ParameterGrid over which we want to run
    # the undulation experiments
    A_min, A_max = sweep_param.A[0], sweep_param.A[1]
    A_step = sweep_param.A[2]

    lam_min, lam_max = sweep_param.lam[0], sweep_param.lam[1]
    lam_step = sweep_param.lam[2]

    a_min, a_max = sweep_param.a[0], sweep_param.a[1]
    a_step = sweep_param.a[2]

    b_min, b_max = sweep_param.b[0], sweep_param.b[1]
    b_step = sweep_param.b[2]

    A_param = {'v_min': A_min, 'v_max': A_max + 0.1*a_step, 
        'N': None, 'step': A_step, 'round': 1}    

    lam_param = {'v_min': lam_min, 'v_max': lam_max + 0.1*lam_step, 
        'N': None, 'step': lam_step, 'round': 2}    

    a_param = {'v_min': a_min, 'v_max': a_max + 0.1*a_step, 
        'N': None, 'step': a_step, 'round': 4, 'log': True}    

    b_param = {'v_min': b_min, 'v_max': b_max + 0.1*b_step, 
        'N': None, 'step': b_step, 'round': 5, 'log': True}

    grid_param = {'A': A_param, 'lam': lam_param, 'a': a_param, 'b': b_param}
    
    PG = ParameterGrid(vars(model_param), grid_param)

    if sweep_param.save_to_storage:
        log_dir, sim_dir, sweep_dir = create_storage_dir()     
    else:
        from minimal_worm.experiments.undulation import sweep_dir, log_dir, sim_dir

    # Experiments are run using the Sweeper class for parallelization 
    if sweep_param.run:
        Sweeper.run_sweep(
            sweep_param.worker, 
            PG, 
            UndulationExperiment.stw_control_sequence, 
            FK,
            log_dir, 
            sim_dir, 
            sweep_param.overwrite, 
            sweep_param.debug,
            'UExp')

    PG_filepath = PG.save(log_dir)
    print(f'Finished sweep! Save ParameterGrid to {PG_filepath}')


    # Pool and save simulation results to hdf5
    filename = Path(
        f'raw_data_'
        f'A_min={lam_min}_A_max={lam_max}_A_step={lam_step}_'        
        f'lam_min={lam_min}_lam_max={lam_max}_lam_step={lam_step}_'
        f'a_min={a_min}_a_max={a_max}_a_step={a_step}_'
        f'b_min={b_min}_b_max={b_max}_b_step={b_step}_'
        f'T={model_param.T}_'        
        f'N={model_param.N}_dt={model_param.dt}.h5')
    
    h5_filepath = sweep_dir / filename

    if sweep_param.pool:        
        Sweeper.save_sweep_to_h5(PG, h5_filepath, sim_dir, FK, CK)

    # Anaylse simulation results
    if sweep_param.analyse:
        analyse(h5_filepath, what_to_calculate=sweep_param)
                
    return

def sweep_lam_a_b(argv):
    '''
    Sweeps over
        - wave length lambda
        - time scale ratio a
        - time scale rario b
    
    Why? 
    
    How does the position of the transition band characterized by the 
    contour U/U_max = 0.5 changes with the wavelength of the undulation?
    '''    
    # Parse sweep parameter
    sweep_parser = default_sweep_parameter()    

    sweep_parser.add_argument('--lam', 
        type=float, nargs=3, default = [0.6, 2.0, 0.2])    
    sweep_parser.add_argument('--a', 
        type=float, nargs=3, default = [0.0, 4, 0.2])    
    sweep_parser.add_argument('--b', 
        type=float, nargs=3, default = [-3, 0, 0.2])    
    
    sweep_param = sweep_parser.parse_known_args(argv)[0]    
    
    # The argumentparser for the sweep parameter has a boolean argument 
    # for ever frame key and control key which can be set to true
    # if it should be saved 
    FK = [k for k in FRAME_KEYS if getattr(sweep_param, k)]    
    CK = [k for k in CONTROL_KEYS if getattr(sweep_param, k)]

    # parse model parameter and convert to dict
    model_parser = UndulationExperiment.parameter_parser()
    model_param = model_parser.parse_known_args(argv)[0]
    model_param.use_c = True

    # Print all model parameter whose value has been
    # set via the command line
    cml_args = {k: v for k, v in vars(model_param).items() 
        if v != model_parser.get_default(k)}
    
    if len(cml_args) != 0: 
        print(cml_args)

    # Create the ParameterGrid over which we want to run
    # the undulation experiments
    lam_min, lam_max = sweep_param.lam[0], sweep_param.lam[1]
    lam_step = sweep_param.lam[2]

    a_min, a_max = sweep_param.a[0], sweep_param.a[1]
    a_step = sweep_param.a[2]

    b_min, b_max = sweep_param.b[0], sweep_param.b[1]
    b_step = sweep_param.b[2]

    lam_param = {'v_min': lam_min, 'v_max': lam_max + 0.1*lam_step, 
        'N': None, 'step': lam_step, 'round': 2}    

    a_param = {'v_min': a_min, 'v_max': a_max + 0.1*a_step, 
        'N': None, 'step': a_step, 'round': 4, 'log': True}    

    b_param = {'v_min': b_min, 'v_max': b_max + 0.1*b_step, 
        'N': None, 'step': b_step, 'round': 5, 'log': True}

    grid_param = {'lam': lam_param, 'a': a_param, 'b': b_param}
    
    PG = ParameterGrid(vars(model_param), grid_param)

    if sweep_param.save_to_storage:
        log_dir, sim_dir, sweep_dir = create_storage_dir()     
    else:
        from minimal_worm.experiments.undulation import sweep_dir, log_dir, sim_dir
    
    # Experiments are run using the Sweeper class for parallelization     
    if sweep_param.run:
        Sweeper.run_sweep(
            sweep_param.worker, 
            PG, 
            UndulationExperiment.stw_control_sequence, 
            FK,
            log_dir, 
            sim_dir, 
            sweep_param.overwrite, 
            sweep_param.debug,
            'UExp')


    PG_filepath = PG.save(log_dir)
    print(f'Finished sweep! Save ParameterGrid to {PG_filepath}')

    # Pool and save simulation results to hdf5                   
    filename = Path(
        f'raw_data_'
        f'lam_min={lam_min}_lam_max={lam_max}_lam_step={lam_step}_'
        f'a_min={a_min}_a_max={a_max}_a_step={a_step}_'
        f'b_min={b_min}_b_max={b_max}_b_step={b_step}_'
        f'c={model_param.c}_T={model_param.T}_'        
        f'N={model_param.N}_dt={model_param.dt}.h5')
    
    h5_filepath = sweep_dir / filename

    if sweep_param.pool:        
        Sweeper.save_sweep_to_h5(PG, h5_filepath, sim_dir, FK, CK)

    # Anaylse simulation results
    if sweep_param.analyse:
        analyse(h5_filepath, what_to_calculate=sweep_param)
                
    return

def sweep_c_a_b(argv):
    '''
    Sweeps over
        - Amplitude wavenumber ratio c 
        - time scale ratio a    
        - time scale rario b
    
    Why? 
    
    How does the position of the transition band characterized by the 
    contour U/U_max = 0.5 changes with c?
    '''    
    # Parse sweep parameter
    sweep_parser = default_sweep_parameter()    

    sweep_parser.add_argument('--c', 
        type=float, nargs=3, default = [0.4, 1.4, 0.2])    
    sweep_parser.add_argument('--a', 
        type=float, nargs=3, default = [0.0, 4, 0.2])    
    sweep_parser.add_argument('--b', 
        type=float, nargs=3, default = [-3, 0, 0.2])    

    sweep_param = sweep_parser.parse_known_args(argv)[0]    

    # The argumentparser for the sweep parameter has a boolean argument 
    # for ever frame key and control key which can be set to true
    # if it should be saved 
    FK = [k for k in FRAME_KEYS if getattr(sweep_param, k)]
    CK = [k for k in CONTROL_KEYS if getattr(sweep_param, k)]

    # Parse model parameter
    model_parser = UndulationExperiment.parameter_parser()
    model_param = model_parser.parse_known_args(argv)[0]
    model_param.use_c = True

    # Print all model parameter whose value has been
    # set via the command line
    cml_args = {k: v for k, v in vars(model_param).items() 
        if v != model_parser.get_default(k)}
    
    if len(cml_args) != 0: 
        print(cml_args)


    # Create the ParameterGrid over which we want to run
    # the undulation experiments
    c_min, c_max = sweep_param.c[0], sweep_param.c[1]
    c_step = sweep_param.c[2]

    c_param = {'v_min': c_min, 'v_max': c_max + 0.1*c_step, 
        'N': None, 'step': c_step, 'round': 1}    

    a_min, a_max = sweep_param.a[0], sweep_param.a[1]
    a_step = sweep_param.a[2]

    b_min, b_max = sweep_param.b[0], sweep_param.b[1]
    b_step = sweep_param.b[2]

    a_param = {'v_min': a_min, 'v_max': a_max + 0.1*a_step, 
        'N': None, 'step': a_step, 'round': 4, 'log': True}    

    b_param = {'v_min': b_min, 'v_max': b_max + 0.1*b_step, 
        'N': None, 'step': b_step, 'round': 5, 'log': True}

    grid_param = {'c': c_param, 'a': a_param, 'b': b_param}
    
    PG = ParameterGrid(vars(model_param), grid_param)

    if sweep_param.save_to_storage:
        log_dir, sim_dir, sweep_dir = create_storage_dir()     
    else:
        from minimal_worm.experiments.undulation import sweep_dir, log_dir, sim_dir

    # Experiments are run using the Sweeper class for parallelization            
    if sweep_param.run:
        Sweeper.run_sweep(
            sweep_param.worker, 
            PG, 
            UndulationExperiment.stw_control_sequence, 
            FK,
            log_dir, 
            sim_dir, 
            sweep_param.overwrite, 
            sweep_param.debug,
            'UExp')

    PG_filepath = PG.save(log_dir)
    print(f'Finished sweep! Save ParameterGrid to {PG_filepath}')
        
    # Pool and save simulation results to hdf5
    filename = Path(
        f'raw_data_'
        f'c_min={c_min}_c_max={c_max}_c_step={c_step}_'        
        f'a_min={a_min}_a_max={a_max}_a_step={a_step}_'
        f'b_min={b_min}_b_max={b_max}_b_step={b_step}_'
        f'lam={model_param.lam}_T={model_param.T}_'        
        f'N={model_param.N}_dt={model_param.dt}.h5')
    
    h5_filepath = sweep_dir / filename

    if sweep_param.pool:        
        Sweeper.save_sweep_to_h5(PG, h5_filepath, sim_dir, FK, CK)
        
    # Analyse simulation results
    if sweep_param.analyse:
        analyse(h5_filepath, what_to_calculate=sweep_param)
                
    return

def sweep_c_lam_a_b(argv):
    '''
    Parameter sweep over time scale ratios a and b

    Show that swimming speed and energy are fully captured 
    by the system input time scale ratios.          
    '''    
    # Parse sweep parameter
    sweep_parser = default_sweep_parameter()    

    sweep_parser.add_argument('--c', 
        type=float, nargs=3, default = [0.4, 1.4, 0.2])    
    sweep_parser.add_argument('--lam', 
        type=float, nargs=3, default = [0.5, 2.0, 0.5])    
    sweep_parser.add_argument('--a', 
        type=float, nargs=3, default = [-2, 3, 1.0])    
    sweep_parser.add_argument('--b', 
        type=float, nargs=3, default = [-3, 0, 1.0])    

    sweep_param = sweep_parser.parse_known_args(argv)[0]    

    # The argumentparser for the sweep parameter has a boolean argument 
    # for ever frame key and control key which can be set to true
    # if it should be saved 
    FK = [k for k in FRAME_KEYS if getattr(sweep_param, k)]
    CK = [k for k in CONTROL_KEYS if getattr(sweep_param, k)]

    model_parser = UndulationExperiment.parameter_parser()
    model_param = model_parser.parse_known_args(argv)[0]
    model_param.use_c = True

    # Print all model parameter whose value has been
    # set via the command line
    cml_args = {k: v for k, v in vars(model_param).items() 
        if v != model_parser.get_default(k)}
    
    if len(cml_args) != 0: 
        print(cml_args)


    # Create the ParameterGrid over which we want to run
    # the undulation experiments
    c_min, c_max = sweep_param.c[0], sweep_param.c[1]
    c_step = sweep_param.c[2]

    lam_min, lam_max = sweep_param.lam[0], sweep_param.lam[1]
    lam_step = sweep_param.lam[2]

    a_min, a_max = sweep_param.a[0], sweep_param.a[1]
    a_step = sweep_param.a[2]

    b_min, b_max = sweep_param.b[0], sweep_param.b[1]
    b_step = sweep_param.b[2]

    c_param = {'v_min': c_min, 'v_max': c_max + 0.1*c_step, 
        'N': None, 'step': c_step, 'round': 1}    

    lam_param = {'v_min': lam_min, 'v_max': lam_max + 0.1*lam_step, 
        'N': None, 'step': lam_step, 'round': 2}    

    a_param = {'v_min': a_min, 'v_max': a_max + 0.1*a_step, 
        'N': None, 'step': a_step, 'round': 4, 'log': True}    

    b_param = {'v_min': b_min, 'v_max': b_max + 0.1*b_step, 
        'N': None, 'step': b_step, 'round': 5, 'log': True}

    grid_param = {'c': c_param, 'lam': lam_param, 'a': a_param, 'b': b_param}
        
    PG = ParameterGrid(vars(model_param), grid_param)

    if sweep_param.save_to_storage:
        log_dir, sim_dir, sweep_dir = create_storage_dir()     
    else:
        from minimal_worm.experiments.undulation import sweep_dir, log_dir, sim_dir
           
    # Experiments are run using the Sweeper class for parallelization 
    if sweep_param.run:
        Sweeper.run_sweep(
            sweep_param.worker, 
            PG, 
            UndulationExperiment.stw_control_sequence, 
            FK,
            log_dir, 
            sim_dir, 
            sweep_param.overwrite, 
            sweep_param.debug,
            'UExp')

    PG_filepath = PG.save(log_dir)
    print(f'Finished sweep! Save ParameterGrid to {PG_filepath}')

    # Pool and save simulation results to hdf5
    filename = Path(
        f'raw_data_'
        f'c_min={c_min}_c_max={c_max}_c_step={c_step}_'        
        f'lam_min={lam_min}_lam_max={lam_max}_lam_step={lam_step}_'
        f'a_min={a_min}_a_max={a_max}_a_step={a_step}_'
        f'b_min={b_min}_b_max={b_max}_b_step={b_step}_'
        f'T={model_param.T}_'        
        f'N={model_param.N}_dt={model_param.dt}.h5')
    
    h5_filepath = sweep_dir / filename

    if sweep_param.pool:        
        Sweeper.save_sweep_to_h5(PG, h5_filepath, sim_dir, FK, CK)

    # Analyse simulation results
    if sweep_param.analyse:
        analyse(h5_filepath, what_to_calculate=sweep_param)
                
    return

def sweep_C_a_b(argv):
    '''
    Sweeps over
        - drag coefficient ratio C
        - time scale ratio a
        - time scale rario b
    
    Why? 
    
    Find out if the region which marks the transition from internal 
    to external dissipation dominated regime changes as function of C
    '''    
    # Parse sweep parameter
    sweep_parser = default_sweep_parameter()    

    sweep_parser.add_argument('--C', 
        type=float, nargs=3, default = [1.0, 10.0, 2.0])    
        
    sweep_parser.add_argument('--a', 
        type=float, nargs=3, default = [-2, 3, 0.5])    
    sweep_parser.add_argument('--b', 
        type=float, nargs=3, default = [-3, 0, 0.5])    

    sweep_param = sweep_parser.parse_known_args(argv)[0]    

    # The argumentparser for the sweep parameter has a boolean argument 
    # for ever frame key and control key which can be set to true
    # if it should be saved 
    FK = [k for k in FRAME_KEYS if getattr(sweep_param, k)]
    CK = [k for k in CONTROL_KEYS if getattr(sweep_param, k)]

    # Parse model parameter
    model_parser = UndulationExperiment.parameter_parser()
    model_param = model_parser.parse_known_args(argv)[0]

    # Print all model parameter whose value has been
    # set via the command line
    cml_args = {k: v for k, v in vars(model_param).items() 
        if v != model_parser.get_default(k)}
    
    if len(cml_args) != 0: 
        print(cml_args)

    # Create the ParameterGrid over which we want to run
    # the undulation experiments
    C_min, C_max = sweep_param.C[0], sweep_param.C[1]
    C_step = sweep_param.C[2]

    a_min, a_max = sweep_param.a[0], sweep_param.a[1]
    a_step = sweep_param.a[2]

    b_min, b_max = sweep_param.b[0], sweep_param.b[1]
    b_step = sweep_param.b[2]


    C_param = {'v_min': C_min, 'v_max': C_max + 0.1*C_step, 
        'N': None, 'step': C_step, 'round': 1}    

    a_param = {'v_min': a_min, 'v_max': a_max + 0.1*a_step, 
        'N': None, 'step': a_step, 'round': 4, 'log': True}    

    b_param = {'v_min': b_min, 'v_max': b_max + 0.1*b_step, 
        'N': None, 'step': b_step, 'round': 5, 'log': True}

    grid_param = {'C': C_param, 'a': a_param, 'b': b_param}
    
    PG = ParameterGrid(vars(model_param), grid_param)

    if sweep_param.save_to_storage:
        log_dir, sim_dir, sweep_dir = create_storage_dir()     
    else:
        from minimal_worm.experiments.undulation import sweep_dir, log_dir, sim_dir

    # Experiments are run using the Sweeper class for parallelization            
    if sweep_param.run:
        Sweeper.run_sweep(
            sweep_param.worker, 
            PG, 
            UndulationExperiment.stw_control_sequence, 
            FK,
            log_dir, 
            sim_dir, 
            sweep_param.overwrite, 
            sweep_param.debug,
            'UExp')

    PG_filepath = PG.save(log_dir)
    print(f'Finished sweep! Save ParameterGrid to {PG_filepath}')

    # Pool and save simulation results to hdf5
    filename = Path(
        f'raw_data_'
        f'C_min={C_min}_C_max={C_max}_C_step={C_step}_'
        f'a_min={a_min}_a_max={a_max}_a_step={a_step}_'
        f'b_min={b_min}_b_max={b_max}_b_step={b_step}_'
        f'A={model_param.A}_lam={model_param.lam}_T={model_param.T}_'        
        f'N={model_param.N}_dt={model_param.dt}.h5')
    
    h5_filepath = sweep_dir / filename

    if sweep_param.pool:        
        Sweeper.save_sweep_to_h5(PG, h5_filepath, sim_dir, FK, CK)

    if sweep_param.analyse:
        analyse(h5_filepath, what_to_calculate=sweep_param)
                
    return

def sweep_C_c_lam(argv):
    '''
    Sweeps over
        - drag coefficient ratio C
        - time scale ratio a
        - time scale rario b
    
    Why? 
    
    Find out if the region which marks the transition from internal 
    to external dissipation dominated regime changes as function of C
    
    Find out if the fluid dissipation energy surface as a function of
    c and lambda changes.
    '''    
    # Parse sweep parameter
    sweep_parser = default_sweep_parameter()    

    sweep_parser.add_argument('--C', 
        type=float, nargs=3, default = [1.0, 10.0, 2.0])        
    sweep_parser.add_argument('--c', 
        type=float, nargs=3, default = [0.4, 1.4, 0.1])    
    sweep_parser.add_argument('--lam', 
        type=float, nargs=3, default = [0.5, 2.0, 0.1])    
        
    sweep_param = sweep_parser.parse_known_args(argv)[0]    

    # The argumentparser for the sweep parameter has a boolean argument 
    # for ever frame key and control key which can be set to true
    # if it should be saved 
    FK = [k for k in FRAME_KEYS if getattr(sweep_param, k)]
    CK = [k for k in CONTROL_KEYS if getattr(sweep_param, k)]

    # Parse model parameter
    model_parser = UndulationExperiment.parameter_parser()
    model_param = model_parser.parse_known_args(argv)[0]

    # Print all model parameter whose value has been
    # set via the command line
    cml_args = {k: v for k, v in vars(model_param).items() 
        if v != model_parser.get_default(k)}
    
    if len(cml_args) != 0: 
        print(cml_args)

    # Create the ParameterGrid over which we want to run
    # the undulation experiments
    mu = model_param.mu                 
    f_mu = fang_yen_fit()[1]    
    T_c = 1.0 / f_mu(np.log10(mu.magnitude)) * ureg.second
    model_param.T_c = T_c

    C_min, C_max = sweep_param.C[0], sweep_param.C[1]
    C_step = sweep_param.C[2]

    lam_min, lam_max = sweep_param.lam[0], sweep_param.lam[1]
    lam_step = sweep_param.lam[2]

    c_min, c_max = sweep_param.c[0], sweep_param.c[1]
    c_step = sweep_param.c[2]

    C_param = {'v_min': C_min, 'v_max': C_max + 0.1*C_step, 
        'N': None, 'step': C_step, 'round': 1}    

    lam_param = {'v_min': lam_min, 'v_max': lam_max + 0.1*lam_step, 
        'N': None, 'step': lam_step, 'round': 2}    

    c_param = {'v_min': c_min, 'v_max': c_max + 0.1*c_step, 
        'N': None, 'step': c_step, 'round': 2}

    grid_param = {'C': C_param, 'lam': lam_param, 'c': c_param}
    
    PG = ParameterGrid(vars(model_param), grid_param)

    if sweep_param.save_to_storage:
        log_dir, sim_dir, sweep_dir = create_storage_dir()     
    else:
        from minimal_worm.experiments.undulation import sweep_dir, log_dir, sim_dir

    # Experiments are run using the Sweeper class which takes care of parallelization            
    if sweep_param.run:
        Sweeper.run_sweep(
            sweep_param.worker, 
            PG, 
            UndulationExperiment.stw_control_sequence, 
            FK,
            log_dir, 
            sim_dir, 
            sweep_param.overwrite, 
            sweep_param.debug,
            'UExp')

    PG_filepath = PG.save(log_dir)
    print(f'Finished sweep! Save ParameterGrid to {PG_filepath}')
    
    # Pool and save simulation results to hdf5
    filename = Path(
        f'raw_data_'
        f'C_min={C_min}_C_max={C_max}_C_step={C_step}_'
        f'lam_min={lam_min}_lam_max={lam_max}_lam_step={lam_step}_'
        f'c_min={c_min}_c_max={c_max}_c_step={c_step}_'
        f'T={model_param.T}_N={model_param.N}_dt={model_param.dt}.h5')
    
    h5_filepath = sweep_dir / filename

    if sweep_param.pool:        
        Sweeper.save_sweep_to_h5(PG, h5_filepath, sim_dir, FK, CK)

    # Analyse simulation result
    if sweep_param.analyse:
        analyse(h5_filepath, what_to_calculate=sweep_param)
                
    return

def sweep_C_xi_mu_fang_yen(argv):
    '''
    Sweeps over
    
    - drag coefficient ratio C
    - internal time scale xi/E
    - fluid viscosity mu         
    
    Fit frequency f, lam and A over log of fluid viscosity mu 
    to Fang Yeng data                       
    '''
    # Parse sweep parameter
    sweep_parser = default_sweep_parameter()    
    
    sweep_parser.add_argument('--C', 
        type=float, nargs=3, default = [1.5, 4.0, 0.5])    
    sweep_parser.add_argument('--xi', 
        type=float, nargs=3, default = [-4, -1, 0.5])                 
    sweep_parser.add_argument('--mu', 
        type=float, nargs=3, default = [-3, 1, 0.5])        
    
    sweep_param = sweep_parser.parse_known_args(argv)[0]    

    # The argumentparser for the sweep parameter has a boolean argument 
    # for ever frame key and control key which can be set to true
    # if it should be saved 
    FK = [k for k in FRAME_KEYS if getattr(sweep_param, k)]
    CK = [k for k in CONTROL_KEYS if getattr(sweep_param, k)]
                
    # Parse model parameter 
    model_parser = UndulationExperiment.parameter_parser()
    model_param = model_parser.parse_known_args(argv)[0]
    
    model_param.use_c = False
    model_param.a_from_physical = True
    model_param.b_from_physical = True
    # Print all model parameter whose value has been
    # set via the command line
    cml_args = {k: v for k, v in vars(model_param).items() 
        if v != model_parser.get_default(k)}
    
    if len(cml_args) != 0: 
        print(cml_args)

    # Create the ParameterGrid over which we want to run
    # the undulation experiments    
    C_min, C_max = sweep_param.C[0], sweep_param.C[1]
    C_step = sweep_param.C[2]

    xi_min, xi_max = sweep_param.xi[0], sweep_param.xi[1]
    xi_step = sweep_param.xi[2]

    xi_exp_arr = np.arange(xi_min, xi_max + 0.1*xi_step, xi_step) 
    # The smaller xi the smaller the time step needs to be
    dt_arr = [1e-4 if xi < -2 else 1e-3 for xi in xi_exp_arr]
    
    mu_exp_min, mu_exp_max = sweep_param.mu[0], sweep_param.mu[1]
    mu_exp_step = sweep_param.mu[2]
             
    mu_exp_arr = np.arange(mu_exp_min, mu_exp_max + 0.1 * mu_exp_step, mu_exp_step)        
    mu_arr = 10**mu_exp_arr                
        
    lam_mu, f_mu, A_mu = fang_yen_fit()    
    T_c_arr = 1.0 / f_mu(mu_exp_arr)
    lam_arr = lam_mu(mu_exp_arr)
    A_arr = A_mu(mu_exp_arr)

    C_param = {'v_min': C_min, 'v_max': C_max + 0.1*C_step, 
        'N': None, 'step': C_step, 'round': 1}    
        
    eta_param = {'v_min': xi_min, 'v_max': xi_max + 0.1*xi_step, 
        'N': None, 'step': xi_step, 'round': 0, 'log': True, 
        'scale': model_param.E.magnitude, 'quantity': 'pascal*second'}    

    dt_param = {'v_arr': dt_arr, 'round': 4}
    
    T_c_param = {'v_arr': T_c_arr.tolist(), 'round': 2, 'quantity': 'second'}    
    lam_param = {'v_arr': lam_arr.tolist(), 'round': 2}
    A_param = {'v_arr': A_arr.tolist(), 'round': 2}    
    mu_param = {'v_arr': mu_arr.tolist(), 'round': 5, 'quantity': 'pascal*second'}
        
    grid_param = {  
        'C': C_param,
        ('eta', 'dt'): (eta_param, dt_param),
        ('T_c', 'mu', 'A', 'lam'): (T_c_param, mu_param, A_param, lam_param), 
        }

    if sweep_param.save_to_storage:
        log_dir, sim_dir, sweep_dir = create_storage_dir()     
    else:
        from minimal_worm.experiments.undulation import sweep_dir, log_dir, sim_dir
         
    PG = ParameterGrid(vars(model_param), grid_param)
                        
    # Experiments are run using the Sweeper class for parallelization 
    if sweep_param.run:
        Sweeper.run_sweep(
            sweep_param.worker, 
            PG, 
            UndulationExperiment.stw_control_sequence, 
            FK,
            log_dir, 
            sim_dir, 
            sweep_param.overwrite, 
            sweep_param.debug,
            'UExp')

    PG_filepath = PG.save(log_dir)
    print(f'Finished sweep! Save ParameterGrid to {PG_filepath}')
            
    # Pool and save simulation results to hdf5
    filename = Path(
        f'raw_data_fang_yeng_'
        f'C_min={C_min}_C_max={C_max}_C_step={C_step}_'
        f'xi_min={xi_min}_xi_max={xi_max}_xi_step={xi_step}_'
        f'mu_min={mu_exp_min}_mu_max={mu_exp_max}_mu_step={mu_exp_step}_'        
        f'T={model_param.T}_N={model_param.N}_dt={model_param.dt}.h5')
    
    h5_filepath = sweep_dir / filename

    if sweep_param.pool:        
        Sweeper.save_sweep_to_h5(PG, h5_filepath, sim_dir, FK, CK)

    # Analyse simulations results
    if sweep_param.analyse:
        analyse(h5_filepath, what_to_calculate=sweep_param)

    return

def sweep_mu_c_lam_fang_yen(argv):
    '''
    Sweeps over
        - mu: Fluid viscosity
        - c = A/q where A is the undulation amplitude and q the wavenumber
        - lam undulation wavelength
    
    Fit frequency f over log of fluid viscosity mu to Fang Yeng data               
        
    Sweep over c lam grid for every (mu, f) pair.        
    '''    
    # Parse sweep parameter
    sweep_parser = default_sweep_parameter()    

    sweep_parser.add_argument('--mu', 
        type=float, nargs=3, default = [-3, 1, 1.0])        
    sweep_parser.add_argument('--c', 
        type=float, nargs=3, default = [0.4, 1.4, 0.2])    
    sweep_parser.add_argument('--lam', 
        type=float, nargs=3, default = [0.5, 2.0, 0.5])    
    sweep_parser.add_argument('--xi', 
        type=float, )
                                
    sweep_param = sweep_parser.parse_known_args(argv)[0]    

    # The argumentparser for the sweep parameter has a boolean argument 
    # for ever frame key and control key which can be set to true
    # if it should be saved 
    FK = [k for k in FRAME_KEYS if getattr(sweep_param, k)]
    CK = [k for k in CONTROL_KEYS if getattr(sweep_param, k)]

    # Parse model parameter
    model_parser = UndulationExperiment.parameter_parser()
    model_param = model_parser.parse_known_args(argv)[0]
    
    model_param.use_c = True

    model_param.a_from_physical = True
    model_param.b_from_physical = True

    # Print all model parameter whose value has been
    # set via the command line
    cml_args = {k: v for k, v in vars(model_param).items() 
        if v != model_parser.get_default(k)}
    
    if len(cml_args) != 0: 
        print(cml_args)

    # Create the ParameterGrid over which we want to run
    # the undulation experiments
    mu_exp_min, mu_exp_max = sweep_param.mu[0], sweep_param.mu[1]
    mu_exp_step = sweep_param.mu[2]
             
    mu_exp_arr = np.arange(mu_exp_min, mu_exp_max + 0.1 * mu_exp_step, mu_exp_step)        
    mu_arr = 10**mu_exp_arr                
    
    f_mu = fang_yen_fit()[1]    
    T_c_arr = 1.0 / f_mu(mu_exp_arr)

    T_c_param = {'v_arr': T_c_arr.tolist(), 'round': 2, 'quantity': 'second'}
    mu_param = {'v_arr': mu_arr.tolist(), 'round': 5, 'quantity': 'pascal*second'}
    
    c_min, c_max = sweep_param.c[0], sweep_param.c[1]
    c_step = sweep_param.c[2]

    lam_min, lam_max = sweep_param.lam[0], sweep_param.lam[1]
    lam_step = sweep_param.lam[2]

    c_param = {'v_min': c_min, 'v_max': c_max + 0.1*c_step, 
        'N': None, 'step': c_step, 'round': 2}    

    lam_param = {'v_min': lam_min, 'v_max': lam_max + 0.1*lam_step, 
        'N': None, 'step': lam_step, 'round': 2}

    grid_param = {('T_c', 'mu'): (T_c_param, mu_param), 
        'c': c_param, 'lam': lam_param}

    PG = ParameterGrid(vars(model_param), grid_param)

    if sweep_param.save_to_storage:
        log_dir, sim_dir, sweep_dir = create_storage_dir()     
    else:
        from minimal_worm.experiments.undulation import sweep_dir, log_dir, sim_dir

    # Experiments are run using the Sweeper class for parallelization            
    if sweep_param.run:
        Sweeper.run_sweep(
            sweep_param.worker, 
            PG, 
            UndulationExperiment.stw_control_sequence, 
            FK,
            log_dir, 
            sim_dir, 
            sweep_param.overwrite, 
            sweep_param.debug,
            'UExp')

    PG_filepath = PG.save(log_dir)
    print(f'Finished sweep! Save ParameterGrid to {PG_filepath}')

    # Pool and save simulation results to hdf5
    filename = Path(
        f'raw_data_fang_yeng_'
        f'mu_min={mu_exp_min}_mu_max={mu_exp_max}_mu_step={mu_exp_step}_'        
        f'c_min={c_min}_c_max={c_max}_c_step={c_step}_'
        f'lam_min={lam_min}_lam_max={lam_max}_lam_step={lam_step}_'
        f'T={model_param.T}_'
        f'N={model_param.N}_dt={model_param.dt}.h5')
    
    h5_filepath = sweep_dir / filename

    if sweep_param.pool:        
        Sweeper.save_sweep_to_h5(PG, h5_filepath, sim_dir, FK, CK)
        
    # Analyse simulation results
    if sweep_param.analyse:
        analyse(h5_filepath, what_to_calculate=sweep_param)
                
    return

def sweep_xi_mu_c_lam_fang_yen(argv):
    '''
    Sweeps over
        - mu: Fluid viscosity
        - c = A/q where A is the undulation amplitude and q the wavenumber
        - lam undulation wavelength
    
    Fit frequency f over log of fluid viscosity mu to Fang Yeng data                       
    '''    
    # Parse sweep parameter
    sweep_parser = default_sweep_parameter()    

    sweep_parser.add_argument('--xi', 
        type=float, nargs=3, default = [-3, -1, 1.0])             
    sweep_parser.add_argument('--mu', 
        type=float, nargs=3, default = [-3, 1, 1.0])        
    sweep_parser.add_argument('--c', 
        type=float, nargs=3, default = [0.4, 1.4, 0.2])    
    sweep_parser.add_argument('--lam', 
        type=float, nargs=3, default = [0.5, 2.0, 0.5])    
                
    sweep_param = sweep_parser.parse_known_args(argv)[0]    

    # The argumentparser for the sweep parameters has a boolean argument 
    # for ever frame key and control key which can be set to true
    # if it should be saved 
    FK = [k for k in FRAME_KEYS if getattr(sweep_param, k)]
    CK = [k for k in CONTROL_KEYS if getattr(sweep_param, k)]

    # Parse model parameter
    model_parser = UndulationExperiment.parameter_parser()
    model_param = model_parser.parse_known_args(argv)[0]
    
    model_param.use_c = True

    model_param.a_from_physical = True
    model_param.b_from_physical = True

    # Print all model parameter whose value has been
    # set via the command line
    cml_args = {k: v for k, v in vars(model_param).items() 
        if v != model_parser.get_default(k)}
    
    if len(cml_args) != 0: 
        print(cml_args)

    # Create the ParameterGrid over which we want to run
    # the undulation experiments
    xi_min, xi_max = sweep_param.xi[0], sweep_param.xi[1]
    xi_step = sweep_param.xi[2]

    eta_param = {'v_min': xi_min, 'v_max': xi_max + 0.1*xi_step, 
        'N': None, 'step': xi_step, 'round': 0, 'log': True, 
        'scale': model_param.E.magnitude, 'quantity': 'pascal*second'}    
    
    mu_exp_min, mu_exp_max = sweep_param.mu[0], sweep_param.mu[1]
    mu_exp_step = sweep_param.mu[2]
             
    mu_exp_arr = np.arange(mu_exp_min, mu_exp_max + 0.1 * mu_exp_step, mu_exp_step)        
    mu_arr = 10**mu_exp_arr                
    
    f_mu = fang_yen_fit()[1]    
    T_c_arr = 1.0 / f_mu(mu_exp_arr)

    T_c_param = {'v_arr': T_c_arr.tolist(), 'round': 2, 'quantity': 'second'}
    mu_param = {'v_arr': mu_arr.tolist(), 'round': 5, 'quantity': 'pascal*second'}
    
    c_min, c_max = sweep_param.c[0], sweep_param.c[1]
    c_step = sweep_param.c[2]

    lam_min, lam_max = sweep_param.lam[0], sweep_param.lam[1]
    lam_step = sweep_param.lam[2]

    c_param = {'v_min': c_min, 'v_max': c_max + 0.1*c_step, 
        'N': None, 'step': c_step, 'round': 2}    

    lam_param = {'v_min': lam_min, 'v_max': lam_max + 0.1*lam_step, 
        'N': None, 'step': lam_step, 'round': 2}

    grid_param = {
        'eta': eta_param,
        ('T_c', 'mu'): (T_c_param, mu_param), 
        'c': c_param, 'lam': lam_param
    }

    PG = ParameterGrid(vars(model_param), grid_param)

    if sweep_param.save_to_storage:
        log_dir, sim_dir, sweep_dir = create_storage_dir()     
    else:
        from minimal_worm.experiments.undulation import sweep_dir, log_dir, sim_dir

    # Experiments are run using the Sweeper class for parallelization            
    if sweep_param.run:
        Sweeper.run_sweep(
            sweep_param.worker, 
            PG, 
            UndulationExperiment.stw_control_sequence, 
            FK,
            log_dir, 
            sim_dir, 
            sweep_param.overwrite, 
            sweep_param.debug,
            'UExp')

    PG_filepath = PG.save(log_dir)
    print(f'Finished sweep! Save ParameterGrid to {PG_filepath}')
        
    # Pool and save simulation results to hdf5
    filename = Path(
        f'raw_data_fang_yeng_'
        f'xi_min={xi_min}_xi_max={xi_max}_xi_step={xi_step}_'        
        f'mu_min={mu_exp_min}_mu_max={mu_exp_max}_mu_step={mu_exp_step}_'        
        f'c_min={c_min}_c_max={c_max}_c_step={c_step}_'
        f'lam_min={lam_min}_lam_max={lam_max}_lam_step={lam_step}_'
        f'phi={model_param.phi}_T={model_param.T}_'
        f'N={model_param.N}_dt={model_param.dt}.h5')
    
    h5_filepath = sweep_dir / filename

    if sweep_param.pool:        
        Sweeper.save_sweep_to_h5(PG, h5_filepath, sim_dir, FK, CK)
        
    # Analyse simulation results
    if sweep_param.analyse:
        analyse(h5_filepath, what_to_calculate=sweep_param)
                
    return

def sweep_C_mu_c_lam_fang_yen(argv):
    '''
    Sweeps over
        - C: Linear drag coefficient ratio
        - mu: Fluid viscosity
        - c = A/q where A is the undulation amplitude and q the wavenumber
        - lam undulation wavelength
    
    Fit frequency f over log of fluid viscosity mu to Fang Yeng data                   
    '''    
    # Parse sweep parameter
    sweep_parser = default_sweep_parameter()    
     
    sweep_parser.add_argument('--C', 
        type=float, nargs=3, default = [1.5, 5.0, 0.5])            
    sweep_parser.add_argument('--mu', 
        type=float, nargs=3, default = [-3, 1, 1.0])        
    sweep_parser.add_argument('--c', 
        type=float, nargs=3, default = [0.4, 1.6, 0.1])    
    sweep_parser.add_argument('--lam', 
        type=float, nargs=3, default = [0.4, 2.0, 0.1])    
    sweep_parser.add_argument('--xi', 
        type=float, )
                
    # The argumentparser for the sweep parameter has a boolean argument 
    # for ever frame key and control key which can be set to true
    # if it should be saved                 
    sweep_param = sweep_parser.parse_known_args(argv)[0]    
    FK = [k for k in FRAME_KEYS if getattr(sweep_param, k)]
    CK = [k for k in CONTROL_KEYS if getattr(sweep_param, k)]

    # Parse model parameter
    model_parser = UndulationExperiment.parameter_parser()
    model_param = model_parser.parse_known_args(argv)[0]
    
    model_param.use_c = True
    model_param.a_from_physical = True
    model_param.b_from_physical = True

    # Print all model parameter whose value has been
    # set via the command line
    cml_args = {k: v for k, v in vars(model_param).items() 
        if v != model_parser.get_default(k)}
    
    if len(cml_args) != 0: 
        print(cml_args)

    # Create the ParameterGrid over which we want to run
    # the undulation experiments
    C_min, C_max= sweep_param.C[0], sweep_param.C[1]
    C_step = sweep_param.C[2]
    
    C_param = {'v_min': C_min, 'v_max': C_max + 0.1*C_step, 
        'N': None, 'step': C_step, 'round': 1}    

    mu_exp_min, mu_exp_max = sweep_param.mu[0], sweep_param.mu[1]
    mu_exp_step = sweep_param.mu[2]
             
    mu_exp_arr = np.arange(mu_exp_min, mu_exp_max + 0.1 * mu_exp_step, mu_exp_step)        
    mu_arr = 10**mu_exp_arr                
    
    f_mu = fang_yen_fit()[1]    
    T_c_arr = 1.0 / f_mu(mu_exp_arr)

    T_c_param = {'v_arr': T_c_arr.tolist(), 'round': 2, 'quantity': 'second'}
    mu_param = {'v_arr': mu_arr.tolist(), 'round': 5, 'quantity': 'pascal*second'}
    
    c_min, c_max = sweep_param.c[0], sweep_param.c[1]
    c_step = sweep_param.c[2]

    lam_min, lam_max = sweep_param.lam[0], sweep_param.lam[1]
    lam_step = sweep_param.lam[2]

    c_param = {'v_min': c_min, 'v_max': c_max + 0.1*c_step, 
        'N': None, 'step': c_step, 'round': 2}    

    lam_param = {'v_min': lam_min, 'v_max': lam_max + 0.1*lam_step, 
        'N': None, 'step': lam_step, 'round': 2}

    grid_param = {'C': C_param, ('T_c', 'mu'): (T_c_param, mu_param), 
        'c': c_param, 'lam': lam_param}

    PG = ParameterGrid(vars(model_param), grid_param)

    if sweep_param.save_to_storage:
        log_dir, sim_dir, sweep_dir = create_storage_dir()     
    else:
        from minimal_worm.experiments.undulation import sweep_dir, log_dir, sim_dir
    
    # Experiments are run using the Sweeper class for parallelization            
    if sweep_param.run:
        Sweeper.run_sweep(
            sweep_param.worker, 
            PG, 
            UndulationExperiment.stw_control_sequence, 
            FK,
            log_dir, 
            sim_dir, 
            sweep_param.overwrite, 
            sweep_param.debug,
            'UExp')

    PG_filepath = PG.save(log_dir)
    print(f'Finished sweep! Save ParameterGrid to {PG_filepath}')

    # Pool and save simulation results to hdf5
    filename = Path(
        f'raw_data_fang_yeng_'
        f'C_min={C_min}_C_max={C_max}_C_step={C_step}_'                
        f'mu_min={mu_exp_min}_mu_max={mu_exp_max}_mu_step={mu_exp_step}_'        
        f'c_min={c_min}_c_max={c_max}_c_step={c_step}_'
        f'lam_min={lam_min}_lam_max={lam_max}_lam_step={lam_step}_'
        f'phi={model_param.phi}_T={model_param.T}_'
        f'N={model_param.N}_dt={model_param.dt}.h5')
    
    h5_filepath = sweep_dir / filename

    if sweep_param.pool:        
        Sweeper.save_sweep_to_h5(PG, h5_filepath, sim_dir, FK, CK)

    if sweep_param.analyse:
        analyse(h5_filepath, what_to_calculate=sweep_param)
                
    return

def sweep_C_xi_mu_c_lam_fang_yen(argv):
    '''
    Sweeps over
        - C: Linear drag coefficient ratio
        - mu: Fluid viscosity
        - c = A/q where A is the undulation amplitude and q the wavenumber
        - lam undulation wavelength
    
    Fit frequency f over log of fluid viscosity mu to Fang Yeng data                   
    '''        
    # Parse sweep parameter
    sweep_parser = default_sweep_parameter()    
     
    sweep_parser.add_argument('--C', 
        type=float, nargs=3, default = [2.0, 10.0, 2.0])            
    sweep_parser.add_argument('--xi', 
        type=float, nargs=3, default = [-3, -1, 1.0])        
    
    sweep_parser.add_argument('--mu', 
        type=float, nargs=3, default = [-3, 1, 1.0])        
    sweep_parser.add_argument('--c', 
        type=float, nargs=3, default = [0.4, 1.6, 0.1])    
    sweep_parser.add_argument('--lam', 
        type=float, nargs=3, default = [0.4, 2.0, 0.1])    
                                
    # The argumentparser for the sweep parameter has a boolean argument 
    # for ever frame key and control key which can be set to true
    # if it should be saved 
    sweep_param = sweep_parser.parse_known_args(argv)[0]    
    FK = [k for k in FRAME_KEYS if getattr(sweep_param, k)]
    CK = [k for k in CONTROL_KEYS if getattr(sweep_param, k)]

    # Parse model parameter
    model_parser = UndulationExperiment.parameter_parser()
    model_param = model_parser.parse_known_args(argv)[0]
    
    model_param.use_c = True
    model_param.a_from_physical = True
    model_param.b_from_physical = True

    # Print all model parameter whose value has been
    # set via the command line
    cml_args = {k: v for k, v in vars(model_param).items() 
        if v != model_parser.get_default(k)}
    
    if len(cml_args) != 0: 
        print(cml_args)

    # Pool and save simulation results to hdf5
    C_min, C_max= sweep_param.C[0], sweep_param.C[1]
    C_step = sweep_param.C[2]
    
    C_param = {'v_min': C_min, 'v_max': C_max + 0.1*C_step, 
        'N': None, 'step': C_step, 'round': 1}    

    xi_min, xi_max = sweep_param.xi[0], sweep_param.xi[1]
    xi_step = sweep_param.xi[2]

    eta_param = {'v_min': xi_min, 'v_max': xi_max + 0.1*xi_step, 
        'N': None, 'step': xi_step, 'round': 0, 'log': True, 
        'scale': model_param.E.magnitude, 'quantity': 'pascal*second'}    

    mu_exp_min, mu_exp_max = sweep_param.mu[0], sweep_param.mu[1]
    mu_exp_step = sweep_param.mu[2]
             
    mu_exp_arr = np.arange(mu_exp_min, mu_exp_max + 0.1 * mu_exp_step, mu_exp_step)        
    mu_arr = 10**mu_exp_arr                
    
    f_mu = fang_yen_fit()[1]    
    T_c_arr = 1.0 / f_mu(mu_exp_arr)

    T_c_param = {'v_arr': T_c_arr.tolist(), 'round': 2, 'quantity': 'second'}
    mu_param = {'v_arr': mu_arr.tolist(), 'round': 5, 'quantity': 'pascal*second'}
    
    c_min, c_max = sweep_param.c[0], sweep_param.c[1]
    c_step = sweep_param.c[2]

    lam_min, lam_max = sweep_param.lam[0], sweep_param.lam[1]
    lam_step = sweep_param.lam[2]

    c_param = {'v_min': c_min, 'v_max': c_max + 0.1*c_step, 
        'N': None, 'step': c_step, 'round': 2}    

    lam_param = {'v_min': lam_min, 'v_max': lam_max + 0.1*lam_step, 
        'N': None, 'step': lam_step, 'round': 2}

    grid_param = {
        'C': C_param, 
        'eta': eta_param,
        ('T_c', 'mu'): (T_c_param, mu_param), 
        'c': c_param, 'lam': lam_param}

    PG = ParameterGrid(vars(model_param), grid_param)

    if sweep_param.save_to_storage:
        log_dir, sim_dir, sweep_dir = create_storage_dir()     
    else:
        from minimal_worm.experiments.undulation import sweep_dir, log_dir, sim_dir

    # Experiments are run using the Sweeper class for parallelization            
    if sweep_param.run:
        Sweeper.run_sweep(
            sweep_param.worker, 
            PG, 
            UndulationExperiment.stw_control_sequence, 
            FK,
            log_dir, 
            sim_dir, 
            sweep_param.overwrite, 
            sweep_param.debug,
            'UExp')

    PG_filepath = PG.save(log_dir)
    print(f'Finished sweep! Save ParameterGrid to {PG_filepath}')

    # Pool and save simulation results to hdf5            
    filename = Path(
        f'raw_data_fang_yeng_'
        f'C_min={C_min}_C_max={C_max}_C_step={C_step}_'                
        f'mu_min={mu_exp_min}_mu_max={mu_exp_max}_mu_step={mu_exp_step}_'        
        f'xi_min={xi_min}_xi_max={xi_max}_eta_step={xi_step}_'                        
        f'c_min={c_min}_c_max={c_max}_c_step={c_step}_'
        f'lam_min={lam_min}_lam_max={lam_max}_lam_step={lam_step}_'
        f'phi={model_param.phi}_T={model_param.T}_'
        f'N={model_param.N}_dt={model_param.dt}.h5')
    
    h5_filepath = sweep_dir / filename

    if sweep_param.pool:        
        Sweeper.save_sweep_to_h5(PG, h5_filepath, sim_dir, FK, CK)
        
    # Analyse simulation results
    if sweep_param.analyse:
        analyse(h5_filepath, what_to_calculate=sweep_param)
                
    return

# Sea urchin sperm        
sperm_param_dict = {            
    
    'L0': 43*ureg.micrometer,
    'B_max': 15e-22*ureg.newton*ureg.meter**2,
    'B_min': 3e-22*ureg.newton*ureg.meter**2,        
    'B': 10e-22*ureg.newton*ureg.meter**2,    
    'mu': 1*ureg.millipascal*ureg.second,
    'f_avg': 35*ureg.hertz,
    'R': 0.1*ureg.micrometer
}        
    
def sweep_f_rikmenspoel(argv):
    '''
    Sweep over undulation frequency.
    
    Use the wavelength and curvature amplitude from paper        
    '''
    
    # Parse sweep parameter
    sweep_parser = default_sweep_parameter()    
            
    sweep_parser.add_argument('--f', 
        type=float, nargs=3, default = [10, 50, 1.0])    
    sweep_parser.add_argument('--const_A', action = BooleanOptionalAction, default = False, 
        help = 'If true, curvature amplitude is assumed to be constant')

    # The argumentparser for the sweep parameter has a boolean argument 
    # for ever frame key and control key which can be set to true
    # if it should be saved 
    sweep_param = sweep_parser.parse_known_args(argv)[0]    
    FK = [k for k in FRAME_KEYS if getattr(sweep_param, k)]
    CK = [k for k in CONTROL_KEYS if getattr(sweep_param, k)]

    # Parse model parameter
    model_parser = UndulationExperiment.parameter_parser()
    model_param = model_parser.parse_known_args(argv)[0]

    # Set curvature amplitude    
    if sweep_param.const_A:    
        data = rikmenspoel_1978()
        model_param.A = data['A_star'].magnitude.mean()            
        CS = UndulationExperiment.stw_control_sequence
    
    else:
        A_fit = fit_rikmenspoel_1978()[2]
        model_param.A = A_fit.coef.tolist()
        CS = UndulationExperiment.stw_va_control_sequence
            
    # Set material parameters to the experimental sperm data
    L0 = sperm_param_dict['L0']
    R = sperm_param_dict['R']
    B = sperm_param_dict['B']

    I = 0.25*np.pi*R**4
    E = B/I
    xi = 1e-3 * ureg.second
    eta = E * xi  
    
    model_param.L0 = L0.to_base_units()
    model_param.R = R.to_base_units()
    model_param.E = E.to_base_units()    
    model_param.eta = eta.to_base_units()    

    # Specify all dimensionless parameters which 
    # should be calculated from phyisical parameters                
    model_param.C_from_physical = True
    model_param.D_from_physical = True
    model_param.Y_from_physical = True        
    model_param.a_from_physical = True
    model_param.b_from_physical = True

    # Create the ParameterGrid over which we want to run
    # the undulation experiments    
    f_min, f_max, f_step = sweep_param.f[0], sweep_param.f[1], sweep_param.f[2] 
    f_arr = np.arange(f_min, f_max + 0.1*f_step, f_step)
    T_c_arr = 1.0 / f_arr

    lam_fit = fit_rikmenspoel_1978()[0]
    lam_arr = lam_fit(f_arr) 
            
    T_c_param = {'v_arr': T_c_arr.tolist(), 'round': 3, 'quantity': 'second'}    
    lam_param = {'v_arr': lam_arr.tolist(), 'round': 2}
    
    grid_param = {('T_c', 'lam'): (T_c_param, lam_param)} 

    PG = ParameterGrid(vars(model_param), grid_param)

    if sweep_param.save_to_storage:
        log_dir, sim_dir, sweep_dir = create_storage_dir()     
    else:
        from minimal_worm.experiments.undulation import sweep_dir, log_dir, sim_dir



    # Experiments are run using the Sweeper class for parallelization            
    if sweep_param.run:
        Sweeper.run_sweep(
            sweep_param.worker, 
            PG, 
            CS, 
            FK,
            log_dir, 
            sim_dir, 
            sweep_param.overwrite, 
            sweep_param.debug,
            'UExp')

    PG_filepath = PG.save(log_dir)
    print(f'Finished sweep! Save ParameterGrid to {PG_filepath}')

    # Pool and save simulation results to hdf5            
    filename = Path(
        f'raw_data_rikmenspoel'
        f'f_min={f_min}_f_max={f_max}_f_step={f_step}_'                
        f'const_A={sweep_parser.const_A}'
        f'phi={model_param.phi}_T={model_param.T}_'
        f'N={model_param.N}_dt={model_param.dt}.h5')
    
    h5_filepath = sweep_dir / filename

    if sweep_param.pool:        
        Sweeper.save_sweep_to_h5(PG, h5_filepath, sim_dir, FK, CK)
        
    # Analyse simulation results
    if sweep_param.analyse:
        analyse(h5_filepath, what_to_calculate=sweep_param)

    return

def sweep_f_lam_rikmenspoel(argv):
    '''
    Sweeps over
        - f: frequency
        - c: shape factor
        - lam: wavelength
    
    Frequency range is adjusted to experimental data.
    
    Do sperm with optimal wavelength?
    '''
    
    # Parse sweep parameter
    sweep_parser = default_sweep_parameter()    
            
    sweep_parser.add_argument('--f', 
        type=float, nargs=3, default = [10, 50, 5.0])    
    sweep_parser.add_argument('--lam', 
        type=float, nargs=3, default = [0.4, 2.0, 0.1])
    
    # The argumentparser for the sweep parameter has a boolean argument 
    # for ever frame key and control key which can be set to true
    # if it should be saved 
    sweep_param = sweep_parser.parse_known_args(argv)[0]    
    FK = [k for k in FRAME_KEYS if getattr(sweep_param, k)]
    CK = [k for k in CONTROL_KEYS if getattr(sweep_param, k)]

    A_fit = fit_rikmenspoel_1978()[2]

    # Parse model parameter
    model_parser = UndulationExperiment.parameter_parser()
    model_param = model_parser.parse_known_args(argv)[0]
    model_param.A = A_fit.coef.tolist()

    # Set material parameters to the experimental sperm data
    L0 = sperm_param_dict['L0']
    R = sperm_param_dict['R']
    B = sperm_param_dict['B']

    I = 0.25*np.pi*R**4
    E = B/I
    xi = 1e-3 * ureg.second
    eta = E * xi  
    
    model_param.L0 = L0.to_base_units()
    model_param.R = R.to_base_units()
    model_param.E = E.to_base_units()    
    model_param.eta = eta.to_base_units()    
    
    # Specify all dimensionless parameters which 
    # should be calculated from phyisical parameters                
    model_param.C_from_physical = True
    model_param.D_from_physical = True
    model_param.Y_from_physical = True        
    model_param.a_from_physical = True
    model_param.b_from_physical = True

    # Create the ParameterGrid over which we want to run
    # the undulation experiments    
    f_min, f_max, f_step = sweep_param.f[0], sweep_param.f[1], sweep_param.f[2] 

    T_c_param = {'v_min': f_min, 'v_max': f_max + 0.1*f_step, 
        'N': None, 'step': f_step, 'round': 3, 'inverse': True, 'quantity': 'second'}    
                
    lam_min, lam_max, lam_step = sweep_param.lam[0], sweep_param.lam[1], sweep_param.lam[2]

    lam_param = {'v_min': lam_min, 'v_max': lam_max + 0.1*lam_step, 
        'N': None, 'step': lam_step, 'round': 2}
    
    grid_param = {'T_c': T_c_param, 'lam': lam_param} 

    PG = ParameterGrid(vars(model_param), grid_param)

    if sweep_param.save_to_storage:
        log_dir, sim_dir, sweep_dir = create_storage_dir()     
    else:
        from minimal_worm.experiments.undulation import sweep_dir, log_dir, sim_dir

    # Experiments are run using the Sweeper class for parallelization            
    if sweep_param.run:
        Sweeper.run_sweep(
            sweep_param.worker, 
            PG, 
            UndulationExperiment.stw_va_control_sequence, 
            FK,
            log_dir, 
            sim_dir, 
            sweep_param.overwrite, 
            sweep_param.debug,
            'UExp')

    PG_filepath = PG.save(log_dir)
    print(f'Finished sweep! Save ParameterGrid to {PG_filepath}')

    # Pool and save simulation results to hdf5            
    filename = Path(
        f'raw_data_rikmenspoel_'
        f'f_min={f_min}_f_max={f_max}_f_step={f_step}_'                
        f'lam_min={lam_min}_lam_max={lam_max}_f_step={lam_step}_'                        
        f'phi={model_param.phi}_T={model_param.T}_'
        f'N={model_param.N}_dt={model_param.dt}.h5')
    
    h5_filepath = sweep_dir / filename

    if sweep_param.pool:        
        Sweeper.save_sweep_to_h5(PG, h5_filepath, sim_dir, FK, CK)
        
    # Analyse simulation results
    if sweep_param.analyse:
        analyse(h5_filepath, what_to_calculate=sweep_param)


def sweep_f_c_lam_rikmenspoel(argv):
    '''
    Sweeps over
        - f: frequency
        - c: shape factor
        - lam: wavelength
    
    Frequency range is adjusted to experimental data.
    
    Do sperm with optimal wavelength?
    '''
    
    # Parse sweep parameter
    sweep_parser = default_sweep_parameter()    
            
    sweep_parser.add_argument('--f', 
        type=float, nargs=3, default = [10, 50, 5.0])    
    sweep_parser.add_argument('--c', 
        type=float, nargs=3, default = [0.4, 1.6, 0.2])
    sweep_parser.add_argument('--lam', 
        type=float, nargs=3, default = [0.4, 2.0, 0.2])
    
    # The argumentparser for the sweep parameter has a boolean argument 
    # for ever frame key and control key which can be set to true
    # if it should be saved 
    sweep_param = sweep_parser.parse_known_args(argv)[0]    
    FK = [k for k in FRAME_KEYS if getattr(sweep_param, k)]
    CK = [k for k in CONTROL_KEYS if getattr(sweep_param, k)]

    # Parse model parameter
    model_parser = UndulationExperiment.parameter_parser()
    model_param = model_parser.parse_known_args(argv)[0]
    model_param.use_c = True

    # Set material parameters to the experimental sperm data
    L0 = sperm_param_dict['L0']
    R = sperm_param_dict['R']
    B = sperm_param_dict['B']

    I = 0.25*np.pi*R**4
    E = B/I
    xi = 1e-3 * ureg.second
    eta = E * xi  
    
    model_param.L0 = L0.to_base_units()
    model_param.R = R.to_base_units()
    model_param.E = E.to_base_units()    
    model_param.eta = eta.to_base_units()    
    
    # Specify all dimensionless parameters which 
    # should be calculated from phyisical parameters                
    model_param.C_from_physical = True
    model_param.D_from_physical = True
    model_param.Y_from_physical = True        
    model_param.a_from_physical = True
    model_param.b_from_physical = True

    # Create the ParameterGrid over which we want to run
    # the undulation experiments    
    f_min, f_max, f_step = sweep_param.f[0], sweep_param.f[1], sweep_param.f[2] 

    T_c_param = {'v_min': f_min, 'v_max': f_max + 0.1*f_step, 
        'N': None, 'step': f_step, 'round': 3, 'inverse': True, 'quantity': 'second'}    
                
    c_min, c_max, c_step = sweep_param.c[0], sweep_param.c[1], sweep_param.c[2]
    
    c_param = {'v_min': c_min, 'v_max': c_max + 0.1*c_step, 
        'N': None, 'step': c_step, 'round': 2}    

    lam_min, lam_max, lam_step = sweep_param.lam[0], sweep_param.lam[1], sweep_param.lam[2]

    lam_param = {'v_min': lam_min, 'v_max': lam_max + 0.1*lam_step, 
        'N': None, 'step': lam_step, 'round': 2}
    
    grid_param = {'T_c': T_c_param, 'c': c_param, 'lam': lam_param} 

    PG = ParameterGrid(vars(model_param), grid_param)

    if sweep_param.save_to_storage:
        log_dir, sim_dir, sweep_dir = create_storage_dir()     
    else:
        from minimal_worm.experiments.undulation import sweep_dir, log_dir, sim_dir

    # Experiments are run using the Sweeper class for parallelization            
    if sweep_param.run:
        Sweeper.run_sweep(
            sweep_param.worker, 
            PG, 
            UndulationExperiment.stw_control_sequence, 
            FK,
            log_dir, 
            sim_dir, 
            sweep_param.overwrite, 
            sweep_param.debug,
            'UExp')

    PG_filepath = PG.save(log_dir)
    print(f'Finished sweep! Save ParameterGrid to {PG_filepath}')

    # Pool and save simulation results to hdf5            
    filename = Path(
        f'raw_data_rikmenspoel_'
        f'f_min={f_min}_f_max={f_max}_f_step={f_step}_'                
        f'c_min={c_min}_c_max={c_max}_c_step={c_step}_'                
        f'lam_min={lam_min}_lam_max={lam_max}_f_step={lam_step}_'                        
        f'phi={model_param.phi}_T={model_param.T}_'
        f'N={model_param.N}_dt={model_param.dt}.h5')
    
    h5_filepath = sweep_dir / filename

    if sweep_param.pool:        
        Sweeper.save_sweep_to_h5(PG, h5_filepath, sim_dir, FK, CK)
        
    # Analyse simulation results
    if sweep_param.analyse:
        analyse(h5_filepath, what_to_calculate=sweep_param)

    return

if __name__ == '__main__':
        
    parser = ArgumentParser()
    parser.add_argument('-sweep',  
        choices = ['a_b', 'A_lam_a_b', 'c_lam_a_b', 'mu_c_lam_fang_yen', 
            'xi_mu_c_lam_fang_yen', 'C_c_lam', 'c_lam_a_b', 'C_a_b', 
            'c_lam', 'lam_a_b', 'c_a_b', 'C_mu_c_lam_fang_yen',
            'C_xi_mu_c_lam_fang_yen', 'C_xi_mu_fang_yen', 
            'f_rikmenspoel', 'f_lam_rikmenspoel', 'f_c_lam_rikmenspoel'], help='Sweep to run')
            
    # Run function passed via command line
    args = parser.parse_known_args(argv)[0]    
    globals()['sweep_' + args.sweep](argv)





