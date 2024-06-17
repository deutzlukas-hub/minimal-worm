'''
Created on 15 Jun 2023

@author: amoghasiddhi
'''
# Built-in
from pathlib import Path
from sys import argv
from argparse import ArgumentParser, BooleanOptionalAction
# from decimal import Decimal

# Third-party
from parameter_scan import ParameterGrid
import pint
import numpy as np
from scipy.optimize import curve_fit

# Local imports
from minimal_worm import FRAME_KEYS
from minimal_worm.worm import CONTROL_KEYS
from minimal_worm.experiments import Sweeper
from minimal_worm.experiments.undulation import UndulationExperiment
from minimal_worm.experiments.undulation import create_storage_dir
from minimal_worm.experiments.undulation.analyse_sweeps import analyse

ureg = pint.UnitRegistry()


def default_sweep_parameter():
    '''
    Default sweep hyper parameter
    '''
    parser = ArgumentParser(description='sweep-parameter', allow_abbrev=False)

    parser.add_argument('--worker', type=int, default=10,
                        help='Number of processes')
    parser.add_argument('--run', action=BooleanOptionalAction, default=True,
                        help='If true, sweep is run. Set to false if sweep has already been run and cached.')
    parser.add_argument('--pool', action=BooleanOptionalAction, default=True,
                        help='If true, FrameSequences are pickled to disk')
    parser.add_argument('--analyse', action=BooleanOptionalAction, default=True,
                        help='If true, analyse pooled raw data')
    parser.add_argument('--overwrite', action=BooleanOptionalAction, default=False,
                        help='If true, already existing simulation results are overwritten')
    parser.add_argument('--debug', action=BooleanOptionalAction, default=False,
                        help='If true, exception handling is turned off which is helpful for debugging')
    parser.add_argument('--save_to_storage', action=BooleanOptionalAction, default=False,
                        help='If true, results are saved to external storage filesystem specified in dirs.py')

    # Frame keys
    parser.add_argument('--t', action=BooleanOptionalAction, default=True,
                        help='If true, save time stamps')
    parser.add_argument('--r', action=BooleanOptionalAction, default=True,
                        help='If true, save centreline coordinates')
    parser.add_argument('--theta', action=BooleanOptionalAction, default=True,
                        help='If true, save Euler angles')
    parser.add_argument('--d1', action=BooleanOptionalAction, default=False,
                        help='If true, save director 1')
    parser.add_argument('--d2', action=BooleanOptionalAction, default=False,
                        help='If true, save director 2')
    parser.add_argument('--d3', action=BooleanOptionalAction, default=False,
                        help='If true, save director 3')
    parser.add_argument('--k', action=BooleanOptionalAction, default=True,
                        help='If true, save curvature')
    parser.add_argument('--sig', action=BooleanOptionalAction, default=True,
                        help='If true, saved shear-stretch')
    parser.add_argument('--k_norm', action=BooleanOptionalAction, default=True,
                        help='If true, save L2 norm of the real and preferred curvature difference')
    parser.add_argument('--sig_norm', action=BooleanOptionalAction, default=True,
                        help='If true, save L2 norm of the real and preferred shear-stretch difference')

    # Velocity keys
    parser.add_argument('--r_t', action=BooleanOptionalAction, default=True,
                        help='If trues, save centreline velocity')
    parser.add_argument('--w', action=BooleanOptionalAction, default=False,
                        help='If true, save angular velocity')
    parser.add_argument('--k_t', action=BooleanOptionalAction, default=False,
                        help='If true, save curvature strain rate')
    parser.add_argument('--sig_t', action=BooleanOptionalAction, default=False,
                        help='If true, save shear-stretch strain rate')

    # Forces and torque keys
    parser.add_argument('--f_F', action=BooleanOptionalAction, default=False,
                        help='If true, save fluid drag force density')
    parser.add_argument('--l_F', action=BooleanOptionalAction, default=False,
                        help='If true, save fluid drag torque density')
    parser.add_argument('--f_M', action=BooleanOptionalAction, default=False,
                        help='If true, save muscle force density')
    parser.add_argument('--l_M', action=BooleanOptionalAction, default=False,
                        help='If true, save muscle torque density')
    parser.add_argument('--N_force', action=BooleanOptionalAction, default=False,
                        dest='N', help='If true, save internal force resultant')
    parser.add_argument('--M_torque', action=BooleanOptionalAction, default=False,
                        dest='M', help='If true, save internal torque resultant')

    # Power keys
    parser.add_argument('--D_F_dot', action=BooleanOptionalAction, default=True,
                        help='If true, save fluid dissipation rate')
    parser.add_argument('--D_I_dot', action=BooleanOptionalAction, default=True,
                        help='If true, save internal dissipation rate')
    parser.add_argument('--W_dot', action=BooleanOptionalAction, default=True,
                        help='If true, save mechanical muscle power')
    parser.add_argument('--V_dot', action=BooleanOptionalAction, default=True,
                        help='If true, save potential rate')
    parser.add_argument('--V', action=BooleanOptionalAction, default=False,
                        help='If true, save potential')

    # Control key
    parser.add_argument('--k0', action=BooleanOptionalAction, default=True,
                        help='If true, save controls')
    parser.add_argument('--sig0', action=BooleanOptionalAction, default=True,
                        help='If true, save controls')

    # Analyse
    parser.add_argument('--calc_R', action=BooleanOptionalAction, default=False,
                        dest='R', help='If true, calculate final position of centroid')
    parser.add_argument('--calc_U', action=BooleanOptionalAction, default=True,
                        dest='U', help='If true, calculate swimming speed U')
    parser.add_argument('--calc_E', action=BooleanOptionalAction, default=True,
                        dest='E', help='If true, calculate L2 norm between real and preferred curvature')
    parser.add_argument('--calc_A', action=BooleanOptionalAction, default=False,
                        dest='A', help='If true, calculate real curvature amplitude')
    parser.add_argument('--calc_f', action=BooleanOptionalAction, default=False,
                        dest='f', help='If true, calculate undulation frequency')
    parser.add_argument('--calc_lag', action=BooleanOptionalAction, default=False,
                        dest='lag', help='If true, calculate lag')
    parser.add_argument('--calc_lam', action=BooleanOptionalAction, default=False,
                        dest='lam', help='If true, calculate undulation wavelength')
    parser.add_argument('--calc_psi', action=BooleanOptionalAction, default=False,
                        dest='psi', help='If true, calculate angle of attack')
    parser.add_argument('--calc_Y', action=BooleanOptionalAction, default=False,
                        dest='Y', help='If true, calculate wobbling speed')
    parser.add_argument('--calc_fp', action=BooleanOptionalAction, default=False,
                        dest='fp', help='If true, calculate propulsion force')
    return parser

#================================================================================================
# Experimental data
#================================================================================================

def fang_yen_data(return_theta=False):
    '''
    Undulation wavelength, amplitude, frequencies for different viscosities
    from Fang-Yen 2010 paper
    '''
    # Experimental Fang Yeng 2010
    mu_arr = 10 ** (np.array([0.000, 0.966, 2.085, 2.482, 2.902, 3.142, 3.955, 4.448]) - 3)  # Pa*s
    lam_arr = np.array([1.516, 1.388, 1.328, 1.239, 1.032, 0.943, 0.856, 0.799])
    f_arr = [1.761, 1.597, 1.383, 1.119, 0.790, 0.650, 0.257, 0.169]  # Hz
    A_arr = [2.872, 3.126, 3.290, 3.535, 4.772, 4.817, 6.226, 6.735]

    theta_max = [44.11, 41.25, 42.13, 41.79, 44.58, 45.46, 54.43, 55.65]  # degrees

    if return_theta:
        return mu_arr, theta_max

    return mu_arr, lam_arr, f_arr, A_arr

def fang_yen_fit(return_param=False):
    '''
    Fit sigmoids to fang yen data
    '''
    mu_arr, lam_arr, f_arr, A_arr = fang_yen_data()

    log_mu_arr = np.log10(mu_arr)

    # Define the sigmoid function
    def sigmoid(x, a, b, c, d):
        y = a / (1 + np.exp(-c * (x - b))) + d
        return y

    # Fit the sigmoid to wavelength
    popt_lam, _ = curve_fit(sigmoid, log_mu_arr, lam_arr)
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

    if not return_param:
        return lam_sig_fit, f_sig_fit, A_sig_fit
    return lam_sig_fit, f_sig_fit, A_sig_fit, popt_lam, popt_f, popt_A

#================================================================================================
# Predict range
#================================================================================================

def predict_required_a_b_range():

    model_parser = UndulationExperiment.parameter_parser()
    model_param = model_parser.parse_known_args()[0]

    E = model_param.E
    L0 = model_param.L0
    R = model_param.R
    I = 0.25 * np.pi * R**4
    c_t = model_param.c_t

    lam_sig_fit, f_sig_fit, A_sig_fit = fang_yen_fit()
    mu_arr = np.logspace(-3, 1, int(1e2))
    f_arr = f_sig_fit(np.log10(mu_arr))
    f_arr = f_arr / model_param.T_c.units

    mu_arr = mu_arr * model_param.mu.units
    tau_arr = c_t * mu_arr * L0**4 / (E * I)
    a_arr = tau_arr * f_arr

    xi = 10**(-1.73) * model_param.T_c.units
    b_arr = xi * f_arr

    log_a_arr = np.log10(a_arr)
    log_b_arr = np.log10(b_arr)

    print(f'log_a_min = {log_a_arr.min()}')
    print(f'log_a_max = {log_a_arr.max()}')
    print(f'log_b_min = {log_b_arr.min()}')
    print(f'log_b_max = {log_b_arr.max()}')

    return

def sweep_mu_a_b(argv):
    '''
    Parameter sweep over time scale ratios a and b

    Show that swimming speed and energy are fully captured
    by the system input time scale ratios.
    '''

    # Parse sweep parameter
    sweep_parser = default_sweep_parameter()

    sweep_parser.add_argument('--mu',
                              type=float, nargs=3, default=[-3, 1, 0.2])
    sweep_parser.add_argument('--a',
                              type=float, nargs=3, default=[-2, 3, 0.1])
    sweep_parser.add_argument('--b',
                              type=float, nargs=3, default=[-3, 0, 0.1])

    sweep_param = sweep_parser.parse_known_args(argv)[0]

    # Print keys that will be saved
    FK = [k for k in FRAME_KEYS if getattr(sweep_param, k)]
    CK = [k for k in CONTROL_KEYS if getattr(sweep_param, k)]

    # Parse model parameter
    model_parser = UndulationExperiment.parameter_parser()
    model_param = model_parser.parse_known_args(argv)[0]

    # Customize parameter
    model_param.Ds_h = 0.01
    model_param.Ds_t = 0.01
    model_param.s0_h = 0.05
    model_param.s0_t = 0.95
    model_param.use_c = False
    model_param.T = 5.0

    # Print all model parameter whose defaults have been overwritten via the command line
    cml_args = {k: v for k, v in vars(model_param).items()
                if v != model_parser.get_default(k)}

    if len(cml_args) != 0:
        print(cml_args)

    # Create the ParameterGrid over which we want to run
    # the undulation experiments
    mu_min, mu_max = sweep_param.mu[0], sweep_param.mu[1]
    mu_step = sweep_param.mu[2]

    a_min, a_max = sweep_param.a[0], sweep_param.a[1]
    a_step = sweep_param.a[2]

    b_min, b_max = sweep_param.b[0], sweep_param.b[1]
    b_step = sweep_param.b[2]

    log_mu_arr = np.arange(mu_min, mu_max, mu_step)

    lam_sig_fit, f_sig_fit, A_sig_fit =  fang_yen_fit()

    lam_arr = lam_sig_fit(log_mu_arr)
    A_arr = A_sig_fit(log_mu_arr)

    lam_param = {'v_arr': lam_arr.tolist(), 'round': 3}
    A_param = {'v_arr': A_arr.tolist(), 'round': 3}

    a_param = {'v_min': a_min, 'v_max': a_max + 0.1 * a_step,
               'N': None, 'step': a_step, 'round': 4, 'log': True}

    b_param = {'v_min': b_min, 'v_max': b_max + 0.1 * b_step,
               'N': None, 'step': b_step, 'round': 5, 'log': True}

    grid_param = {('lam', 'A'): (lam_param, A_param), 'a': a_param, 'b': b_param}

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
        f'mu_min={mu_min}_mu_max={mu_max}_mu_step={mu_step}_'
        f'a_min={a_min}_a_max={a_max}_a_step={a_step}_'
        f'b_min={b_min}_b_max={b_max}_b_step={b_step}_'
        f'E={np.log10(model_param.E.magnitude):.2f}_xi={model_param.E.magnitude}'
        f'N={model_param.N}_dt={model_param.dt}_'
        f'T={model_param.T}.h5')

    h5_filepath = sweep_dir / filename

    if sweep_param.pool:
        Sweeper.save_sweep_to_h5(PG, h5_filepath, sim_dir, FK, CK)

    if sweep_param.analyse:
        analyse(h5_filepath, what_to_calculate=sweep_param)
    return

if __name__ == '__main__':

    # predict_required_a_b_range()

    parser = ArgumentParser()
    parser.add_argument('-sweep',
                        choices=['mu_a_b'], help='Sweep to run')

    # Run function passed via command line
    args = parser.parse_known_args(argv)[0]
    globals()['sweep_' + args.sweep](argv)






