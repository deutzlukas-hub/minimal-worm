# Built-in
from pathlib import Path
from sys import argv
from argparse import ArgumentParser, BooleanOptionalAction
#from decimal import Decimal

# Third-party
import h5py
from parameter_scan import ParameterGrid
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import RectBivariateSpline

# Local imports
from minimal_worm import FRAME_KEYS
from minimal_worm.worm import CONTROL_KEYS
from minimal_worm.experiments import Sweeper
from minimal_worm.experiments.undulation import UndulationExperiment
from minimal_worm.experiments.undulation import log_dir, sweep_dir
from minimal_worm.experiments.undulation.analyse_sweeps import analyse

from minimal_worm.experiments.undulation.dirs import create_storage_dir


#================================================================================================
# utils
#================================================================================================

def load_data(filename):
    '''
    Loads hdf5 simulation file
    '''

    h5_filepath = sweep_dir / filename
    h5 = h5py.File(str(h5_filepath), 'r')
    PG_filepath = log_dir / h5.attrs['grid_filename']
    PG = ParameterGrid.init_pg_from_filepath(str(PG_filepath))

    return h5, PG

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
# Sweep parameter
#================================================================================================

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


def sweep_lam0_c0(argv):
    '''
    Sweeps over

    - fluid viscosity mu

    Frequency f, lam0 and A0 as a function of mu are determined from fit to Fang Yeng data
    '''

    # Parse sweep parameter
    sweep_parser = default_sweep_parameter()

    sweep_parser.add_argument('--lam0',
                              type=float, nargs=3, default=[0.5, 2.0, 0.1])
    sweep_parser.add_argument('--c0',
                              type=float, nargs=3, default=[0.4, 2.0, 0.1])

    sweep_param = sweep_parser.parse_known_args(argv)[0]

    #================================================================================================
    # Customize Model parameter
    #================================================================================================

    model_parser = UndulationExperiment.parameter_parser()
    model_param = model_parser.parse_known_args(argv)[0]

    model_param.Ds_h = 0.01
    model_param.Ds_t = 0.01
    model_param.s0_h = 0.05
    model_param.s0_t = 0.95
    model_param.T = 5.0
    model_param.use_c = True

    model_param.a_from_physical = True
    model_param.b_from_physical = True

    # Set xi
    log_xi = -1.73
    eta = 10 ** log_xi * model_param.E.magnitude
    model_param.eta = eta * model_param.eta.units

    # Print all model parameter whose value has been set via the command line
    cml_args = {k: v for k, v in vars(model_param).items()
                if v != model_parser.get_default(k)}

    if len(cml_args) != 0:
        print(cml_args)

    # ===============================================================================
    # Create Parameter Grid
    # ===============================================================================

    # Determine undualtion frequency from experimental fit
    f_sig_fit = fang_yen_fit()[1]
    T_c = 1.0 / f_sig_fit(np.log10(model_param.mu.magnitude))

    model_param.T_c = T_c * model_param.T_c.units

    # Waveform sweep
    c_min, c_max, c_step = sweep_param.c0[0], sweep_param.c0[1], sweep_param.c0[2]
    lam_min, lam_max, lam_step = sweep_param.lam0[0], sweep_param.lam0[1], sweep_param.lam0[2]

    c_param = {'v_min': c_min, 'v_max': c_max + 0.1 * c_step, 'N': None, 'step': c_step, 'round': 2}
    lam_param = {'v_min': lam_min, 'v_max': lam_max + 0.1 * lam_step, 'N': None, 'step': lam_step, 'round': 2}

    grid_param = {'c': c_param, 'lam': lam_param}
    PG = ParameterGrid(vars(model_param), grid_param)

    # =======================================================================
    # Run experiments
    # =======================================================================

    # Define which output paramter to save
    FK = [k for k in FRAME_KEYS if getattr(sweep_param, k)]
    CK = [k for k in CONTROL_KEYS if getattr(sweep_param, k)]

    # Set output path
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
        f'mu={model_param.mu.magnitude}_'
        f'lam_min={lam_min}_lam_max={lam_max}_lam_step={lam_step}_'
        f'c_min={c_min}_c_max={c_max}_c_step={c_step}_'
        f'E={np.round(np.log10(model_param.E.magnitude), 2)}_xi={np.round(log_xi, 2)}_'
        f'N={model_param.N}_dt={model_param.dt}_'
        f'T={model_param.T}.h5')

    h5_filepath = sweep_dir / filename

    if sweep_param.pool:
        Sweeper.save_sweep_to_h5(PG, h5_filepath, sim_dir, FK, CK)

    # ===============================================================================
    # Post analysis
    # ===============================================================================
    if sweep_param.analyse:
        sweep_param.A = True
        sweep_param.lam = True
        sweep_param.psi = True
        analyse(h5_filepath, what_to_calculate=sweep_param)

    return


def sweep_mu_lam0_c0(argv):
    '''
    Sweeps over

    - fluid viscosity mu

    Frequency f, lam0 and A0 as a function of mu are determined from fit to Fang Yeng data
    '''

    # Parse sweep parameter
    sweep_parser = default_sweep_parameter()

    sweep_parser.add_argument('--mu',
                              type=float, nargs=3, default=[-3, 1.2, 0.2])
    sweep_parser.add_argument('--lam0',
                              type=float, nargs=3, default=[0.5, 2.0, 0.1])
    sweep_parser.add_argument('--c0',
                              type=float, nargs=3, default=[0.4, 2.0, 0.1])

    sweep_param = sweep_parser.parse_known_args(argv)[0]

    #================================================================================================
    # Customize Model parameter
    #================================================================================================

    model_parser = UndulationExperiment.parameter_parser()
    model_param = model_parser.parse_known_args(argv)[0]

    model_param.Ds_h = 0.01
    model_param.Ds_t = 0.01
    model_param.s0_h = 0.05
    model_param.s0_t = 0.95
    model_param.T = 5.0
    model_param.use_c = True

    model_param.a_from_physical = True
    model_param.b_from_physical = True

    # Set xi
    log_xi = -1.73
    eta = 10 ** log_xi * model_param.E.magnitude
    model_param.eta = eta * model_param.eta.units

    # Print all model parameter whose value has been set via the command line
    cml_args = {k: v for k, v in vars(model_param).items() if v != model_parser.get_default(k)}

    if len(cml_args) != 0:
        print(cml_args)

    # ===============================================================================
    # Create Parameter Grid
    # ===============================================================================

    log_mu_min, log_mu_max, log_mu_step = sweep_param.mu[0], sweep_param.mu[1], sweep_param.mu[2]
    log_mu_arr = np.arange(log_mu_min, log_mu_max + 0.1 * log_mu_step, log_mu_step)
    mu_arr = 10 ** log_mu_arr
    # Determine undualtion frequency from experimental fit
    _, f_sig_fit, _ = fang_yen_fit()
    T_c_arr = 1.0 / f_sig_fit(log_mu_arr)

    c_min, c_max, c_step = sweep_param.c0[0], sweep_param.c0[1], sweep_param.c0[2]
    lam_min, lam_max, lam_step = sweep_param.lam0[0], sweep_param.lam0[1], sweep_param.lam0[2]

    c_param = {'v_min': c_min, 'v_max': c_max + 0.1 * c_step, 'N': None, 'step': c_step, 'round': 2}
    lam_param = {'v_min': lam_min, 'v_max': lam_max + 0.1 * lam_step, 'N': None, 'step': lam_step, 'round': 2}
    T_c_param = {'v_arr': T_c_arr.tolist(), 'round': 3, 'quantity': 'second'}
    mu_param = {'v_arr': mu_arr.tolist(), 'round': 6, 'quantity': 'pascal*second'}

    grid_param = {('T_c', 'mu'): (T_c_param, mu_param), 'c': c_param, 'lam': lam_param}
    PG = ParameterGrid(vars(model_param), grid_param)

    # =======================================================================
    # Run experiments
    # =======================================================================

    # Define which output paramter to save
    FK = [k for k in FRAME_KEYS if getattr(sweep_param, k)]
    CK = [k for k in CONTROL_KEYS if getattr(sweep_param, k)]

    # Set output path
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

    log_E = np.log10(PG.base_parameter['E'].magnitude)

    # Pool and save simulation results to hdf5
    filename = Path(
        f'raw_data_'
        f'mu_min={round(log_mu_min, 1)}_mu_max={round(log_mu_max, 1)}_mu_step={round(log_mu_step, 1)}_'
        f'lam_min={lam_min}_lam_max={lam_max}_lam_step={lam_step}_'
        f'c_min={c_min}_c_max={c_max}_c_step={c_step}_'
        f'E={np.round(log_E, 2)}_xi={np.round(log_xi, 2)}_'
        f'N={model_param.N}_dt={model_param.dt}_'
        f'T={model_param.T}.h5')

    h5_filepath = sweep_dir / filename

    if sweep_param.pool:
        Sweeper.save_sweep_to_h5(PG, h5_filepath, sim_dir, FK, CK)

    # ===============================================================================
    # Post analysis
    # ===============================================================================
    if sweep_param.analyse:
        sweep_param.A = True
        sweep_param.lam = True
        sweep_param.psi = True
        analyse(h5_filepath, what_to_calculate=sweep_param)

    return


def sweep_a_b(argv):

    sweep_parser = default_sweep_parameter()

    sweep_parser.add_argument('--a', type=float, nargs=3, default=[-2, 3, 0.2])
    sweep_parser.add_argument('--b', type=float, nargs=3, default=[-3, 0, 0.2])

    sweep_param = sweep_parser.parse_known_args(argv)[0]

    model_parser = UndulationExperiment.parameter_parser()
    model_param = model_parser.parse_known_args(argv)[0]

    # Customize parameter
    model_param.Ds_h = 0.01
    model_param.Ds_t = 0.01
    model_param.s0_h = 0.05
    model_param.s0_t = 0.95
    model_param.use_c = False
    model_param.T = 5.0

    #================================================================================================
    # Parameter Grid
    #================================================================================================

    a_min, a_max, a_step = sweep_param.a[0], sweep_param.a[1], sweep_param.a[2]
    b_min, b_max, b_step = sweep_param.b[0], sweep_param.b[1], sweep_param.b[2]

    # Choose preferred wavelength and amplitude such that model yields correct output wavenlength and amplitude
    h5_filename = 'analysis_mu_min=-3.0_mu_max=1.0_mu_step=0.2_lam_min=0.5_lam_max=2.0_lam_step=0.1_c_min=0.5_c_max=2.0_c_step=0.1_E=5.08_xi=-1.73_N=250_dt=0.01_T=5.0.h5'
    h5, PG = load_data(h5_filename)
    lam0_arr, c0_arr = PG.v_from_key('lam'), PG.v_from_key('c')
    lam0_arr_refine = np.linspace(lam0_arr.min(), lam0_arr.max(), 100*len(lam0_arr))
    c0_arr_refine = np.linspace(c0_arr.min(), c0_arr.max(), 100*len(c0_arr))
    log_mu_arr = np.arange(-3.0, 1.01, 0.2)

    # 1: Load sweep mu over lambda_0 and c_0
    mu = model_param.mu.magnitude
    log_mu = np.log10(mu)
    idx = (log_mu_arr - log_mu).argmin()

    lam, A = h5['lam'][idx, :], h5['A'][idx, :]
    lam_spline = RectBivariateSpline(lam0_arr, c0_arr, lam.T)
    A_spline = RectBivariateSpline(lam0_arr, c0_arr, A.T)
    lam = lam_spline(lam0_arr_refine, c0_arr_refine)
    A = A_spline(lam0_arr_refine, c0_arr_refine)

    # 3: Get target lambda and c from experiments
    lam_sig_fit, f_sig_fit, A_sig_fit = fang_yen_fit()
    lam_exp, A_exp = lam_sig_fit(log_mu), A_sig_fit(log_mu)

    idx_min = (np.abs(lam - lam_exp) / lam_exp + np.abs(A - A_exp) / A_exp).argmin()
    i_min, j_min = np.unravel_index(idx_min, A.shape)

    lam0_input = lam0_arr_refine[i_min]
    c0_input = c0_arr_refine[j_min]
    A0_input = 2 * np.pi * c0_input  / lam0_input

    # TODO: Do this plot
    # import matplotlib.pyplot as plt
    #
    # gs = plt.GridSpec(3,1)
    # ax0 = plt.subplot(gs[0])
    # ax1 = plt.subplot(gs[1])
    #
    # err_lam = np.abs(lam - lam_exp) / lam_exp
    #
    # CS = ax0.contourf(lam0_arr_refine, c0_arr_refine, err_lam.T, cmap = 'plasma')
    # ax0.contour(lam0_arr_refine, c0_arr_refine, err_lam, colors = 'k')
    # ax0.plot(lam0_input, c0_input, 'o', c='r')
    # plt.colorbar(CS, ax=ax0)
    #
    # err_A = np.abs(A - A_exp) / A_exp
    #
    # CS = ax1.contourf(lam0_arr_refine, c0_arr_refine, err_A.T, cmap = 'plasma')
    # ax1.contour(lam0_arr_refine, c0_arr_refine, err_A.T, colors = 'k')
    # ax1.plot(lam0_input, c0_input, 'o', c='r')
    # plt.colorbar(CS, ax=ax1)
    #
    # ax2 = plt.subplot(gs[2])
    #
    # err = err_lam + err_A
    #
    # CS = ax2.contourf(lam0_arr_refine, c0_arr_refine, np.log10(err.T), cmap = 'cividis')
    # ax2.contour(lam0_arr_refine, c0_arr_refine, np.log10(err.T), colors = 'k')
    # ax2.plot(lam0_input, c0_input, 'o', c='r')
    # plt.colorbar(CS, ax=ax2)
    #
    # plt.show()

    model_param.lam, model_param.A = lam0_input, A0_input

    a_param = {'v_min': a_min, 'v_max': a_max + 0.1 * a_step, 'N': None, 'step': a_step, 'round': 4, 'log': True}
    b_param = {'v_min': b_min, 'v_max': b_max + 0.1 * b_step, 'N': None, 'step': b_step, 'round': 5, 'log': True}

    grid_param = {'a': a_param, 'b': b_param}

    PG = ParameterGrid(vars(model_param), grid_param)

    #================================================================================================
    # Run simulation
    #================================================================================================

    # Decide what to save
    FK = [k for k in FRAME_KEYS if getattr(sweep_param, k)]

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


def sweep_mu_a_b(argv):

    sweep_parser = default_sweep_parameter()

    sweep_parser.add_argument('--a', type=float, nargs=3, default=[-2, 3, 0.2])
    sweep_parser.add_argument('--b', type=float, nargs=3, default=[-3, 0, 0.2])

    sweep_param = sweep_parser.parse_known_args(argv)[0]

    model_parser = UndulationExperiment.parameter_parser()
    model_param = model_parser.parse_known_args(argv)[0]

    # Customize parameter
    model_param.Ds_h = 0.01
    model_param.Ds_t = 0.01
    model_param.s0_h = 0.05
    model_param.s0_t = 0.95
    model_param.use_c = False
    model_param.T = 5.0

    #================================================================================================
    # Parameter Grid
    #================================================================================================

    a_min, a_max, a_step = sweep_param.a[0], sweep_param.a[1], sweep_param.a[2]
    b_min, b_max, b_step = sweep_param.b[0], sweep_param.b[1], sweep_param.b[2]

    # Choose preferred wavelength and amplitude such that model yields correct output wavenlength and amplitude
    h5_filename = 'analysis_mu_min=-3.0_mu_max=1.0_mu_step=0.2_lam_min=0.5_lam_max=2.0_lam_step=0.1_c_min=0.5_c_max=2.0_c_step=0.1_E=5.08_xi=-1.73_N=250_dt=0.01_T=5.0.h5'
    h5, PG = load_data(h5_filename)

    # 1: Load sweep mu over lambda_0 and c_0
    mu_arr = PG.v_from_key('mu')
    log_mu_arr = np.log10(mu_arr)
    log_mu_min, log_mu_max  = log_mu_arr[0], log_mu_arr[1]
    log_mu_step = log_mu_max - log_mu_min

    lam0_arr, c0_arr = PG.v_from_key('lam'), PG.v_from_key('c')

    lam0_arr_refine = np.linspace(lam0_arr.min(), lam0_arr.max(), 100*len(lam0_arr))
    c0_arr_refine = np.linspace(c0_arr.min(), c0_arr.max(), 100*len(c0_arr))

    A_mat, lam_mat = h5['A'][:], h5['lam'][:]

    # 3: Get target lambda and c from experiments
    lam_sig_fit, f_sig_fit, A_sig_fit = fang_yen_fit()
    lam_exp_arr, A_exp_arr = lam_sig_fit(log_mu_arr), A_sig_fit(log_mu_arr)

    # 4: Get input lambda_0 and c_0 such that output lambda and c yield
    lam0_arr_input = np.zeros_like(log_mu_arr)
    A0_arr_input = np.zeros_like(log_mu_arr)

    for i, (lam_exp, A_exp) in enumerate(zip(lam_exp_arr, A_exp_arr)):

        lam, A = lam_mat[i, :], A_mat[i, :]

        A_spline = RectBivariateSpline(lam0_arr, c0_arr, A.T)
        lam_spline = RectBivariateSpline(lam0_arr, c0_arr, lam.T)

        A = A_spline(lam0_arr_refine, c0_arr_refine)
        lam = lam_spline(lam0_arr_refine, c0_arr_refine)

        idx_min = (np.abs(lam - lam_exp) + np.abs(A - A_exp)).argmin()
        i_min, j_min = np.unravel_index(idx_min, A.shape)

        lam0_arr_input[i] = lam0_arr_refine[i_min]
        A0_arr_input[i] = 2 * np.pi * c0_arr_refine[j_min] / lam0_arr_input[i]

    lam0_param = {'v_arr': lam0_arr_input.tolist(), 'round': 2}
    A0_param = {'v_arr': A0_arr_input.tolist(), 'round': 2}

    a_param = {'v_min': a_min, 'v_max': a_max + 0.1 * a_step, 'N': None, 'step': a_step, 'round': 4, 'log': True}
    b_param = {'v_min': b_min, 'v_max': b_max + 0.1 * b_step, 'N': None, 'step': b_step, 'round': 5, 'log': True}

    grid_param = {('lam', 'A'): (lam0_param, A0_param), 'a': a_param, 'b': b_param}

    PG = ParameterGrid(vars(model_param), grid_param)

    #================================================================================================
    # Run simulation
    #================================================================================================

    # Decide what to save
    FK = [k for k in FRAME_KEYS if getattr(sweep_param, k)]

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

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--sweep', choices=['lam0_c0', 'mu_lam0_c0', 'a_b', 'mu_a_b'], help='Sweep to run')

    # Run function passed via command line
    args = parser.parse_known_args(argv)[0]
    globals()['sweep_' + args.sweep](argv)


