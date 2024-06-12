'''
Created on 12 Jun 2023

@author: amoghasiddhi
'''
from fenics import *
import numpy as np
import matplotlib.pyplot as plt

from parameter_scan import ParameterGrid

from minimal_worm.util import f2n
from minimal_worm.experiments import PostProcessor
from minimal_worm.experiments.undulation import UndulationExperiment
from minimal_worm import ModelParameter, Worm

e1 = Constant((1, 0, 0))
e2 = Constant((0, 1, 0))
e3 = Constant((0, 0, 1))

#------------------------------------------------------------------------------ 
# Finite difference approximation used in weak form
    
def grad(f):
    return Dx(f, 0)
        
def _finite_difference_coefficients(n, k):
    '''
    Calculates weighting coefficients for finite backwards 
    difference of order k for nth derivative
    '''
    
    # Number points for required for nth derivative of kth order 
    N = n + k

    # point indexes [-k, ..., -1, 0]
    # 0 = current, -1 = previous time point, ...        
    s_arr = np.arange(-N+1, 1)
    _A = np.vander(s_arr, increasing=True).T
    b = np.zeros(N)
    b[n] = np.math.factorial(n)

    # Weighting coefficients correspond to 
    # points as indexed by s_arr 
    c_arr = np.linalg.solve(_A, b)
    # Fenics can't handle numpy floats
    c_arr = c_arr.tolist()
    
    return c_arr, s_arr

def _finite_backwards_difference(
        n,
        k,
        z, 
        z_old_arr,
        dt):
    """Approximate nth derivative by finite backwards difference
    of order k of given fenics function z"""

    c_arr, s_arr = _finite_difference_coefficients(n, k)

    z_t = 0        
    # Add terms to finite backwards difference 
    # from most past to most recent time point
    for s, c in zip(s_arr, c_arr):
        if s == 0:  
            z_t += c * z            
        else:
            z_t += c * z_old_arr[s]

    z_t = z_t / dt**n

    return z_t

def _Q(theta):
    '''
    Matrix Q rotates lab frame to the body frames
    '''
    
    alpha, beta, gamma = theta[0], theta[1], theta[2] 

    R_x = as_matrix(
        [[1, 0, 0], 
         [0, cos(gamma), -sin(gamma)], 
         [0, sin(gamma), cos(gamma)]]
    )
    R_y = as_matrix(
        [[cos(beta), 0, sin(beta)], 
         [0, 1, 0], 
         [-sin(beta), 0, cos(beta)]]
    )
    
    R_z = as_matrix(
        [[cos(alpha), -sin(alpha), 0], 
         [sin(alpha), cos(alpha), 0], 
         [0, 0, 1]]
    )        

    return R_z * R_y * R_x     

def T(r):
        x, y, z = r[0], r[1], r[2] 
                    
        # Cross product matrix
    
        return as_matrix(
            [[0, -grad(z), grad(y)], 
             [grad(z), 0, -grad(x)], 
             [-grad(y), grad(x), 0]]
        )


def _A(theta):
    """The matrix A is used to calculate the curvature k and
    angular velocity w from first Euler angle derivatives
    with respect to s and t"""

    alpha, beta, _ = split(theta)

    A = as_matrix(
        [
            [0, sin(alpha), -cos(alpha) * cos(beta)],
            [0, -cos(alpha), -sin(alpha) * cos(beta)],
            [-1, 0, sin(beta)],
        ]
    )
    return A

def _A_t(theta, theta_t): #alpha_t, beta_t):
    """Time derivative of matrix _A is used to calculate the 
    time derivative of the curvature vector k"""

    alpha, beta = theta[0], theta[1]
    alpha_t, beta_t = theta_t[0], theta_t[1] 

    A_t = as_matrix(
        [
            [
                0,
                cos(alpha) * alpha_t,
                sin(alpha) * cos(beta) * alpha_t - cos(alpha) * sin(beta) * beta_t,
            ],
            [
                0,
                sin(alpha) * alpha_t,
                - cos(alpha) * cos(beta) * alpha_t + sin(alpha) * sin(beta) * beta_t ,
            ],
            [0, 0, cos(beta) * beta_t],
        ]
    )      
  
    return A_t

def solve_weak_form(param, CS):


    N = param.N
    dt = param.dt

    MP = ModelParameter(param)
    sig_pref, k_pref, t_pref = CS['sig'], CS['k'], CS['t']
    
    C, D, Y, S, S_tilde, B, B_tilde = MP.to_fenics()
    
    mesh = UnitIntervalMesh(N - 1)
    
    # Finite elements for 1 dimensional spatial coordinate s        
    P1 = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
            
    # State variables r and theta are 3 dimensional vector-valued functions of s                        
    P1_3 = MixedElement([P1] * 3)
    
    # Function space for scalar functions of s
    V = FunctionSpace(mesh, P1)
    # Function space for 3 component vector-valued functions of s        
    V3 = FunctionSpace(mesh, P1_3)
    
    # Trial function space for 6 component vector-valued function composed of r and theta
    W = FunctionSpace(mesh, MixedElement(P1_3, P1_3))
    
    u = TrialFunction(W)
    phi = TestFunction(W)
    
    r, theta = split(u)
    phi_r, phi_theta = split(phi)
    
    u_h = Function(W)
    
    r_h, theta_h = split(u_h)
    
    u_old_arr = [Function(W), Function(W)] 
    
    r0 = Function(V3)                
    theta0 = Function(V3)
    r0.assign(Expression(("0", "0", "x[0]"), degree=1))
    theta0.assign(Expression(("0", "0", "0"), degree=1))
    
    fa = FunctionAssigner(W, [V3, V3])
    
    r_old_arr = []
    theta_old_arr = []
    
    for u_old in u_old_arr:
        fa.assign(u_old, [r0, theta0])
        
        r_old, theta_old = split(u_old)
        r_old_arr.append(r_old)
        theta_old_arr.append(theta_old)
            
    r_t = _finite_backwards_difference(1, 2, r, r_old_arr, dt)
    theta_t = _finite_backwards_difference(1, 2, theta, theta_old_arr, dt)
        
    A_h = _A(theta_h)
    A_h_t = _A_t(theta_h, theta_t)
    Q_h = _Q(theta_h)
    T_h = T(r_h)
        
    # length-element
    eps_h = sqrt(dot(grad(r_h), grad(r_h)))
    
    # angular velocity vector
    w = A_h * theta_t
    # shear/stretch vector
    sig = Q_h * grad(r) - e3
    # time derivative shear stretch vector
    sig_t = Q_h * grad(r_t) - cross(w, Q_h * grad(r_h))
        
    # generalized curvature
    k = A_h * grad(theta) # / eps_h
    # time derivative generalized curvature
    # k_t = A_h * grad(theta_t) / eps_h + A_h_t * grad(theta_h) / eps_h
    k_t = A_h * grad(theta_t) + A_h_t * grad(theta_h)

    # internal force
    n = Q_h.T * (S * (sig - sig_pref) + S_tilde * sig_t)
    # internal torque
    m = Q_h.T * (B * (k - k_pref) + B_tilde * k_t)
    
    # external fluid drag force
    d3_h = Q_h.T * e3
    d3d3_h = outer(d3_h, d3_h)
    f_F = -(d3d3_h + C * (Identity(3) - d3d3_h)) * r_t
    # external fluid drag torque
    e3e3 = outer(e3, e3)
    l_F = -Q_h.T*D*(e3e3 + Y * (Identity(3) - e3e3)) * w
        
    # linear balance
    eq1 = dot(f_F, phi_r) * dx - dot(n, grad(phi_r)) * dx
    # angular balance
    eq2 = (
        dot(l_F, phi_theta) * dx
        + dot(T_h * n, phi_theta) * dx
        - dot(m, grad(phi_theta)) * dx
    )
    
    equation = eq1 + eq2
    
    F_op, L = lhs(equation), rhs(equation)
          
    t_arr = np.arange(dt, param.T + 0.1*dt, dt)
    
    r_mat = np.zeros((len(t_arr), 3, N)) 
    theta_mat = np.zeros((len(t_arr), 3, N))
    
    V_dot_arr = np.zeros(len(t_arr))
    D_I_dot_arr = np.zeros(len(t_arr))
    D_F_dot_arr = np.zeros(len(t_arr))
    W_dot_arr = np.zeros(len(t_arr))
    
    for i, t in enumerate(t_arr):
        
        print(f't={t:.5f}')
        
        t_pref.assign(t)
        
        u_h.assign(u_old)    
        u = Function(W)
        solve(F_op == L, u)
        assert not np.isnan(u.vector().get_local()).any(), (
        f'Solution at t={t} contains nans!')    
    
        r, theta = u.split(deepcopy=True)

        r_mat[i, :] = f2n(r)
        theta_mat[i, :] = f2n(theta)
        
        #------------------------------------------------------------------------------ 
        # Compute powers
                
        r_old_arr = []
        theta_old_arr = []
    
        for u_old in u_old_arr:        
            r_old, theta_old = split(u_old)
            r_old_arr.append(r_old)
            theta_old_arr.append(theta_old)
                
        r_t = _finite_backwards_difference(1, 2, r, r_old_arr, dt)
        theta_t = _finite_backwards_difference(1, 2, theta, theta_old_arr, dt)
        
        Q = _Q(theta)
        A = _A(theta)
        A_t = _A_t(theta, theta_t) 
                        
        w_bar = A * theta_t
        k_bar = A * grad(theta)
        sig_bar = Q * grad(r) - e3
                
        k_bar_t = A * grad(theta_t) + A_t * grad(theta) 
        sig_bar_t = Q * grad(r_t) - cross(w_bar, Q * grad(r))

        d3 = Q.T * e3
        d3d3 = outer(d3, d3)
        f_F = -(d3d3_h + C * (Identity(3) - d3d3)) * r_t        
        l_F_bar = -D*(e3e3 + Y * (Identity(3) - e3e3)) * w_bar
                                                                                        
        f_M_bar = -grad(project(S * sig_pref, V3))
        l_M_bar = -grad(project(B * k_pref, V3))
                                        
        V_dot = assemble((dot(k_bar, B * k_bar_t) + dot(sig_bar, S * sig_bar_t)) * dx)                        
        D_I_dot = -assemble((dot(k_bar_t, B_tilde * k_bar_t) + dot(sig_bar_t, S_tilde * sig_bar_t)) * dx)
        D_F_dot = assemble((dot(f_F, r_t) + dot(l_F_bar, w_bar)) * dx)
        W_dot = assemble((dot(f_M_bar, Q*r_t) + dot(l_M_bar, w_bar)) * dx)

        V_dot_arr[i] = V_dot
        D_I_dot_arr[i] = D_I_dot
        D_F_dot_arr[i] = D_F_dot
        W_dot_arr[i] = W_dot
                                                                        
        for n, u_n in enumerate(u_old_arr[:-1]):
            u_n.assign(u_old_arr[n + 1])
                                
        u_old_arr[-1].assign(u)
                  
    powers = {}
    powers['V_dot'] = V_dot_arr
    powers['D_I_dot'] = D_I_dot_arr
    powers['D_F_dot'] = D_F_dot_arr
    powers['W_dot'] = W_dot_arr
                               
    return r_mat, theta_mat, powers, t_arr
           
def test_zero_control():
    
    sig_pref = Constant((0,0,0))
    k_pref = Constant((0,0,0))

    r_list, theta_list = solve_weak_form(sig_pref, k_pref)
    
    N = r_list[0].shape[1]
    r_tar_arr = np.zeros((3, N))
    r_tar_arr[2, :] = np.linspace(0, 1, N)
    
    for r_arr, theta_arr in zip(r_list, theta_list):    
        
        assert np.allclose(r_arr, r_arr)
        assert np.allclose(theta_arr, 0)

    print('Passed solve weak form for zero controls test! ')
            
    return
    
def test_undulation():
    
    parser = UndulationExperiment.parameter_parser()
    param = parser.parse_args()            
    CS = UndulationExperiment.stw_control_sequence(param)
    
    r_mat, t_arr = solve_weak_form(param, CS)

    # Midpoint 2d trajectory
    r_mp = r_mat[:, 1:, r_mat.shape[2] // 2]
    # 2d centre of mass trajectory
    r_com = r_mat[:, 1:].mean(axis=2)    
    # Swimming speed
    U_avg, U, t = PostProcessor.comp_mean_swimming_speed(r_mat, t_arr)

    gs = plt.GridSpec(2, 2)
    
    ax00 = plt.subplot(gs[0,0])
    ax01 = plt.subplot(gs[0,1])
    ax10 = plt.subplot(gs[1,0])
    ax11 = plt.subplot(gs[1,1])
    
    # plot_scalar_field(ax00, CS.k[:, 0, :])
    # plot_scalar_field(ax01, FS.k[:, 0, :])
    
    ax10.plot(r_mp[:, 0], r_mp[:, 1])
    ax10.plot(r_com[:, 0], r_com[:, 1])
    
    ax11.plot(t, U)
    ax11.plot([t[0], t[-1]], [U_avg, U_avg])
    
    plt.show()

def test_power_balance():
    
    parser = UndulationExperiment.parameter_parser()
    param = parser.parse_args()            
    
    param.dt = 1e-2
    param.N = 250
        
    CS = UndulationExperiment.stw_control_sequence(param)

    r_mat, powers, t_arr = solve_weak_form(param, CS)

    gs = plt.GridSpec(2, 1)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    
    V_dot, D_I_dot = powers['V_dot'], powers['D_I_dot']
    D_F_dot, W_dot = powers['D_F_dot'], powers['W_dot']
    
    ax0.plot(t_arr, V_dot, label = r'$\dot{V}$')
    ax0.plot(t_arr, D_I_dot, label = r'$\dot{D}_I$')
    ax0.plot(t_arr, D_F_dot, label = r'$\dot{D}_F$')
    ax0.plot(t_arr, W_dot, label = r'$\dot{W}$')

    ax0.legend()

    ax1.plot(V_dot - D_I_dot - D_F_dot - W_dot)

    plt.show()

    return

def test_swimming_speed():
    

    a_min, a_max = -3, 2
    a_step = 0.2

    b_min, b_max = -3, 0
    b_step = 0.2

    a_param = {'v_min': a_min, 'v_max': a_max + 0.1*a_step, 
        'N': None, 'step': a_step, 'round': 4, 'log': True}    

    b_param = {'v_min': b_min, 'v_max': b_max + 0.1*b_step, 
        'N': None, 'step': b_step, 'round': 5, 'log': True}

    grid_param = {'a': a_param, 'b': b_param}

    parser = UndulationExperiment.parameter_parser()
    model_param = parser.parse_args([])
    
    model_param.dt = 0.001
    model_param.N = 250
    model_param.dt_report = 0.01
    model_param.N_report = 125

    PG = ParameterGrid(vars(model_param), grid_param)

    CS = UndulationExperiment.stw_control_sequence(model_param)

    r_list = []

    for param in PG.param_arr:
        r, t = solve_weak_form(param, CS)
        r_list.append(r)
    
        
    return

def test_worm_vs_weak_form():

    parser = UndulationExperiment.parameter_parser()
    param = parser.parse_args()            
    param.N = 250
    param.dt = 1e-2
    param.T = 2.0
    
    MP = ModelParameter(param)
                        
    CS = UndulationExperiment.stw_control_sequence(param)

    r_mat, theta_mat, _, _ = solve_weak_form(param, CS)
    
    worm = Worm(param.N, param.dt)    
    
    FS = worm.solve(param.T, MP, CS, 
        FK = ['t', 'r', 'theta', 'V_dot', 'D_I_dot', 'D_F_dot', 'W_dot'])[0]

    assert np.allclose(r_mat, FS.r, atol = 1e-3)
    assert np.allclose(theta_mat, FS.theta, atol = 1e-3)

    print('Passed test. MWE of the weak form and Worm class yields same centreline'
          'coordinates and Euler angles')

    return

if __name__ == '__main__':
    
    #test_zero_control()
    #test_undulation()
    # test_power_balance()
    test_worm_vs_weak_form()
    