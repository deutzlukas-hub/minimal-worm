"""
Created on 12 May 2022

@author: lukas
"""


#Built-in imports
from typing import Dict, Optional, Tuple, List
from types import SimpleNamespace
import logging
import ffc

# Third-part imports
import numpy as np
from fenics import *

# Local imports
from minimal_worm.util import v2f, f2n
from minimal_worm.frame import FRAME_KEYS, Frame, FrameSequence

from minimal_worm.model_parameters import ModelParameter

try:
    from fenics_adjoint import *
except ModuleNotFoundError:
    # This optional import is only needed if derivatives are being taken.
    pass

# Set Fenics LogLevel to Error to
# avoid logging to mess with progressbar
from dolfin import set_log_level, LogLevel
set_log_level(LogLevel.ERROR)

CONTROL_KEYS = ['k0', 'sig0', 't']

def grad(f):
    return Dx(f, 0)

# global geometry helpers
dxL = dx(
    scheme="vertex", degree=1, metadata={"representation": "quadrature", "degree": 1}
)


# Lab frame
e1 = Constant((1, 0, 0))
e2 = Constant((0, 1, 0))
e3 = Constant((0, 0, 1))

class Worm:
    
    # Default input parameter
    solver = {}
    
#------------------------------------------------------------------------------ 
#

    def __init__(self, 
            N: int, 
            dt: float,             
            fe = {'type': 'Lagrange', 'degree': 1},            
            fdo = 2,
            quiet= False):
        '''
        
        :param N *():
        :param dt:
        :param fe:
        :param fdo:
        :param quiet:
        '''
        
        self.N = N
        self.dt = dt
        self.fdo = fdo

        self.fe = fe
        
        self.quiet = quiet

        self._init_function_space()
        
    def _init_function_space(self):
        '''
        Initialise finite element function spaces
        '''
        # mesh
        self.mesh = UnitIntervalMesh(self.N - 1)

        # Finite elements for 1 dimensional spatial coordinate s        
        P1 = FiniteElement(self.fe['type'], self.mesh.ufl_cell(), self.fe['degree'])
                
        # State variables r and theta are 3 dimensional vector-valued functions of s                        
        P1_3 = MixedElement([P1] * 3)

        # Function space for scalar functions of s
        self.V = FunctionSpace(self.mesh, P1)
        # Function space for 3 component vector-valued functions of s        
        self.V3 = FunctionSpace(self.mesh, P1_3)

        # Trial function space for 6 component vector-valued function composed of r and theta
        self.W = FunctionSpace(self.mesh, MixedElement(P1_3, P1_3))
        
        
        # Define function space for outputs
        self.output_func_spaces = {            
            'd1': self.V3,
            'd2': self.V3,
            'd3': self.V3,
            'k': self.V3,
            'sig': self.V3,
            'r_t': self.V3,
            'w': self.V3,
            'k_t': self.V3,
            'sig_t': self.V3,
            'f_F': self.V3,
            'l_F': self.V3,
            'f_F': self.V3,
            'f_M': self.V3,
            'l_M': self.V3,
            'N': self.V3,
            'M': self.V3
        }
            
                                             
        return

    def _assign_initial_values(self, F0: Optional[Frame] = None):
        '''
        Initialise initial state and state history 
        '''
                
        # Trial functions
        r0 = Function(self.V3)                
        theta0 = Function(self.V3)

        # If no frame is given, use default
        if F0 is None:
            # Set initial configuration
            r0.assign(Expression(("0", "0", "x[0]"), degree=self.fe['degree']))
            theta0.assign(Expression(("0", "0", "0"), degree=self.fe['degree']))

        # If Numpy frame is given, assign array values to fenics functions
        else:
            v2f(F0.r, r0, self.V3)
            v2f(F0.theta, theta0, self.V3)

        self._init_state_history(r0, theta0)

        return
  
    def _init_state_history(self, r0, theta0):
        '''
        Creates array which stores the past k states of the system
        to approximate nth derivatives of order k.
        '''
        
        # For nth derivative of order k, we need n+k-1 past time points
        N = self.fdo
                
        # Array for past states
        self.u_old_arr = [Function(self.W) for _ in np.arange(N)]
        
        # Assign (r, theta) tuple in [V3, V3] to u in W
        fa = FunctionAssigner(self.W, [self.V3, self.V3])

        for u_old_n in self.u_old_arr:
            fa.assign(u_old_n, [r0, theta0])
        
        return

#------------------------------------------------------------------------------ 
# Finite difference approximation used in weak form
    
    def _finite_difference_coefficients(self, n, k):
        '''
        Calculates weighting coefficients for finite backwards 
        difference of order k for nth derivative
        '''
        if not hasattr(self, 'c_arr_cache'):
            self.c_arr_cache = {}

        # Return coefficients if cached            
        if (n, k) in self.c_arr_cache:
            return self.c_arr_cache[(n, k)]
        
        # Number points for required for nth derivative of kth order 
        N = n + k

        # point indexes [-k, ..., -1, 0]
        # 0 = current, -1 = previous time point, ...        
        s_arr = np.arange(-N+1, 1)
        A = np.vander(s_arr, increasing=True).T
        b = np.zeros(N)
        b[n] = np.math.factorial(n)

        # Weighting coefficients correspond to 
        # points as indexed by s_arr 
        c_arr = np.linalg.solve(A, b)
        # Fenics can't handle numpy floats
        c_arr = c_arr.tolist()
        
        self.c_arr_cache[(n, k)] = (c_arr, s_arr)
    
        return c_arr, s_arr

    def _finite_backwards_difference(self, 
            n,
            k,
            z, 
            z_old_arr):
        """Approximate nth derivative by finite backwards difference
        of order k of given fenics function z"""

        c_arr, s_arr = self._finite_difference_coefficients(n, k)

        z_t = 0        
        # Add terms to finite backwards difference 
        # from most past to most recent time point
        for s, c in zip(s_arr, c_arr):
            if s == 0:  
                z_t += c * z            
            else:
                z_t += c * z_old_arr[s]

        z_t = z_t / self.dt**n

        return z_t

    def _init_first_time_derivatives(self, r, theta):
        '''
        Initialises finite backwards difference for first time 
        derivatives of the centreline and Euler angles in Fenics        
        '''         
        # Old centreline coordinates
        r_old_arr = [split(u)[0] for u in self.u_old_arr]
                        
        # Old Euler angles
        theta_old_arr = [split(u)[1] for u in self.u_old_arr]

        r_t = self._finite_backwards_difference(
            1, self.fdo, r, r_old_arr)
        theta_t = self._finite_backwards_difference(
            1, self.fdo, theta, theta_old_arr)        
        
        return r_t, theta_t

    def _init_form(self):
                    
        u = TrialFunction(self.W)
        phi = TestFunction(self.W)
        
        r, theta = split(u)
        phi_r, phi_theta = split(phi)
        
        self.u_h = Function(self.W)
        
        r_h, theta_h = split(self.u_h)
        
        r_t, theta_t = self._init_first_time_derivatives(r, theta)
        
        A_h = Worm.A(theta_h)
        A_h_t = Worm.A_t(theta_h, theta_t) #alpha_t, beta_t)
        Q_h = Worm.Q(theta_h)
        T_h = Worm.T(r_h)
            
        # length-element
        eps_h = self.eps(r_h)
        
        # angular velocity vector
        w = Worm.w(A_h, theta_t)
        # shear/stretch vector
        sig = Worm.sig(Q_h, r)
        # time derivative shear stretch vector
        sig_t = Worm.sig_t(Q_h, r_h, r_t, w)
            
        # generalized curvature
        k = Worm.k(A_h, theta, eps_h)
        
        # time derivative generalized curvature
        k_t = Worm.k_t(A_h, A_h_t, theta_h, theta_t, eps_h)
                                        
        # internal force
        N = self.N_(Q_h, sig, sig_t)
        # internal torque
        M = self.M(Q_h, k, k_t)
                
        # external fluid drag torque
        l_F = self.l_F(Q_h, w)
        # external fluid drag force
        f_F = self.f_F(Q_h, r_t)
        # linear balance
        eq1 = dot(f_F, phi_r) * dx - dot(N, grad(phi_r)) * dx
        # angular balance
        eq2 = (
            dot(l_F, phi_theta) * dx
            + dot(T_h * N, phi_theta) * dx
            - dot(M, grad(phi_theta)) * dx
        )
                
        equation = eq1 + eq2
                
        self.F_op, self.L = lhs(equation), rhs(equation)
                                
        return

    def include_boundary(self):
        
        # Include boundaries        
        # boundary_term_left = u * v * ds_left(1)  # Subdomain ID 1 corresponds to the left boundary
        # boundary_term_right = u * v * ds_right(2)  # Subdomain ID 2 corresponds to the right boundary
                
        # Define the left boundary condition
        def left_boundary(x, on_boundary):
            return on_boundary and near(x[0], 0.0)
        
        # Define the right boundary condition
        def right_boundary(x, on_boundary):
            return on_boundary and near(x[0], 1.0)
        
        # Create boundary measures for the left and right boundaries
        ds_left = Measure('ds', domain=mesh, subdomain_data=mesh.domains(), subdomain_id=1)
        ds_right = Measure('ds', domain=mesh, subdomain_data=mesh.domains(), subdomain_id=2)

        return ds_left, ds_right

    def _print(self, s):
        # todo: proper logging!
        if not self.quiet:
            print(s)

    def initialise(
        self,
        MP: ModelParameter,
        CS: Dict,
        FK: List,        
        F0: Optional[Frame] = None,
        solver: Dict = None,
        picard: Dict = None,
        pbar: bool = None,
        logger = None, 
        dt_report: Optional[float] = None,
        N_report: Optional[int] = None,
    ):
        """
        Initialise worm object for given model parameters, control
        sequence (optional) and initial frame (optional).
        """
        
        self.cache = {}
        
        if solver is not None:
            Worm.solver.update(solver)        
        
        self.picard = picard
        
        if pbar is not None:
            pbar.total = self.n
        self.logger = logger

        assert all(k in FRAME_KEYS for k in FK), 'output keys must be in FRAME_KEYS' 
        self.FK = FK

        # Get time steps number of significant digits after decimal point
        self.sd = len(str(self.dt)) - str(self.dt).find('.') - 1  

        if dt_report is not None:            
            if dt_report == self.dt:
                self.dt = None
            else:
                assert dt_report > self.dt, (f'Reported time step dt_report={dt_report} ' 
                f'must be larger than simulation time step dt={self.dt}')                                            
                self.t_step = round(dt_report/self.dt)
        else:
            self.t_step = None
            
        if N_report is not None:
            if N_report == self.N:
                self.s_step = None
            else:            
                assert N_report < self.N, (f'The reported numbers of mesh points along ' 
                    f'the centreline N_report={N_report}, must be smaller than ' 
                    f'the total number of mesh points N={self.N}')
                self.s_step = round(self.N/N_report) 
        else: 
            self.s_step = None
        
        if F0 is not None:
            self._t = F0.t
        else:
            self._t = 0.0

        self.MP = MP
        self.C, self.D, self.Y, self.S, self.S_tilde, self.B, self.B_tilde = MP.to_fenics()

        # If the preferred curvature is specified as a Fenics.Expressions 
        # or Constant then assign epxression to self.k0   
        if isinstance(CS['k0'], (Expression, Constant)):
            self.k0 = CS['k0']
        # If the preffered curvature is specified in terms of a numpy.ndarray 
        # then assign fenics.Function to self.k0                 
        elif isinstance(CS['k0'], np.ndarray):        
            assert CS['k0'].shape[0] == self.n, ("Preferred curvature vector" 
                "not available for every simulation step.")            
            self.k0 = Function(self.V3)
        else:
            assert False, ("Preferred curvature CS['k0'] must be one of" 
                "[Fenics.Expression, Fenics.Constant, np.ndarray]")

        # If the preferred shear/stretch is specified as a Fenics.Expressions 
        # or Constant then assign epxression to self.sig0   
        if isinstance(CS['sig0'], (Expression, Constant)):
            self.sig0 = CS['sig0']
        # If the preffered shear/strech is specified in terms of a numpy.ndarray 
        # then assign fenics.Function to self.sig0                 
        elif isinstance(CS['sig0'], np.ndarray):        
            assert CS['sig0'].shape[0] == self.n, ("Preferred shear/stretch vector" 
                "not available for every simulation step.")            
            self.sig0 = Function(self.V3)
        else:
            assert False, ("Preferred shear/stretch CS['sig0'] must be one of" 
                "[Fenics.Expression, Fenics.Constant, np.ndarray]")
                    
        self._assign_initial_values(F0)
        self._init_form()

        return

    def solve(self, 
        T: float, 
        MP: ModelParameter, 
        CS: Dict, 
        F0=None, 
        solver = None,
        picard = None,
        FK = None,
        pbar=None, 
        logger=None, 
        dt_report=None, 
        N_report=None
    ) -> Tuple[FrameSequence, Optional[Exception]]:
        
        """
        Run the forward model for T seconds.
        """

        self.n = int(T / self.dt) # number of timesteps
        
        if FK is None:
            FK = FRAME_KEYS
        
        self.initialise(
            MP, CS, FK, F0, solver, picard, pbar, logger, dt_report, N_report
        )

        self._print(f'Solve forward' 
            f'(t={self._t:.{self.sd}f}..{self._t + T:.{self.sd}f}) / n_steps={self.n}')
                        
        # Frame Sequence
        FS = []        
        # Controls
        Cs = []
                
        # Try block allows for exception handling. If we run simulations 
        # in parallel, we don't want the whole queue to crash if individual 
        # simulations fail
        try:
            for self.i in range(self.n):
                
                self._print(f"t={self._t:.{self.sd}f}")
                               
                F, C = self.update_solution(CS)

                if F is not None:
                    FS.append(F)
                if C is not None:
                    Cs.append(C)
                    
                if pbar is not None:
                    pbar.update(1)

        except Exception as e:
            CS = {k: np.array([C[k] for C in Cs]) for k in CONTROL_KEYS}
            return FrameSequence(FS), CS, e

        CS = {k: np.array([C[k] for C in Cs]) for k in CONTROL_KEYS}
        
        return FrameSequence(FS), SimpleNamespace(**CS), None 
        
    def _update_control(self, CS): 
        '''
        Update preferred curvature and shear/stretch in weak form
        '''

        # If CS['k_pref'] is a Fenics.Expression then
        # then we update the Fenics.Constant
        # which is used as the time variable in the expression
        if isinstance(CS['k0'], Expression):                                        
            CS['t'].assign(self._t)        
        # If CS['k_pref'] is a numpy.ndarray then 
        # self.k0 is a Fenics.Function and we  
        # assign row i to self.k0        
        elif isinstance(CS['k0'], np.ndarray):
            v2f(self.k0, CS['k0'][self.i, :])                        
        
        if isinstance(CS['sig0'], np.ndarray):                                        
            v2f(self.sig0, CS['sig0'][self.i, :])                
            
        return

    def update_solution(self, CS) -> Optional[Frame]:
        '''        
        Solve time step and save solution to Frame
        '''
        
        self._t += self.dt

        self._update_control(CS)
        self.u_h.assign(self.u_old_arr[-1])

        
        if self.picard['on']:
            u = self.picard_iteration()
        else:        
            u = Function(self.W)            
            solve(self.F_op == self.L, u, solver_parameters=Worm.solver)
        
        assert not np.isnan(u.vector().get_local()).any(), (
            f'Solution at t={self._t:.{self.D}f} contains nans!')
        
        self._r, self._theta = u.split(deepcopy=True)

        # Frame and outputs need to be assembled before u_old_arr
        # is updated for derivatives to use correct data points        
        if self.t_step is not None and not (self.i + 1) % self.t_step == 0:
            F = None            
            C = None
        else:
            F = self._assemble_frame()
            C = self._assemble_controls()
                                                                    
        # update past solution cache                
        for n, u_n in enumerate(self.u_old_arr[:-1]):
            u_n.assign(self.u_old_arr[n + 1])
        
        self.u_old_arr[-1].assign(u)

        return F, C

    def picard_iteration(self):

        """Solve nonlinear system of equations using picard iteration"""

        # Trial function
        u = Function(self.W)
        
        # Solution from previous time step
        u_old = self.u_old_arr[-1]
        r_old, theta_old = u_old.split()        
        r_old_arr = r_old.compute_vertex_values(self.mesh).reshape(3, self.N)
        theta_old_arr = theta_old.compute_vertex_values(self.mesh).reshape(3, self.N) 
                
        # Initial guess        
        self.u_h.assign(u_old)

        tol = self.picard['tol']
        lr = self.picard['lr'] 
        maxiter = self.picard['max_iter'] 
        
        i = 0
        converged = False
                
        while i < maxiter:            
            solve(self.F_op == self.L, u, solver_parameters=Worm.solver)
            r, theta = u.split()
            r_h, theta_h = self.u_h.split()
            
            # Error   
            err_r = assemble(sqrt((r-r_h)**2)*dx)
            err_theta = assemble(sqrt((theta-theta_h)**2)*dx)
                        
            # Normalize by average change per time step            
            norm_r = assemble(sqrt((r - r_old)**2)*dx)
            norm_theta = assemble(sqrt((theta - theta_old)**2)*dx)
            
            rel_err_r  = err_r / max(norm_r, 1.0e-12)
            rel_err_theta  = err_theta / max(norm_theta, 1.0e-12)
                                                            
            if rel_err_r < tol and rel_err_theta < tol:                
                if not self.quiet:
                    print(
                        f"Picard iteration converged after {i} iterations: err_r={err_r}, err_theta={err_theta}"
                    )
                converged = True
                break

            self.u_h.assign(lr * u + (1.0 - lr) * self.u_h)
            i += 1
            
        assert converged, 'Picard iteration did not converge'

        return u
                                        
    def _assemble_frame(self):
        '''
        Assemble frames
        '''
                
        self.cache.clear()
        
        kwargs = {}
    
        # Check if float, Expression or Function        
        for k in self.FK:
        
            v = getattr(self, f'_{k}')
            
            if isinstance(v, float):
                kwargs[k] = v
                continue
            
            if isinstance(v, Function):
                v_arr = f2n(v)
            else:
                v_arr = f2n(project(v, self.output_func_spaces[k]))
                                
            if self.s_step is not None:
                v_arr = v_arr[..., ::self.s_step]
             
            kwargs[k] = v_arr
                                
        return Frame(**kwargs)

    def _assemble_controls(self):
        '''
        Assemble control
        '''  

        C = {}

        for k in ['sig0', 'k0']:
            v_pref = getattr(self, k)                          
            if isinstance(v_pref, Expression):
                v_arr = f2n(project(v_pref, self.V3))
            elif isinstance(v_pref, Constant):
                v_arr = np.tile(v_pref.values()[:, None], (1, self.N))
            elif isinstance(v_pref, np.ndarray):
                v_arr = v_pref[self.i, :]                                    
        
            if self.s_step is not None:
                v_arr = v_arr[..., ::self.s_step]
            
            C[k] = v_arr
                                    
        C['t'] = self._t
                                                 
        return C

#------------------------------------------------------------------------------ 
# Define all relevant variables and terms and in the equations of motion 
    
    @staticmethod
    def Q(theta):
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
    @staticmethod
    def A(theta):
        """The matrix A is used to calculate the curvature k and
        angular velocity w from first Euler angle derivatives
        with respect to s and t"""

        alpha, beta = theta[0], theta[1] 

        A = as_matrix(
            [
                [0, sin(alpha), -cos(alpha) * cos(beta)],
                [0, -cos(alpha), -sin(alpha) * cos(beta)],
                [-1, 0, sin(beta)],
            ]
        )
        return A

    @staticmethod
    def A_t(theta, theta_t):
        """Time derivative of matrix A is used to calculate the 
        time derivative of the curvature vector k"""

        alpha, beta, _ = split(theta)
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

    @staticmethod
    def T(r):
        '''
        Matrix representation of centreline tangent cross product
        '''
        
        x, y, z = r[0], r[1], r[2] 
                    
        # Cross product matrix
    
        return as_matrix(
            [[0, -grad(z), grad(y)], 
             [grad(z), 0, -grad(x)], 
             [-grad(y), grad(x), 0]]
        )

    @staticmethod
    def eps(r):
        '''
        Local compression/stretch ratio
        '''
        
        return sqrt(dot(grad(r), grad(r)))
    
    @staticmethod
    def w(A, theta_t):
        '''
        Angular velocity
        '''
        
        return A * theta_t
    
    @staticmethod
    def sig(Q, r):
        '''
        Shear/stretch vector
        '''
                
        return Q * grad(r) - e3        
    
    @staticmethod
    def k(A, theta, eps):
        '''
        Generalized curvature vector
        '''
        
        return A * grad(theta) # / eps        
    
    @staticmethod
    def sig_t(Q, r, r_t, w):
        '''
        Time derivative of shear/stretch vector
        '''        
        
        return Q * grad(r_t) - cross(w, Q * grad(r))        

    @staticmethod    
    def k_t(A, A_t, theta, theta_t, eps):
        '''
        Time derivative of curvature vector
        '''        
        # return A * grad(theta_t) / eps + A_t * grad(theta) / eps
    
        return A * grad(theta_t) + A_t * grad(theta) 
    
    
    def f_F(self, Q, r_t):
        '''
        Fluid drag force line density
        '''                
        d3 = Q.T * e3
        d3d3 = outer(d3, d3)
        f_F = -(d3d3 + self.C * (Identity(3) - d3d3)) * r_t
        
        return f_F

    def l_F(self, Q, w):
        '''
        Fluid drag force line density
        '''                
        e3e3 = outer(e3, e3)
        
        l_F = -Q.T*self.D*(e3e3 + self.Y * (Identity(3) - e3e3)) * w
        
        return l_F

    def N_(self, Q, sig, sig_t):
        '''
        Internal force resultant
        '''

        return Q.T * (self.S * (sig - self.sig0) + self.S_tilde * sig_t)
            
    
    def M(self, Q, k, k_t):
        '''
        Internal torque resultant
        '''
        
        return Q.T * (self.B * (k - self.k0) + self.B_tilde * k_t)


#------------------------------------------------------------------------------ 
# Wrapper functions which cache and return ouput variables of interest 
    
    @property
    def _Q(self):
        """Rotation matrix from global to local frames"""

        if 'Q' in self.cache:
            return self.cache['Q']

        self.cache['Q'] = Worm.Q(self._theta)

        return self.cache['Q']

    @property
    def _d1(self):
        '''
        Calculate body frame vector 1
        '''
        
        if 'd1' in self.cache:
            return self.cache['d1']
        
        self.cache['d1'] = self._Q.T * e1
        
        return self.cache['d1']
        
    @property
    def _d2(self):
        '''
        Calculate body frame vector 2
        '''
        
        if 'd2' in self.cache:
            return self.cache['d2']
        
        self.cache['d2'] = self._Q.T * e2
        
        return self.cache['d2']

    @property
    def _d3(self):
        '''
        Calculate body frame vector 3
        '''
        
        if 'd3' in self.cache:
            return self.cache['d3']
        
        self.cache['d3'] = self._Q.T * e3
        
        return self.cache['d3']

    @property
    def _A(self):
        """A matrix is used to calculate the curvature and
        angular velocity from first Euler angle derivatives
        with respect to s and t"""

        if 'A' in self.cache:
            return self.cache['A']

        self.cache['A'] = Worm.A(self._theta)

        return self.cache['A']

    @property
    def _eps(self):
        '''
        Local stretch/compression ratio
        '''
        if 'eps' in self.cache:
            return self.cache['eps']
        
        self.cache['eps'] = Worm.eps(self._r)
        
        return self.cache['eps']
            
    @property
    def _A_t(self):
        """Time derivative of A is needed for curvature rate"""

        if 'A_t' in self.cache:
            return self.cache['A_t']

        self.cache['A_t'] = Worm.A_t(self._theta, self._theta_t)
        
        return self.cache['A_t']
    
    @property
    def _sig(self):
        '''
        Shear/stretch vector
        '''
        if 'sig' in self.cache:
            return self.cache['sig']
                        
        self.cache['sig'] =  Worm.sig(self._Q, self._r)
                
        return self.cache['sig']
    
    @property                    
    def _k(self):
        '''
        Generalized curvature vector
        '''
        
        if 'k' in self.cache:
            return self.cache['k']
        
        self.cache['k'] = Worm.k(self._A, self._theta, self._eps)
        
        return self.cache['k']
    @property   
    def _sig_norm(self):
        '''
        L1 norm real minus preferred shear/stretch vector 
        '''
        sig_err = self._sig - self.sig0
        
        return assemble(sqrt(dot(sig_err, sig_err))*dx)        

    @property
    def _k_norm(self):
        '''
        L1 norm real curvature minus preferred curvature norm
        '''        
        k_err = self._k - self.k0
        
        return assemble(sqrt(dot(k_err, k_err))* dx)        

    @property        
    def _r_t(self):
        '''
        Centreline velocity
        '''
        if 'r_t' in self.cache:
            return self.cache['r_t']
        
        r_old_arr = [split(u)[0] for u in self.u_old_arr]
        r_t = self._finite_backwards_difference(
            1, self.fdo, self._r, r_old_arr)
        
        self.cache['r_t'] = r_t 
        
        return r_t
    
    @property
    def _theta_t(self):
        '''
        Euler angle time derivative
        '''
        
        if 'theta_t' in self.cache:
            return self.cache['theta_t']
        
        theta_old_arr = [split(u)[1] for u in self.u_old_arr]
        theta_t = self._finite_backwards_difference(
            1, self.fdo, self._theta, theta_old_arr)
        
        self.cache['theta_t'] = theta_t
        
        return self.cache['theta_t']

    
    @property
    def _w(self):
        '''
        Angular velocity
        '''
        if 'w' in self.cache:
            return self.cache['w']
        
        w = self.w(self._A, self._theta_t)
        
        self.cache['w'] = w
        
        return self.cache['w']
    
    @property
    def _sig_t(self):
        '''
        Shear/stretch rate
        '''

        if 'sig_t' in self.cache:
            return self.cache['sig_t']

        self.cache['sig_t'] = self.sig_t(
            self._Q, self._r, self._r_t, self._w)
         
        return self.cache['sig_t']
    
    @property
    def _k_t(self):
        '''
        Curvature rate
        '''
        
        if 'k_t' in self.cache:
            return self.cache['k_t']

        self.cache['k_t'] = self.k_t(
            self._A, self._A_t, self._theta, self._theta_t, self._eps)
        
        return self.cache['k_t']

    
    @property        
    def _f_F(self):
        '''
        External fluid force line density per unit reference arc-length s in
        the lab frame
        '''                
        
        if 'f_F' in self.cache:
            return self.cache['f_F']
        
        self.cache['f_F'] = self.f_F(self._Q, self._r_t)

        return self.cache['f_F']
    
    @property
    def _l_F(self):
        '''
        External fluid torque line density per unit reference arc-length s in the lab frame
        '''        

        if 'l_F' in self.cache:
            return self.cache['l_F']
        
        self.cache['l_F'] = self.l_F(self._Q, self._w)
        
        return self.cache['l_F']
    
    @property
    def _f_M(self):
        
        if 'f_M' in self.cache:
            return self.cache['f_M']
        
        self.cache['f_M'] = -grad(project(self.S*self.sig0, self.V3))
        
        return self.cache['f_M']
    
    @property    
    def _l_M(self):
        
        if 'l_M' in self.cache:
            return self.cache['l_M']
        
        self.cache['l_M'] = -grad(project(self.B*self.k0, self.V3))
        
        return self.cache['l_M']
    
    @property    
    def _M(self):
        '''
        Internal torque resultant M for passive viscoelastic material
        '''
        if 'M' in self.cache:
            return self.cache['M']

        self.cache['M'] = self.M(self._Q, self._k, self._k_t)
        
        return self.cache['M']

    @property
    def _N(self):
        '''
        Internal force resultant N for passive viscoelastic material 
        '''
        if 'N' in self.cache:
            return self.cache['N']

        self.cache['N'] = self.N_(self._Q, self._sig, self._sig_t)

        return self.cache['N']
    
    
#------------------------------------------------------------------------------ 
# Energies
    
    @property
    def _V(self):
        '''
        Calculate elastic energy
        '''
                
        V_k = 0.5 * assemble(dot(self._k, self.B * self._k) * dx)
        V_sig = 0.5 * assemble(dot(self._sig, self.S * self._sig) * dx)

        return V_k + V_sig

    @property
    def _D_F_dot(self):
        '''
        Calculate fluid dissipation rate
        '''        
        
        D_F_dot_f = assemble(dot(self._f_F, self._r_t) * dx)
        D_F_dot_l = assemble(dot(self._l_F, self._w) * dx)

        return D_F_dot_f + D_F_dot_l 

    @property
    def _D_I_dot(self):
        '''
        Calculate internal dissipation rate
        '''
        D_I_dot_sig = -assemble(dot(self._sig_t, self.S_tilde * self._sig_t) * dx)
        D_I_dot_k = -assemble(dot(self._k_t, self.B_tilde * self._k_t) * dx)

        return D_I_dot_sig + D_I_dot_k

    @property
    def _V_dot(self):
        '''
        Calculate rate of change in potential energy
        '''                
                
        V_dot_k = assemble(dot(self._k, self.B * self._k_t) * dx)                        
        V_dot_sig = assemble(dot(self._sig, self.S * self._sig_t) * dx)
        
        return V_dot_sig + V_dot_k
    
    @property    
    def _W_dot(self):
        '''
        Calculate mechanical muscle power
        '''
        W_dot_f = assemble(dot(self._f_M, self._Q * self._r_t) * dx)
        W_dot_l = assemble(dot(self._l_M, self._w) * dx)
        
        return W_dot_f + W_dot_l 
