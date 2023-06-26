'''
Created on 9 Jun 2023

@author: amoghasiddhi


The rotation matrix _Q operates on a vector v in the original coordinate system with basis vector 
e1, e2, e3 and transforms it into a vector v_bar in the new coordinate system with basis
vectors d1, d2, d3. The transformation is defined as

v_bar = _Q*v 

The components of v_bar can be used to express v in terms of the new basis vector d1, d2, d3

v = v_bar_1*d1 + v_bar_2*d2 + v_bar_3*d3

Multiplying with _Q from the left yields 

v =  v_bar_1*_Q*d1 + v_bar_2*_Q*d1 + v_bar_3*_Q*d1 

from which follows that ei=_Q*di and _Q^T*ei=di where we used that _Q^T*_Q=I. From
_Q^T*ei=di follows that _Q must have the new basis vectors di as rows. 
'''

import numpy as np
from fenics import *
from ufl import atan_2

from minimal_worm.util import f2n

def R1_numpy(g):
    
    cos_g = np.cos(g)
    sin_g = np.sin(g)
    return np.array([
        [1, 0, 0],
        [0, cos_g, -sin_g],
        [0, sin_g, cos_g]]
    )

def R2_numpy(b):
    cos_b = np.cos(b)
    sin_b = np.sin(b)
    
    return np.array([
        [cos_b, 0, sin_b],
        [0, 1, 0],
        [-sin_b, 0, cos_b]]
    )
    
def R3_numpy(a):

    cos_a = np.cos(a)
    sin_a = np.sin(a)
    return np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]]
    )

def R1_fenics(g):

    return as_matrix(
        [[1, 0, 0], 
         [0, cos(g), -sin(g)], 
         [0, sin(g), cos(g)]]
    )
    

def R2_fenics(b):

    return as_matrix(
        [[cos(b), 0, sin(b)], 
         [0, 1, 0], 
         [-sin(b), 0, cos(b)]]
    )
    
def R3_fenics(a):

    return as_matrix(
        [[cos(a), -sin(a), 0], 
         [sin(a), cos(a), 0], 
         [0, 0, 1]]
    )        
    
def Q_from_abg_numpy(a, b, g):
    
    R1 = R1_numpy(g)
    R2 = R2_numpy(b)
    R3 = R3_numpy(a)
    
    return np.matmul(R3, np.matmul(R2, R1))
            
def Q_from_abg_fenics(a, b, g):
    
    R1 = R1_fenics(g)
    R2 = R2_fenics(b) 
    R3 = R3_fenics(a)
    
    return R3*R2*R1


        
def abg_from_Q_numpy(_Q):
    
    a = np.arctan2(_Q[1,0],_Q[0,0])
    b = np.arcsin(-_Q[2,0])
    g = np.arctan2(_Q[2,1], _Q[2,2])
    
    return a,b,g

def abg_from_Q_fenis(_Q):
    
    a = atan_2(_Q[1,0], _Q[0,0])
    b = asin(-_Q[2,0])
    g = atan_2(_Q[2,1], _Q[2,2])
    
    return a,b,g

def test_Euler_angle_round_trip_numpy():    
    
    N = 100
    s_arr = np.linspace(0, 2*np.pi, N)
    
    # Lab frame
    e1 = np.array([1, 0, 0])
    e2 = np.array([0, 1, 0])
    e3 = np.array([0, 0, 1])
    
    # Body frame vectors
    d3_arr = np.zeros((N, 3)) 
    d3_arr[:, 1] = np.sin(s_arr)
    d3_arr[:, 2] = np.cos(s_arr)
    
    d1_arr = np.zeros((N, 3))
    d1_arr[:, 0] = 1    
    d2_arr = np.cross(d3_arr, d1_arr)
    
    a_arr, b_arr, g_arr = np.zeros(N), np.zeros(N), np.zeros(N) 
        
    atol = 1e-3
        
    for i, (d1, d2, d3) in enumerate(zip(d1_arr, d2_arr, d3_arr)):
                        
        # Rotation matrix has body frame vectors as rows
        _Q = np.vstack((d1, d2, d3))
        
        # Check that di = _Q^T*ei
        assert np.allclose(d1, np.matmul(_Q.T, e1), atol = atol)
        assert np.allclose(d2, np.matmul(_Q.T, e2), atol = atol)
        assert np.allclose(d3, np.matmul(_Q.T, e3), atol = atol)

        # Euler angles from _Q
        a, b, g = abg_from_Q_numpy(_Q)                        
        
        # Check that _Q from body frame vectors and Euler angles
        # are identical
        assert np.allclose(_Q, Q_from_abg_numpy(a,b,g), atol = atol)
        
        a_arr[i], b_arr[i], g_arr[i] = a,b,g

    
    assert np.allclose(a_arr, 0, atol=atol)
    assert np.allclose(b_arr, 0, atol=atol)
    assert np.allclose(g_arr, np.arctan2(np.sin(s_arr), np.cos(s_arr)), atol = atol)
    
        
    print('Body frame vectors to rotation matrix to Euler angles' 
          'to rotation matrix passed round trip test')

def test_Euler_angle_round_trip_fenics():    

    N = 500

    K = 1
    # mesh = UnitIntervalMesh(N - 1)
    mesh = UnitIntervalMesh(N - 1)
    # Finite elements for 1 dimensional spatial coordinate s        
    P1 = FiniteElement('Lagrange', mesh.ufl_cell(), degree = K)    
    # State variables r and theta are 3 dimensional vector-valued functions of s                        
    P1_3 = MixedElement([P1] * 3)
    # Function space for scalar functions of s
    V = FunctionSpace(mesh, P1)
    # Function space for 3 component vector-valued functions of s        
    V3 = FunctionSpace(mesh, P1_3)
    # Function space for 3x3 tensor-valued functions of s        
    V33 = TensorFunctionSpace(mesh, P1, shape=(3,3), degree = K)

    e1 = Constant((1, 0, 0))
    e2 = Constant((0, 1, 0))
    e3 = Constant((0, 0, 1))
            
    # Body frame vectors
    d3 = Expression(('0', 'sin(2*pi*x[0])', 'cos(2*pi*x[0])'), degree = K)
    d1 = Expression(('1', '0', '0'), degree = K)
    d2 = cross(d3, d1)
            
    atol = 1e-4
    
    assert np.allclose(f2n(project(dot(d1, d2), V)), 0, atol)
    assert np.allclose(f2n(project(dot(d2, d3), V)), 0, atol)
    assert np.allclose(f2n(project(dot(d1, d3), V)), 0, atol)
    
    _Q = outer(e1, d1) + outer(e2, d2) + outer(e3, d3)
                                                                     
    assert np.allclose(f2n(project(d1 - _Q.T*e1, V3)), 0, atol)   
    assert np.allclose(f2n(project(d2 - _Q.T*e2, V3)), 0, atol)   
    assert np.allclose(f2n(project(d3 - _Q.T*e3, V3)), 0, atol)   
               
    a, b, g = abg_from_Q_fenis(_Q)    

    assert np.allclose(project(a, V).vector().get_local(), 0)
    assert np.allclose(project(b, V).vector().get_local(), 0)
    # assert np.allclose(project(g, V).vector().get_local(), 0)
                           
    assert np.allclose(project(_Q - Q_from_abg_fenics(a, b, g), V33).vector().get_local(), 0, atol=atol)                                                     
    
    print('Passed Euler angles round trip test in Fenics')

if __name__ == '__main__':
    
    Q_sym()
    test_Euler_angle_round_trip_numpy() 
    test_Euler_angle_round_trip_fenics() 
    
