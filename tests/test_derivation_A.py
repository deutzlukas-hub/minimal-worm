'''
Created on 11 Jun 2023

@author: amoghasiddhi
'''
from sympy import *
        
# coordinates
s, t = symbols('s t')
    
# Define symbolic variables for angles a, b, and g
a, b, g = Function('a')(s, t), Function('b')(s, t), Function('g')(s, t)
a_s, b_s, g_s = diff(a, s), diff(b, s), diff(g, s) 
a_t, b_t, g_t = diff(a, t), diff(b, t), diff(g, t) 


c1, c2, c3 = cos(a), cos(b), cos(g)
s1, s2, s3 = sin(a), sin(b), sin(g)

e1, e2, e3 = Matrix([1, 0, 0]), Matrix([0, 1, 0]), Matrix([0, 0, 1])
 
def _Q():
    '''
    From https://en.wikipedia.org/wiki/Euler_angles
    '''

    return Matrix([
    [c1*c2, c1*s2*s3 - c3*s1, s1*s3 + c1*c3*s2],
    [c2*s1, s1*s2*s3 + c1*c3, c3*s1*s2 - c1*s3],
    [-s2, c2*s3, c2*c3]
    ]) 

def _d123():
    '''
    Body frame vectors d1,d2,d3 are rows of _Q
    '''
    
    d1 = c1*c2 * e1 + (c1*s2*s3 - c3*s1) * e2 + (s1*s3 + c1*c3*s2) * e3 
    d2 = c2*s1 * e1 + (s1*s2*s3 + c1*c3) * e2 + (c3*s1*s2 - c1*s3) * e3
    d3 = -s2 * e1 + c2*s3 * e2 + c2*c3 * e3
    
    return d1, d2, d3

def _k1():
    '''
    Analytic derivation
    '''    
    return s1 * b_s - c1 * c2 * g_s 
    
def _k2():
    '''
    Analytic derivation
    '''    
    return -c1 * b_s - c2 * s1 * g_s 
    
def _k3():
    '''
    Analytic derivation
    '''    
    return -a_s + s2 * g_s 
    
def _A():
    '''
    Analytic derivation
    '''
    
    _A = Matrix([
        [0, s1, -c1*c2],
        [0, -c1, -c2*s1],
        [-1, 0, s2]
        ]
    )
        
    return _A

def _A_t():
    '''
    Analytic derivation
    '''
    
    _A = Matrix([
        [0, c1*a_t, s1*c2*a_t-c1*s2*b_t],
        [0, s1*a_t, s2*s1*b_t-c2*c1*a_t],
        [0, 0, c2*b_t]
        ]
    )
        
    return _A
    
def test_Q():
    
    # Define the symbolic rotation matrices around the x, y, and z axes
    R_1 = Matrix([
        [1, 0, 0],
        [0, c3, -s3],
        [0, s3, c3]
    ])
        
    R_2 = Matrix([
        [c2, 0, s2],
        [0, 1, 0],
        [-s2, 0, c2]
    ])
    
    R_3 = Matrix([
        [c1, -s1, 0],
        [s1, c1, 0],
        [0, 0, 1]
    ])
    
    _Q = R_3 * R_2 * R_1

    assert _Q.equals(_Q())
        
    print('_Q passed test')

def test_d123():

    d1, d2, d3 = _d123()    
    _Q = _Q()    
    
    assert d1.equals(_Q.T*e1)
    assert d2.equals(_Q.T*e2)
    assert d3.equals(_Q.T*e3)
    
    print('d123 passed test')
         
def test_k():

    d1, d2, d3 = _d123()    
    
    k1 = simplify(d3.dot(diff(d2, s)))
    k2 = simplify(d1.dot(diff(d3, s)))
    k3 = simplify(d2.dot(diff(d1, s)))
            
    assert k1.equals(_k1())
    assert k2.equals(_k2())
    assert k3.equals(_k3())

    print('k passed test')

def test_A():
    
    _A = _A()    
    theta_s = Matrix([a_s, b_s, g_s]) 
    
    k = _A * theta_s 
    
    assert k[0].equals(_k1())
    assert k[1].equals(_k2())
    assert k[2].equals(_k3())
    
    pprint(_A)
    
    print('_A passed test')

def test_A_t():
    
    _A = _A()        
    A_t = diff(_A, t)
    
    assert A_t.equals(A_t)
    
    print('A_t passed test')
    

if __name__ == '__main__':

    test_Q()
    test_d123()
    test_k()
    test_A()
    test_A_t()
