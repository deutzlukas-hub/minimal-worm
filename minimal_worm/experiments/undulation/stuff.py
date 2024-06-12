'''
Created on 4 Jul 2023

@author: amoghasiddhi
'''
import numpy as np
import matplotlib.pyplot as plt

def c_A():
    
    c_arr = np.linspace(0.4, 1.4, int(1e2))
    lam_arr = np.linspace(0.5, 2.0, int(1e2))
    
    lam_grid, c_grid = np.meshgrid(lam_arr, c_arr)        
    q_grid = (2 * np.pi) / lam_grid 
    
    A_grid = c_grid * q_grid
    
    gs = plt.GridSpec(2,1)

    # Create axes objects
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    
    # Scatter plot on ax0
    ax0.scatter(lam_grid.flatten(), c_grid.flatten(), marker='s', s=50)
    ax0.set_xlabel('lambda')
    ax0.set_ylabel('c')
    
    # Scatter plot on ax1
    ax1.scatter(lam_grid.flatten(), A_grid.flatten(), marker='s', s=50)
    ax1.axhline(10, linestyle = '--', color = 'red')
    
    ax1.set_xlabel('lambda')
    ax1.set_ylabel('A')
    
    plt.show()


def drag_coefficients(param):

    a = 2*param.R/param.L0 # slenderness parameter        
    
    # Linear drag coefficients
    c_t = 2 * np.pi / (np.log(2/a) - 0.5)
    c_n = 4 * np.pi / (np.log(2/a) + 0.5)

    d3 = np.array([0,0,1])
    
    # Angular drag coefficients
    y_t = 0.25 * np.pi * a**2
    y_n = np.pi * a**2
                      
    D = y_t / c_t 
    C = c_n / c_t 
    Y = y_n / y_t 


if __name__ == '__main__':
    
    c_A()
    
    
    
    
    
    
    


    
    
    
