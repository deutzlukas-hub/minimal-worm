'''
Created on 19 Jul 2023

@author: amoghasiddhi
'''

import matplotlib.pyplot as plt
import numpy as np

def kinematic_data():

    mu_arr_U = np.array([1.02, 2.02, 3.68, 12.27, 151.89, 218.18, 309.68, 434.62, 645.85])
    U_arr = np.array([0.38, 0.3, 0.37, 0.31, 0.27, 0.25, 0.23, 0.22, 0.18])

    mu_arr_f = np.array([ 1., 1.98, 3.62, 12.08, 147.73, 216.56, 310.18, 434.09, 643.76])
    f_arr = np.array([1.84, 1.47, 2.08, 1.43, 1.4, 1.3, 1.39, 1.16, 1.05])

    mu_arr_A = np.array([1.01, 2., 3.65, 12.26, 148.32, 217.17, 310.38,433.76, 642.42])    
    A_arr = np.array([0.26, 0.31, 0.33, 0.29, 0.24, 0.24, 0.22, 0.21,0.21])
    
    mu_arr_c = np.array([1., 1.96, 3.58, 11.94, 146.03, 214.06, 306.6, 429.09, 643.76 ])
    c_arr = np.array([3.77, 3.45, 4.39, 3.05, 2.22, 2.06, 2.06, 1.86, 1.72])

    mu_arr = np.mean(np.vstack([mu_arr_U, mu_arr_f, mu_arr_c, mu_arr_A]), axis = 0)

    return mu_arr, U_arr, f_arr, A_arr, c_arr 
    
def reproduce_figure_3():
    
    gs = plt.GridSpec(2, 2)
    ax00 = plt.subplot(gs[0, 0])
    ax10 = plt.subplot(gs[1, 0])
    ax01 = plt.subplot(gs[0, 1])
    ax11 = plt.subplot(gs[1, 1])
    
    mu_arr, U_arr, f_arr, A_arr, c_arr = kinematic_data()    
    ax00.semilogx(mu_arr, U_arr, 'o', ms = 10)
    ax00.set_xlim([0.8, 1e3])
    ax00.set_ylim([0, 0.5])

    ax00.set_ylabel('U', fontsize = 20)

    ax10.semilogx(mu_arr, f_arr, 'o', ms = 10)
    ax10.set_xlim([0.8, 1e3])
    ax10.set_ylim([0, 3.2])
    ax10.set_ylabel('f', fontsize = 20)
    ax10.set_xlabel('\log(\eta)', fontsize = 20)

    ax01.semilogx(mu_arr, A_arr, 'o', ms = 10)
    ax01.set_xlim([0.8, 1e3])
    ax01.set_ylim([0, 0.45])
    ax01.set_ylabel('A', fontsize = 20)

    ax11.semilogx(mu_arr, c_arr, 'o', ms = 10)
    ax11.set_xlim([0.8, 1e3])
    ax11.set_ylim([0, 7])
    ax11.set_ylabel('c', fontsize = 20)
    ax11.set_xlabel('\log(\eta)', fontsize = 20)

    plt.tight_layout()

    plt.show()
    
    return

if __name__ == '__main__':
    
    reproduce_figure_3()
    

