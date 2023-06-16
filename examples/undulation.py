'''
Created on 14 Jun 2023

@author: amoghasiddhi
'''
# Built-in
from sys import argv

# third-party
import matplotlib.pyplot as plt

# local imports
from minimal_worm import Worm
from minimal_worm import ModelParameter  
from minimal_worm.experiments import simulate_experiment 
from minimal_worm.experiments.undulation import UndulationExperiment
from minimal_worm.experiments import PostProcessor
from minimal_worm.plot import plot_scalar_field

if __name__ == '__main__':

    parser = UndulationExperiment.parameter_parser()        
    param = parser.parse_args(argv[1:])
    
    MP = ModelParameter(param)
    worm = Worm(param.N, param.dt, fdo = param.fdo, quiet=False)
    CS = UndulationExperiment.stw_control_sequence(param)
        
    FS, CS, MP, e = simulate_experiment(worm, param, CS)
    
    if e is not None:
        raise

    # Make video
    # TODO
    
    # Midpoint 2d trajectory
    r_mp = FS.r[:, 1:, FS.r.shape[2] // 2]
    # 2d centre of mass trajectory
    r_com = FS.r[:, 1:].mean(axis=2)    
    # Swimming speed
    U_avg, U, t = PostProcessor.comp_mean_swimming_speed(FS.r, FS.t)
    
    gs = plt.GridSpec(2, 2)
    ax00 = plt.subplot(gs[0,0])
    ax01 = plt.subplot(gs[0,1])
    ax10 = plt.subplot(gs[1,0])
    ax11 = plt.subplot(gs[1,1])

    plot_scalar_field(ax00, CS.k[:, 0, :])
    plot_scalar_field(ax01, FS.k[:, 0, :])

    ax10.plot(r_mp[:, 0], r_mp[:, 1])
    ax10.plot(r_com[:, 0], r_com[:, 1])
    
    ax11.plot(t, U)
    ax11.plot([t[0], t[-1]], [U_avg, U_avg])
    
    plt.show()

    print('Finished example')
