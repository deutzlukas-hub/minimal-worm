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

    model_parser = UndulationExperiment.parameter_parser()        
    model_param = model_parser.parse_args(argv[1:])

    cml_args = {k: v for k, v in vars(model_param).items() 
        if v != model_parser.get_default(k)}
    
    if len(cml_args) != 0: 
        print(cml_args)

    MP = ModelParameter(model_param)
    worm = Worm(model_param.N, model_param.dt, fdo = model_param.fdo, quiet=False)
    CS = UndulationExperiment.stw_control_sequence(model_param)
        
    FS, CS, MP, e = simulate_experiment(worm, model_param, CS)
    
    if e is not None:
        raise

    # Make video
    # TODO
    
    # Midpoint 2d trajectory
    r_mp = FS.r[:, 1:, FS.r.shape[2] // 2]
    # Tail 2d trajectory 
    r_tale = FS.r[:, 1:, -1]
    
    # 2d centre of mass trajectory
    r_com = FS.r[:, 1:].mean(axis=2)    
    # Swimming speed
    U_avg, U, t = PostProcessor.comp_mean_swimming_speed(FS.r, FS.t)
    
    k = PostProcessor.comp_centreline_curvature(FS.r)
    
    gs = plt.GridSpec(3, 2)
    ax00 = plt.subplot(gs[0,0])
    ax10 = plt.subplot(gs[1,0])
    ax20 = plt.subplot(gs[2,0])
    
    ax01 = plt.subplot(gs[0,1])
    ax11 = plt.subplot(gs[1,1])
    ax12 = plt.subplot(gs[2,1])

    k0_min, k0_max = CS.k0[:, 0, :].min(), CS.k0[:, 0, :].max() 
    
    plot_scalar_field(ax00, CS.k0[:, 0, :], v_lim = [k0_min, k0_max])
    plot_scalar_field(ax10, FS.k[:, 0, :], v_lim = [k0_min, k0_max])
    plot_scalar_field(ax20, k, v_lim = [k0_min, k0_max])

    ax01.plot(r_mp[:, 0], r_mp[:, 1])
    ax01.plot(r_tale[:, 0], r_mp[:, 1])
    ax01.plot(r_com[:, 0], r_com[:, 1])
    
    ax11.plot(t, U)
    ax11.plot([t[0], t[-1]], [U_avg, U_avg])
    
    plt.show()

    print('Finished example')
