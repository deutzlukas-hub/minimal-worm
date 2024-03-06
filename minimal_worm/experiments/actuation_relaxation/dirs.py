'''
Created on 9 Feb 2023

@author: lukas
'''
from pathlib import Path

result_path = Path('../results/')
data_path = result_path / 'data/actuation_relaxation'
log_dir = data_path / 'logs'
sim_dir = data_path / 'simulations'
sweep_dir = data_path / 'parameter_sweeps'

fig_dir = result_path / 'figures/actuation_relaxation'
video_dir = result_path / 'videos/actuation_relaxation'

if not data_path.is_dir(): data_path.mkdir(parents = True, exist_ok = True)
if not sim_dir.is_dir(): sim_dir.mkdir(exist_ok = True)
if not log_dir.is_dir(): log_dir.mkdir(exist_ok = True)
if not sweep_dir.is_dir(): sweep_dir.mkdir(exist_ok = True)
if not fig_dir.is_dir(): fig_dir.mkdir(parents = True, exist_ok = True)
if not video_dir.is_dir(): video_dir.mkdir(parents = True, exist_ok=True)

storage_dir = Path('/home/lukas/storage/minimal-worm/results/actuation_relaxation')  
base_path = Path('/home/lukas/git/minimal-worm/minimal_worm/experiments/actuation_relaxation')

storage_dir = storage_dir.relative_to('/home/lukas/')
base_path = base_path.relative_to('/home/lukas/')
relative_path = Path(*(['../']*len(base_path.parts)))

storage_dir = relative_path / storage_dir

def create_storage_dir():
        
    storage_dir.mkdir(parents = True, exist_ok = True)
                    
    log_dir = storage_dir / 'logs'
    sim_dir = storage_dir / 'simulations'
    sweep_dir = storage_dir / 'parameter_sweeps'

    log_dir.mkdir(exist_ok = True)
    sim_dir.mkdir(exist_ok = True)
    sweep_dir.mkdir(exist_ok = True)
    
    return log_dir, sim_dir, sweep_dir
    
    