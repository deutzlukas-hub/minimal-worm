'''
Created on 9 Feb 2023

@author: lukas
'''
from pathlib import Path

result_path = Path('../results/')
data_path = result_path / 'data/undulation'
log_dir = data_path / 'logs'
sim_dir = data_path / 'simulations'
sweep_dir = data_path / 'parameter_sweeps'

fig_dir = result_path / 'figures/undulation'
video_dir = result_path / 'videos/undulation'

if not data_path.is_dir(): data_path.mkdir(parents = True, exist_ok = True)
if not sim_dir.is_dir(): sim_dir.mkdir(exist_ok = True)
if not log_dir.is_dir(): log_dir.mkdir(exist_ok = True)
if not sweep_dir.is_dir(): sweep_dir.mkdir(exist_ok = True)
if not fig_dir.is_dir(): fig_dir.mkdir(parents = True, exist_ok = True)
if not video_dir.is_dir(): video_dir.mkdir(parents = True, exist_ok=True)
