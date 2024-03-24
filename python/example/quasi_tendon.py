
'''
Directly use the result from routing_tendon example. Read default 
configuration. Then constantly try diffferent combination of
Initializations and target. See if it convege better.
'''

import sys
sys.path.append('../')
import os

from pathlib import Path
import time
import numpy as np
import scipy.optimize
import pickle
import matplotlib.pyplot as plt

from py_diff_pd.common.common import ndarray, create_folder
from py_diff_pd.common.common import print_info, print_ok, print_error, print_warning
from py_diff_pd.common.grad_check import check_gradients
from py_diff_pd.common.display import export_gif
from py_diff_pd.core.py_diff_pd_core import StdRealVector
from py_diff_pd.env.soft_starfish_env_3d import SoftStarfishEnv3d
from py_diff_pd.common.project_path import root_path
from py_diff_pd.core.py_diff_pd_core import HexMesh3d, HexDeformable, StdRealVector
import py_diff_pd.common.hex_mesh as hex

# visualize the hex mesh
from py_diff_pd.common.renderer import PbrtRenderer
def render_tendon(mesh_file, png_file):
    options = {
        'file_name': png_file,
        'light_map': 'uffizi-large.exr',
        'sample': 4,
        'max_depth': 2,
        'camera_pos': (0.4, -1., .25),
        'camera_lookat': (0, .15, .15),
        
    }
    renderer = PbrtRenderer(options)
    
    mesh = HexMesh3d()
    mesh.Initialize(mesh_file)
    renderer.add_hex_mesh(mesh, render_voxel_edge=True, color=(.3, .7, .5), transforms=[
        ('s', 0.4),
    ])
    renderer.add_tri_mesh(Path(root_path) / 'asset/mesh/curved_ground.obj',
        texture_img='chkbd_24_0.7', transforms=[('s', 2)]) 
    
    
    renderer.render()

routing_tendon_folder = Path('/mnt/e/wsl_projects/diff_pd_public/python/example/routing_tendon_3d/pd_eigen/')
output_folder = Path('/mnt/e/wsl_projects/diff_pd_public/python/example/routing_tendon_3d/custom/')

# deformable related parameters. Copied from quasistarfish
youngs_modulus = 5e5
poissons_ratio = 0.45
la = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
mu = youngs_modulus / (2 * (1 + poissons_ratio))

density = 1e3
thread_ct = 20
dt = 1e-2    
fw_options = {
        'max_pd_iter': 500,
        'thread_ct': 20,
        'abs_tol': 1e-6,
        'rel_tol': 1e-6,
        'verbose': 0,
        'use_bfgs': 1,
        'bfgs_history_size': 10,
        'max_ls_iter': 10,     
    }
bw_options = {
        'max_pd_iter': 500,
        'thread_ct': 20,
        'abs_tol': 1e-6,
        'rel_tol': 1e-6,
        'verbose': 0,
        'use_bfgs': 1,
        'bfgs_history_size': 10,
        'max_ls_iter': 10,        
    }
# prepare mesh and deformable
default_num = 0
default_bin_file = routing_tendon_folder / f'{default_num:04}.bin'
tendon_hex_mesh = HexMesh3d()
tendon_hex_mesh.Initialize(str(default_bin_file))
q_default_std = tendon_hex_mesh.py_vertices()

        

def get_deformed_q_np(frame_num): 
    num = f'{frame_num:04}'
    target_bin_file = routing_tendon_folder / f'{num}.bin'
    local_deformed_mesh = HexMesh3d()
    local_deformed_mesh.Initialize(str(target_bin_file))
    q_ = local_deformed_mesh.py_vertices()
    q_np = np.array(q_)
    return q_np

def get_deformed_act(frame_num, q_ideal):  
    local_deformable = HexDeformable()
    local_deformable.Initialize(str(default_bin_file), density, 'none', youngs_modulus, poissons_ratio)
    act = StdRealVector(0)
    local_deformable.PyGetShapeTargetSMatrixFromDeformation(q_ideal, act)
    act = np.array(act)
    return act


deformable_tendon = HexDeformable()
deformable_tendon.Initialize(str(default_bin_file), density, 'none', youngs_modulus, poissons_ratio)
deformable_tendon_2 = HexDeformable()
deformable_tendon_2.Initialize(str(default_bin_file), density, 'none', youngs_modulus, poissons_ratio)


def find_root_and_set_to_boundary():
    q_default_np = np.array(q_default_std)
    q_length = q_default_np.shape[0]
    vertex_num = q_length // 3
    min_z = q_default_np[2]
    minCount = 0
    for i in range(vertex_num):
        if q_default_np[i*3+2] < min_z:
            min_z = q_default_np[i*3+2]
    for i in range(vertex_num):
        if q_default_np[i*3+2] == min_z:
            deformable_tendon.SetDirichletBoundaryCondition(3*i, q_default_np[i*3])
            deformable_tendon.SetDirichletBoundaryCondition(3*i+1, q_default_np[i*3+1])
            deformable_tendon.SetDirichletBoundaryCondition(3*i+2, q_default_np[i*3+2])
            deformable_tendon_2.SetDirichletBoundaryCondition(3*i, q_default_np[i*3])
            deformable_tendon_2.SetDirichletBoundaryCondition(3*i+1, q_default_np[i*3+1])
            deformable_tendon_2.SetDirichletBoundaryCondition(3*i+2, q_default_np[i*3+2])
            print(f"Set {i} to boundary, z: {q_default_np[i*3+2]}, y: {q_default_np[i*3+1]}, x: {q_default_np[i*3]}")
            minCount += 1
    assert minCount == 25

def forward_pass(act, q_curr_np):
    q_next_std = StdRealVector(0)
    deformable_tendon.PyShapeTargetingForward(q_curr_np, act, fw_options, q_next_std)
    q_next_np = np.array(q_next_std)
    return q_next_np

def backward_pass(q_init_np, act_init_np, q_next_np, dl_dq_next): 
    dl_dq = StdRealVector(10)  
    dl_dact = StdRealVector(10) 
    dl_dmat_w = StdRealVector(10) 
    dl_dact_w = StdRealVector(10) 
    deformable_tendon.PyShapeTargetingBackward(q_init_np, act_init_np, q_next_np, dl_dq_next, bw_options, dl_dq, dl_dact, dl_dmat_w, dl_dact_w)
    return np.array(dl_dact)

def get_loss(q_curr_np, q_ideal_np):
    l2_loss = np.sum((q_curr_np - q_ideal_np)**2)
    l1_mean = np.mean(np.abs(q_curr_np - q_ideal_np))
    print("l2_loss:", l2_loss)
    print("l1_mean:", l1_mean)
    return 2 * (q_curr_np - q_ideal_np), l2_loss

def visualize(q_curr_np, name):
    png_file = output_folder / f'{name}.png'
    render_bin_file = output_folder / f'{name}.bin'
    deformable_tendon.PySaveToMeshFile(q_curr_np, str(render_bin_file))
    render_tendon(str(render_bin_file), str(png_file))
    # os.remove(render_bin_file)

find_root_and_set_to_boundary()  

# deformable_tendon_2.SetupProjectiveDynamicsSolver('pd_eigen', dt, fw_options)
# print("setup 1")

# deformable_tendon.SetupShapeTargetingSolver(fw_options)
# print("setup 2")


q_curr_np = np.array(q_default_std)
act_frame = 6
ideal_frame = 8

q_act_ideal_np = get_deformed_q_np(act_frame)
act_np = get_deformed_act(act_frame, q_act_ideal_np)
q_ideal_np = get_deformed_q_np(ideal_frame)
iter_count = 0
global_loss = 0

def sim_loss_n_grad(act_np):
    global global_loss
    global iter_count
    iter_count += 1
    print("iter_count:", iter_count)
    q_next_np = forward_pass(act_np, q_curr_np)
    visualize(q_next_np, 'iter_'+str(iter_count))
    dl_dq_next, l2_loss = get_loss(q_next_np, q_ideal_np)
    dl_dact_np = backward_pass(q_curr_np, act_np, q_next_np, dl_dq_next) 
    global_loss = l2_loss
    return l2_loss, dl_dact_np
 

result = scipy.optimize.minimize(sim_loss_n_grad, np.copy(act_np), jac=True, method='L-BFGS-B', options={'ftol': 1e-4})
print(f'Optimization finished in {iter_count} iterations, loss: {global_loss}, tolerance: {result.fun}')
