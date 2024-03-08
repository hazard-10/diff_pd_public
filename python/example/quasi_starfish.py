# First, add a custom function to deformable.h and deformable.cpp that make sure compiles and can modify data
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
def render_quasi_starfish(mesh_file, png_file):
    options = {
        'file_name': png_file,
        'light_map': 'uffizi-large.exr',
        'sample': 4,
        'max_depth': 2,
        'camera_pos': (2, 3, 5),
        'camera_lookat': (1, -1, 0), # roughly the center of starfish obj
        
    }
    renderer = PbrtRenderer(options)
    
    mesh = HexMesh3d()
    mesh.Initialize(mesh_file)
    renderer.add_hex_mesh(mesh, render_voxel_edge=True, color=(.3, .7, .5), transforms=[
        ('r', [90, 1, 0, 0]),  # Rotate 90 degrees around the x-axis
        ('t', [0, 0, 0]),
        ])
    # renderer.add_tri_mesh(Path(root_path) / 'asset/mesh/flat_ground.obj',
    #         texture_img='chkbd_24_0.7', transforms=[
    #             ('s', 4),
    #             ('t', [0, 0, -1]),
    #             ])
    
    
    renderer.render()

'''global parameters'''
asset_folder = Path('/mnt/e/muscleCode/sample_muscle_data/starfish')
default_hex_bin_str = str(asset_folder / 'starfish_demo_voxel.bin')
gt_folder = Path('/mnt/e/wsl_projects/diff_pd_public/python/example/quasi_starfish/init_ground_truth')
render_folder = Path('/mnt/e/wsl_projects/diff_pd_public/python/example/quasi_starfish')
youngs_modulus = 5e5
poissons_ratio = 0.45
la = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
mu = youngs_modulus / (2 * (1 + poissons_ratio))

density = 1e3
thread_ct = 20
dt = 1e-2    
options = {
        'max_pd_iter': 500,
        'thread_ct': 20,
        'abs_tol': 1e-6,
        'rel_tol': 1e-6,
        'verbose': 2,
        'use_bfgs': 1,
        'bfgs_history_size': 10,
        'max_ls_iter': 10,
        
    }
default_hex_mesh = HexMesh3d()
default_hex_mesh.Initialize(default_hex_bin_str)  
deformable_shapeTarget = HexDeformable()
deformable_shapeTarget.Initialize(default_hex_bin_str, density, 'none', youngs_modulus, poissons_ratio)

'''Functions '''
def render_deformable(render_id, q_curr):
    # call save to render bin first before calling this function
    png_file = render_folder / f'starfish_{render_id}_shape_target.png'
    render_bin_path = render_folder / f'starfish_{render_id}_shape_target.bin'
    render_bin_str = str(render_bin_path)
    deformable_shapeTarget.PySaveToMeshFile(q_curr, render_bin_str)
    render_quasi_starfish(render_bin_str, png_file) 
    # remove the render bin file
    os.remove(render_bin_str)
    
def get_idea_q(gt_folder, obj_num):
    gt_json = str(gt_folder)+ '/'+ 'starfish_' + str(obj_num) + '_init_ground_truth.json'
    # pass whole file into a list
    q_ideal = []
    with open(gt_json, 'r') as f:
        content = f.read()
        float_strings = content.strip('[]').split(',')
        q_ideal = [float(x) for x in float_strings]

    # get q_ideal
    deformable_default = HexDeformable()
    deformable_default.Initialize(default_hex_bin_str, density, 'none', youngs_modulus, poissons_ratio)
    act = StdRealVector(0)
    deformable_default.PyGetShapeTargetSMatrixFromDeformation(q_ideal, act)
    act = np.array(act)
    print('actuation initialization size check: ', int(act.shape[0] // 48) == default_hex_mesh.NumOfElements()) # 6 per sample, 8 sample per element
    return act, q_ideal

def forward_pass(act, q_curr): 
    dof = deformable_shapeTarget.dofs() 
    print('deform2 dof:', deformable_shapeTarget.dofs())    
    # found a strong correlation between the stiffness and the convergence of the shape targeting
    q_next = StdRealVector(dof)
    deformable_shapeTarget.PyShapeTargetingForward(q_curr, act, options, q_next ) 
    return q_next

def loss(q_next, q_ideal):
    l2_loss = np.sum((q_next - q_ideal) ** 2)
    l1_loss = np.sum(np.abs(q_next - q_ideal))
    print("l2_loss:", l2_loss)
    print("l1_diff:", l1_loss)
    print("mean diff:", np.mean(np.abs(q_next - q_ideal)))
    return l1_loss

init_obj_num = 30
target_obj_num = 30

# initialize local parameters
act_init_np, _ = get_idea_q(gt_folder, init_obj_num)
_, q_ideal_std = get_idea_q(gt_folder, target_obj_num)
q_curr_std = default_hex_mesh.py_vertices() 
render_deformable('default', q_curr_std)
q_ideal_np = np.array(q_ideal_std)

deformable_shapeTarget.SetShapeTargetStiffness(.01)
# main loop
num_iter = 1
for i in range(num_iter):
    time_ = time.time()
    q_next_std = forward_pass(act_init_np, q_curr_std)
    print("forward pass time:", time.time() - time_)
    time_ = time.time()
    q_next_np = np.array(q_next_std)
    render_deformable(i, q_next_np)
    print("forward render time:", time.time() - time_)
    time_ = time.time()
    loss_ = loss(q_next_np, q_ideal_np)
    break    
    # l2 loss gradient
    dl_dq_next = 2 * (q_next_np - q_ideal_np)
    # dl_dq_next = np.sign(q_next_np - q_ideal_np)
    
    dl_dq = StdRealVector(10) # output
    dl_dact = StdRealVector(10) 
    dl_dmat_w = StdRealVector(10) 
    dl_dact_w = StdRealVector(10) 
    deformable_shapeTarget.PyShapeTargetingBackward(q_curr_std, act_init_np, q_next_np, dl_dq_next, options, dl_dq, dl_dact, dl_dmat_w, dl_dact_w)
    print("backward pass time:", time.time() - time_)
    dl_dact_np = np.array(dl_dact)
    print("dl_dact:", dl_dact_np)