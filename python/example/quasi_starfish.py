# First, add a custom function to deformable.h and deformable.cpp that make sure compiles and can modify data
import sys
sys.path.append('../')
sys.path.append('.python/')
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

# global parameters
obj_num = 90
asset_folder = Path('/mnt/e/muscleCode/sample_muscle_data/starfish')
default_hex_bin_str = str(asset_folder / 'starfish_demo_voxel.bin')
gt_folder = Path('/mnt/e/wsl_projects/diff_pd_public/python/example/quasi_starfish/init_ground_truth')
gt_json = str(gt_folder)+ '/'+ 'starfish_' + str(obj_num) + '_init_ground_truth.json'
render_folder = Path('/mnt/e/wsl_projects/diff_pd_public/python/example/quasi_starfish')
render_bin_path = render_folder / str('starfish_voxel_'+str(obj_num)+'.bin')
render_bin_str = str(render_bin_path)
# Deformable param
youngs_modulus = 5e5
poissons_ratio = 0.45
la = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
mu = youngs_modulus / (2 * (1 + poissons_ratio))

density = 1e3
thread_ct = 20
dt = 1e-2    
options = {
        'max_pd_iter': 1000,
        'thread_ct': 20,
        'abs_tol': 1e-6,
        'rel_tol': 1e-6,
        'verbose': 0,
        'use_bfgs': 1,
        'bfgs_history_size': 10,
        'max_ls_iter': 10,
        
    }
default_hex_mesh = HexMesh3d()
default_hex_mesh.Initialize(default_hex_bin_str)  
def get_idea_q(gt_json):
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
    print(int(act.shape[0] // 48) == default_hex_mesh.NumOfElements())
    return act, q_ideal

def do_shape_targeting(act, q_ideal):
    q_ideal = np.array(q_ideal)
    deformable_shapeTarget = HexDeformable()
    deformable_shapeTarget.Initialize(default_hex_bin_str, density, 'none', youngs_modulus, poissons_ratio)
    deformable_shapeTarget.SetShapeTargetStiffness( 2 * mu)
    print('deform2 dof:', deformable_shapeTarget.dofs())
    dof = deformable_shapeTarget.dofs() 

    png_file = render_folder / 'starfish_default.png'

    q_curr = default_hex_mesh.py_vertices()
    v_curr = np.zeros(deformable_shapeTarget.dofs())
    
    # get a default render
    deformable_shapeTarget.PySaveToMeshFile(q_curr, render_bin_str)
    render_quasi_starfish(render_bin_str, png_file) 
    
    import copy
    num_iter  = 51
    # do shape targeting
    for i in range(num_iter):
        print("iter:", i)
        deformable_shapeTarget.SetShapeTargetStiffness( mu * 2)
        q_next, v_next = StdRealVector(dof), StdRealVector(dof)
        deformable_shapeTarget.PyShapeTargetingForward(q_curr, v_curr, act, dt, options, q_next, v_next) 
        q_next = np.array(q_next)
        v_next = np.array(v_next)
        deformable_shapeTarget.PySaveToMeshFile(q_next, render_bin_str)
        png_file = render_folder / f'starfish_{obj_num}_shape_target_{i}.png'
        # every 5 iterations, save a render
        if i % 5 == 0:
            render_quasi_starfish(render_bin_str, png_file)
        print("curr avg speed:", np.mean(v_curr))
        diff_q_ideal = q_next - q_ideal
        print("avg diff:", np.mean(diff_q_ideal))
        q_curr = copy.deepcopy(q_next)
        v_curr = copy.deepcopy(v_next)
        v_curr *= 0.97
        
        act = StdRealVector(0)
        deformable_shapeTarget.PyGetShapeTargetSMatrixFromDeformation(q_ideal, act)
        act = np.array(act)
    
 
if __name__ == '__main__':
    # get_ideal_q
    act, q_ideal = get_idea_q(gt_json)
    do_shape_targeting(act, q_ideal)
