
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

# visualization functions
from py_diff_pd.common.renderer import PbrtRenderer
def render_quasi_starfish(mesh_file, png_file):
    options = {
        'file_name': png_file,
        'light_map': 'uffizi-large.exr',
        'sample': 4,
        'max_depth': 2,
        'camera_pos': (5, 5, 5),
        'camera_lookat': (0, 0, 0.1), # roughly the center of starfish obj
        
    }
    renderer = PbrtRenderer(options)
    
    mesh = HexMesh3d()
    mesh.Initialize(mesh_file)
    renderer.add_hex_mesh(mesh, render_voxel_edge=True, color=(.3, .7, .5),  transforms=[
                ('r', [1.6, 1, 0, 0]),  # Rotate 90 degrees around the x-axis 
                ('t', [-1, 1.5, 0]),
                ])
    renderer.add_tri_mesh(Path(root_path) / 'asset/mesh/flat_ground.obj',
            texture_img='chkbd_24_0.7', transforms=[
                ('s', 16),
                ('t', [0, 0, -1]),
                ])       
    renderer.render()
    
def render_tri_starfish(tri_mesh_path, png_file):
    options = {
        'file_name': png_file,
        'light_map': 'uffizi-large.exr',
        'sample': 4,
        'max_depth': 2,
        'camera_pos': (5, 5, 5),
        'camera_lookat': (0, 0, 0.1), # roughly the center of starfish obj
        
    }
    renderer = PbrtRenderer(options)
    renderer.add_tri_mesh(Path(tri_mesh_path),
            texture_img='starfish_1_diffuse.png', transforms=[
                ('r', [1.6, 1, 0, 0]),  # Rotate 90 degrees around the x-axis 
                ('t', [-1, 1.5, 0]),
                ])
    renderer.add_tri_mesh(Path(root_path) / 'asset/mesh/flat_ground.obj',
            texture_img='chkbd_24_0.7', transforms=[
                ('s', 16),
                ('t', [0, 0, -1]),
                ])    
    renderer.render()
        
def render_tri_starfish_with_angle(tri_mesh_path, png_file, r_angle):
    options = {
        'file_name': png_file,
        'light_map': 'uffizi-large.exr',
        'sample': 4,
        'max_depth': 2,
        'camera_pos': (5, 5, 5),
        'camera_lookat': (0, 0, 0.1), # roughly the center of starfish obj
        
    }
    renderer = PbrtRenderer(options)
    renderer.add_tri_mesh(Path(tri_mesh_path),
            texture_img='starfish_1_diffuse.png', transforms=[
                ('r', [1.6, 1, 0, 0]),  # Rotate 90 degrees around the x-axis
                # ('r', [-0.3, 0, 0, 1]),  # Rotate 90 degrees around the z
                ('t', [-1, 1.5, 0]),
                # ('s', 1)
                ])
    renderer.add_tri_mesh(Path(root_path) / 'asset/mesh/flat_ground.obj',
            texture_img='chkbd_24_0.7', transforms=[
                ('s', 16),
                ('t', [0, 0, -1]),
                ])    
    renderer.render()

# Utility functions with obj
def load_tri_starfish_obj(file_name):
    vertex_lines,first_lines , rest_lines = [],[],[]
    count = 0
    with open(file_name, 'r') as file:
        for line in file:
            if count <3:
                first_lines.append(line)
            elif count >= 1085: # hardcoded for starfish datasets
                rest_lines.append(line)
            else:
                parts = line.strip().split()
                xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
                vertex_lines.append(xyz)
            count += 1
    return vertex_lines, first_lines, rest_lines

def load_hex_starfish_obj(input_dir, file_name):
    vertex_lines, rest_lines = [],[]
    with open(os.path.join(input_dir, file_name), 'r') as file:
        for line in file:
            if not line.startswith('v '):
                rest_lines.append(line)
            else:
                parts = line.strip().split()
                xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
                vertex_lines.append(xyz) 
    return vertex_lines, rest_lines

# overwrite the starfish obj file with new vertex positions
def write_tri_starfish_obj(output_name, first_lines, rest_lines, new_verts):
    # new verts [[x,y,z]]
    with open(output_name, 'w') as file:
        for line in first_lines:
            file.write(line)
        for v in new_verts:
            file.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for line in rest_lines:
            file.write(line)

'''global parameters'''
asset_folder = Path('/mnt/e/muscleCode/sample_muscle_data/starfish')
default_hex_bin_str = str(asset_folder / 'starfish_demo_voxel.bin')
gt_folder = Path('/mnt/e/wsl_projects/diff_pd_public/python/example/quasi_starfish/stretched_gt_corotate_volume/')
render_folder = Path('/mnt/e/wsl_projects/diff_pd_public/python/example/quasi_starfish/renders')
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
default_hex_mesh = HexMesh3d()
default_hex_mesh.Initialize(default_hex_bin_str)  
deformable_shapeTarget = HexDeformable()
deformable_shapeTarget.Initialize(default_hex_bin_str, density, 'none', youngs_modulus, poissons_ratio)

q_init_np = np.array(default_hex_mesh.py_vertices()) 
trilinear_weights_mapping, gt_verts_pos = {}, []
iter_count = 0
final_interp_verts = []
final_hex_verts = []

'''Render calls '''
def render_deformable(render_id, q_curr_np):
    # call save to render bin first before calling this function
    png_file = render_folder / 'quasi' / 'hex_renders' / f'starfish_{render_id}.png'
    render_bin_path = render_folder / 'quasi' / 'bins' / f'starfish_{render_id}.bin'
    render_bin_str = str(render_bin_path)
    deformable_shapeTarget.PySaveToMeshFile(q_curr_np, render_bin_str)
    render_quasi_starfish(render_bin_str, png_file) 
    # remove the render bin file
    # os.remove(render_bin_str)
    
def render_default_obj(obj_path, id):
    png_file = render_folder / 'default' / f'default_starfish_{id}.png'
    render_tri_starfish(obj_path, png_file)
    
def construct_then_render_obj(gt_obj_path, new_verts, id): # new verts nx3
    _, first_lines, rest_lines = load_tri_starfish_obj(gt_obj_path)
    obj_output = render_folder / 'quasi' / 'objs' / f'starfish_surface_{id}.obj'
    write_tri_starfish_obj(obj_output, first_lines, rest_lines, new_verts)
    png_file = render_folder / 'quasi' / 'renders' / f'starfish_surface_{id}.png'
    render_tri_starfish(obj_output, png_file)

'''Load ground truth data'''
def get_idea_q_and_act(gt_folder, obj_num):
    gt_json = str(gt_folder)+ '/'+ 'starfish_' + str(obj_num) + '_init_ground_truth.json'
    # pass whole file into a list
    q_ideal = []
    with open(gt_json, 'r') as f:
        content = f.read()
        float_strings = content.strip('[]').split(',')
        q_ideal = [float(x) for x in float_strings]
    q_ideal = np.array(q_ideal)

    # get q_ideal
    deformable_default = HexDeformable()
    deformable_default.Initialize(default_hex_bin_str, density, 'none', youngs_modulus, poissons_ratio)
    act = StdRealVector(0)
    deformable_default.PyGetShapeTargetSMatrixFromDeformation(q_ideal, act)
    act = np.array(act)
    # print('actuation initialization size check: ', int(act.shape[0] // 48) == default_hex_mesh.NumOfElements()) # 6 per sample, 8 sample per element
    return act, q_ideal

def get_trilinear_mapping():
    import json
    map_file = Path('quasi_starfish/ground_truth/trilinear_weights.json')
    tri_map = json.load(open(map_file, 'r'))
    trilinear_weights_mapping = tri_map['trilinear_weights_mapping']
    return trilinear_weights_mapping
    
def get_default_surface_verts(target_num):
    import json
    gt_vert_file = Path(f'quasi_starfish/ground_truth/default_pos/starfish_obj_{target_num}_verts.json')
    gt_verts_mapping = json.load(open(gt_vert_file, 'r'))
    gt_verts = gt_verts_mapping['starfish']
    return gt_verts

'''Set boundary conditions'''
def set_dirichlet_boundary():
    # the top inner circle  
    dirichlet_v_id = [2805] #, 2625, 2806, 3002, 3001, 3000, 2804, 2623, 2624]
    for v in dirichlet_v_id: 
        deformable_shapeTarget.SetDirichletBoundaryCondition(3*v, q_init_np[3*v])
        deformable_shapeTarget.SetDirichletBoundaryCondition(3*v+1, q_init_np[3*v+1])
        deformable_shapeTarget.SetDirichletBoundaryCondition(3*v+2, q_init_np[3*v+2])

 

def render_cubes(verts_list, element_list, color_list, png_file):
    options = {
        'file_name': png_file,
        'light_map': 'uffizi-large.exr',
        'sample': 4,
        'max_depth': 2,
        'camera_pos': (5, 5, 5),
        'camera_lookat': (0, 0, 0.1), # roughly the center of starfish obj
        
    }
    renderer = PbrtRenderer(options)
    
    element_count = len(element_list)
    for i in range(element_count):
        v_l = verts_list[i]
        el = element_list[i]
        mesh = HexMesh3d()
        mesh.PyInitialize(v_l, el)  
        renderer.add_hex_mesh(mesh, render_voxel_edge=False, color=color_list[i],  transforms=[
                    ('r', [1.6, 1, 0, 0]),  # Rotate 90 degrees around the x-axis 
                    ('t', [-1, 1.5, 0]),
                    ])
    renderer.add_tri_mesh(Path(root_path) / 'asset/mesh/flat_ground.obj',
            texture_img='chkbd_24_0.7', transforms=[
                ('s', 16),
                ('t', [0, 0, -1]),
                ])       
    renderer.render()
    
q_default_np = np.array(default_hex_mesh.py_vertices())

def get_element_verts_pos(element_id):
    verts = []
    v_ids = default_hex_mesh.py_element(element_id)
    for i in range(8):
        verts.append(q_default_np[3*v_ids[i]])
        verts.append(q_default_np[3*v_ids[i]+1])
        verts.append(q_default_np[3*v_ids[i]+2])
    return verts


element_list = []
v_list = []
color_list = []
# for e_id in range(default_hex_mesh.NumOfElements()):
#     e_example = [i for i in range(8)]
#     element_list.append(e_example)
#     v_list.append(get_element_verts_pos(e_id))
#     random_color = tuple(np.random.rand(3))
# # print(element_list, v_list_0)
#     color_list.append(random_color)
# png_file = render_folder / f'cubes_{e_id}.png'
# render_cubes(v_list, element_list, color_list, png_file)


for frame_num in range(90, 91):
    iter_count = 0
    deformable_shapeTarget = HexDeformable()
    deformable_shapeTarget.Initialize(default_hex_bin_str, density, 'none', youngs_modulus, poissons_ratio)
     
    # initialize local parameters
    act_init_np, q_ideal_np = get_idea_q_and_act(gt_folder, frame_num)
    force = StdRealVector(0)
    deformable_shapeTarget.PyShapeTargetForce(q_init_np, act_init_np, force)
    force = np.array(force)
    
    print('force shape: ', force.shape)
    print('force average: ', np.mean(force))
    print('force max: ', np.max(force))
    print('force min: ', np.min(force))
    
    abs_force = np.abs(force)
    cap = 2
    norm_max = np.minimum(abs_force, cap)
    normalized_force = norm_max / cap
    color_list = np.array([[nf, 0, 1] for nf in normalized_force])
    
    for e_id in range(default_hex_mesh.NumOfElements()):
        e_example = [i for i in range(8)]
        element_list.append(e_example)
        v_list.append(get_element_verts_pos(e_id))
        # random_color = tuple(np.random.rand(3))
        # print(element_list, v_list_0)
        # color_list.append(random_color)
    png_file = render_folder / f'cubes_{e_id}.png'
    render_cubes(v_list, element_list, color_list, png_file)
    
