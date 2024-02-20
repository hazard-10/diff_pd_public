
import sys
import os
from pathlib import Path
import time
import numpy as np
from py_diff_pd.core.py_diff_pd_core import HexMesh3d, HexDeformable, StdRealVector
import py_diff_pd.common.hex_mesh as hex
from py_diff_pd.core.py_diff_pd_core import StdRealVector, StdIntVector
# from py_diff_pd.common.project_path import root_path

# Implement one iteration of the PD with zero rest length
# Then use the Deformation gradiant's diagonal as A
import os
import copy
file_count = 120
input_dir =  "/mnt/e/muscleCode/sample_muscle_data/starfish/"
output_dir = "E:/muscleCode/sample_muscle_data/starfish/"
# read star fish obj file trimesh
# open path / starfish_frame_1.obj, parse lines starting with 'v ' and store xyz in a list

def load_tri_starfish_obj(input_dir, file_name):
    vertex_lines,first_lines , rest_lines = [],[],[]
    count = 0
    with open(os.path.join(input_dir, file_name), 'r') as file:
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
def write_tri_starfish_obj(output_dir, output_name, first_lines, rest_lines, new_verts):
    # new verts [[x,y,z]]
    with open(os.path.join(output_dir, output_name), 'w') as file:
        for line in first_lines:
            file.write(line)
        for v in new_verts:
            file.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for line in rest_lines:
            file.write(line)
 
# Env _init_ 
# folder
asset_folder = Path('/mnt/e/muscleCode/sample_muscle_data/starfish')
mesh_bin = asset_folder / 'starfish_demo_voxel.bin'
mesh_bin_str = str(mesh_bin)
voxel_output = asset_folder / 'starfish_demo_voxel_output.obj'
json_file_path = asset_folder / 'starfish_demo_48x9x46.json'
render_folder = Path('/mnt/e/wsl_projects/diff_pd_public/python/example/quasi_starfish')
render_bin_path = render_folder / 'starfish_voxel_90.bin'
render_bin_str = str(render_bin_path)

# Deformable param
youngs_modulus = 5e5
poissons_ratio = 0.45
la = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
mu = youngs_modulus / (2 * (1 + poissons_ratio))
density = 1e3
thread_ct = 4
dt = 1e-3

pd_opt = { 'max_pd_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': 20,
        'use_bfgs': 1, 'bfgs_history_size': 10 }
foward_method = 'pd_eigen'

# Initialize active objects and param related to deformable
default_hex_mesh = HexMesh3d()
default_hex_mesh.Initialize(mesh_bin_str) 
deformable = HexDeformable()
deformable.Initialize(mesh_bin_str, density, 'none', youngs_modulus, poissons_ratio)
deformable.AddPdEnergy('corotated', [2 * mu,], [])

dof = deformable.dofs()
act_maps = np.zeros(deformable.act_dofs())

q_curr = default_hex_mesh.py_vertices()
v_curr = np.zeros(deformable.dofs())
q_next, v_next, contact_index = StdRealVector(dof), StdRealVector(dof), StdIntVector(0)

# Compute initial f_ext
obj_90_verts, _, _ = load_tri_starfish_obj(input_dir, "starfish_90.obj")
obj_1_verts, _, _ = load_tri_starfish_obj(input_dir, "starfish_1.obj")
# load mapping
hex_to_one_json = input_dir + 'hex_to_one.json'
one_to_hex_json = input_dir + 'one_to_hex.json'
hex_to_one_mapping = {}
one_to_hex_mapping = {}
import json
with open(hex_to_one_json, 'r') as f:
    hex_to_one_mapping = json.load(f)
with open(one_to_hex_json, 'r') as f:
    one_to_hex_mapping = json.load(f)
# accumulate forces on hex vertices

def accumulate_forces_on_hex(obj_90_verts, obj_1_verts, close_flag): # need to update obj_1_verts in global scope
    f_ext = np.zeros(deformable.dofs())
    k = 1e3
    if close_flag:
        k = 1
    forces_on_verts = (np.array(obj_90_verts) - np.array(obj_1_verts) ) * k
    
    for k, verts in hex_to_one_mapping.items():
        for v in verts:
            k = int(k) // 3
            v = int(v)
            # write force to f_ext
            current_force = forces_on_verts[v]
            f_ext[k*3] += current_force[0]
            f_ext[k*3+1] += current_force[1]
            f_ext[k*3+2] += current_force[2]
    return f_ext
    
# visualize the hex mesh
from py_diff_pd.common.renderer import PbrtRenderer
png_file = render_folder / 'starfish_default.png'
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
    renderer.add_tri_mesh(Path(root_path) / 'asset/mesh/flat_ground.obj',
            texture_img='chkbd_24_0.7', transforms=[
                ('s', 4),
                ('t', [0, 0, -1]),
                ])
    
    
    renderer.render()

render_quasi_starfish(mesh_bin_str, png_file)

# forward pass parameters  
# compute f_ext from zero rest length springs 
# fxi = -k(length - rest_length) * (xi - xj) / length  
# when rest_length = 0,  fxi = -k * (xi-xj) 
# Initial k guess is 1e3, same as density, so F, m is about the same level
# Then define

# rough pipline 
# initialize related parameters
# for i in range(10):
#     q_next, v_next, contact_index from pyforward
#     visualize q_next, check if v is close to 0
#     compute new diff , update forces accumulation, set as new f_ext
num_iters = 13
close_flag = False
for i in range(num_iters):    
    f_ext = accumulate_forces_on_hex(obj_90_verts, obj_1_verts, close_flag)
    deformable.PyForward(foward_method, q_curr, v_curr, act_maps, f_ext, dt, pd_opt, q_next, v_next, contact_index)
    # render
    png_file = render_folder / f'starfish_90_init_{i}.png'
    deformable.PySaveToMeshFile(q_next, render_bin_str)
    render_quasi_starfish(render_bin_str, png_file)
    # iter obj_1 verts and apply q_diff to them    
    q_diff = np.array(q_next) - np.array(q_curr)
    for v in range(len(obj_1_verts)):
        hex_idx = int(one_to_hex_mapping[str(v)])
        obj_1_verts[v][0] += q_diff[hex_idx]
        obj_1_verts[v][1] += q_diff[hex_idx+1]
        obj_1_verts[v][2] += q_diff[hex_idx+2]
    verts_diff = np.array(obj_1_verts) - np.array(obj_90_verts)
    l2_diff = np.linalg.norm(verts_diff)
    print(f'iter {i} l2_diff {l2_diff}')
    # print avg speed
    avg_speed = np.mean(np.abs(v_next))
    print(f'iter {i} avg_speed {avg_speed}')
    if l2_diff < 1.3:
        close_flag = True
    
    q_curr = q_next
    v_curr = v_next
    q_next, v_next, contact_index = StdRealVector(dof), StdRealVector(dof), StdIntVector(0)
