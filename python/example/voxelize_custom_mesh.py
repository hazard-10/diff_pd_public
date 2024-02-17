import time
time_check = time.time()
import sys
sys.path.append('../')
import os


from pathlib import Path
import time
import numpy as np
import scipy.optimize
import pickle
import matplotlib.pyplot as plt
import json

from py_diff_pd.common.common import ndarray, create_folder
from py_diff_pd.common.common import print_info, print_ok, print_error, print_warning
from py_diff_pd.common.grad_check import check_gradients
from py_diff_pd.common.display import export_gif
from py_diff_pd.core.py_diff_pd_core import StdRealVector
from py_diff_pd.env.soft_starfish_env_3d import SoftStarfishEnv3d
from py_diff_pd.common.project_path import root_path
from py_diff_pd.core.py_diff_pd_core import HexMesh3d, HexDeformable, StdRealVector
import py_diff_pd.common.hex_mesh as hex


print('import time:', time.time()-time_check)
time_check = time.time()

asset_folder = Path('/mnt/e/muscleCode/sample_muscle_data/starfish')
# list all the files in the folder
input_obj = asset_folder / 'starfish_demo_voxel.obj'
mesh_bin = asset_folder / 'starfish_demo_voxel.bin'
voxel_output = asset_folder / 'starfish_demo_voxel_output.obj'
json_file_path = asset_folder / 'starfish_demo_48x9x46.json'

# Load JSON file.
with open(json_file_path, 'r') as file:
    data = json.load(file)

# Extract voxel data.
dimensions = data['dimension']
voxels_data = data['voxels']

# Determine the dimensions of the voxel space.
max_x =int( dimensions[0]['width']) + 1
max_y =int( dimensions[0]['height']) + 1
max_z =int( dimensions[0]['depth']) + 1

# Create a 3D numpy array filled with zeros.
voxels = np.zeros((max_x, max_y, max_z), dtype=np.int32)

# Fill the array with voxel data.
for voxel in voxels_data:
    x = int(voxel['x'])
    y = int(voxel['y'])
    z = int(voxel['z'])
    voxels[x, y, z] = 1
print('loading voxel time:', time.time()-time_check)

time_check = time.time()
origin = ndarray([0, 0, 0])
hex.generate_hex_mesh(voxels, 0.05, origin, mesh_bin)
print('generate hex mesh from voxel time:', time.time()-time_check)

time_check = time.time()
mesh = HexMesh3d()
mesh.Initialize(str(mesh_bin))
print('initialize hex mesh time:', time.time()-time_check)

time_check = time.time()
hex.hex2obj(mesh, voxel_output, 'tri')
print('hex2obj time:', time.time()-time_check)

print_info('starfish processed: elements: {}, dofs: {}'.format(mesh.NumOfElements(), mesh.NumOfVertices() * 3))

