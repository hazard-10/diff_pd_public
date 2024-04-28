'''
Inference of VAE model without simulation layer
Training code in vae_train_wo_simulator.py.
Because the output is a, for visualization and validation, 
we still need to pass it into sim to get a pos, only that we don't 
backpropagate through the sim.
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
from py_diff_pd.common.renderer import PbrtRenderer


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import numpy as np

# Helper functions
import quasi_starfish as shape_target_sim
import vae_train_wo_sim as vae   

print("Running VAE Testing")
# load the test data
test_id_list = [i for i in range(41, 61 )]
# set output path
render_folder = Path('/mnt/e/wsl_projects/diff_pd_public/python/example/vae_starfish_output/')

# initialize the model
model_path = '/mnt/e/wsl_projects/diff_pd_public/python/example/vae_starfish_output/vae.pth'
model = vae.VAE()
model.load_state_dict(torch.load(model_path))
model.eval()

print("Model loaded from: ", model_path, ', Running in eval mode')
# inference. Most code can be found in quasi_starfish.py
for test_id in tqdm(test_id_list):
    # initialize the simulator for forward pass
    deformable_shapeTarget = HexDeformable()
    deformable_shapeTarget.Initialize(shape_target_sim.default_hex_bin_str, 
                                      shape_target_sim.density, 'none', 
                                      shape_target_sim.youngs_modulus, 
                                      shape_target_sim.poissons_ratio)
    # prepare variables for forward pass
    shape_target_sim.get_trilinear_mapping() # fixed in original code
    gt_verts_pos = shape_target_sim.get_default_surface_verts(test_id)
    gt_obj_path_str = '/mnt/e/muscleCode/sample_muscle_data/starfish/starfish_' + str(test_id) + '.obj'
    tri_obj_verts, obj_first_lines, obj_rest_lines = shape_target_sim.load_tri_starfish_obj(gt_obj_path_str)
    tri_obj_verts = np.array(tri_obj_verts).flatten()
    # _, q_ideal_np = shape_target_sim.get_idea_q_and_act(shape_target_sim.gt_folder, test_id) 
    # load the test data
    act_out = []
    with torch.no_grad():
        input_verts = torch.tensor(tri_obj_verts, dtype=torch.float32)
        input_verts = input_verts.unsqueeze(0)
        act_out_hat, mu, logvar = model(input_verts)
        act_out_hat = act_out_hat.squeeze().numpy()
        # add back the offset from the training data
        a_offset = [1,0,0,1,0,1] * (2175 * 8)
        a_offset = np.array(a_offset) - 0.5
        act_out = act_out_hat + a_offset
    print("act_out shape: ", act_out.shape)
    q_next_np = shape_target_sim.forward_pass(act_out, shape_target_sim.q_init_np)
    shape_target_sim.compute_loss(q_next_np, gt_verts_pos) # for trilinear mapping
    # render, breakdown shape_target_sim.construct_then_render_obj here due to different output path
    obj_output = render_folder / 'objs' / f'starfish_reconstructed_{test_id}.obj'
    shape_target_sim.write_tri_starfish_obj(obj_output, obj_first_lines, obj_rest_lines, shape_target_sim.final_interp_verts)
    png_output = render_folder / 'renders' / f'starfish_reconstructed_{test_id}.png'
    shape_target_sim.render_tri_starfish(obj_output, png_output)
    print("Test finished for: ", test_id, ', saved to: ', png_output)
print("Finished VAE Testing") 