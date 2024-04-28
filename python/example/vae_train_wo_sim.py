'''
A simple VAE for the starfish example
Input to encoder:
    trimesh size is 3246 = 1082 * 3
Latent:
    latent size is 16, 8 for mean and 8 for logvar
Output from decoder:
    A size is 104400, 2175 hex with 8 sample per hex and 6 vec per sample
Encoder first hidden layer is 256, second is 64
Decoder first hidden layer is 64, second is 512, third is 4096
Train it with batches of 16, with tqdm progress bar
'''


import sys
sys.path.append('/mnt/e/wsl_projects/diff_pd_public/python/')
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

asset_folder = Path('/mnt/e/muscleCode/sample_muscle_data/starfish')
default_hex_bin_str = str(asset_folder / 'starfish_demo_voxel.bin')
gt_folder = Path('/mnt/e/wsl_projects/diff_pd_public/python/example/quasi_starfish/stretched_gt_corotate_volume/')
density = 1e3
youngs_modulus = 5e5
poissons_ratio = 0.45
la = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
mu = youngs_modulus / (2 * (1 + poissons_ratio))


def data_loader():
    a_offset = [1,0,0,1,0,1] * (2175 * 8)
    a_offset = np.array(a_offset) - 0.5
    
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
    
    a_ground_truth = []
    q_ground_truth = []
    for i in range(1, 121):
        act, _ = get_idea_q_and_act(gt_folder, i)
        a_ground_truth.append(act - a_offset)
        min_a = min(a_ground_truth[-1])
        max_a = max(a_ground_truth[-1])
        if min_a < 0 or max_a > 1:
            print('Actuation out of bound: ', min_a, max_a)
        obj1_verts, first_lines, rest_lines = load_tri_starfish_obj(asset_folder, f"starfish_{i}.obj")
        obj1_verts_flat = np.array(obj1_verts).flatten()
        q_ground_truth.append(obj1_verts_flat)
    return a_ground_truth, q_ground_truth

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import numpy as np
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(3246, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 64),
            nn.LeakyReLU()
        )
        self.fc_mean = nn.Linear(64, 8)
        self.fc_logvar = nn.Linear(64, 8)

    def forward(self, x):
        h = self.fc_layers(x)
        return self.fc_mean(h), self.fc_logvar(h)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(8, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 104400),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.fc_layers(z)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = BCE + KLD
    loss /= x.size(0)  # Normalize by batch size
    return loss

if __name__ == '__main__':
    print("Running VAE Training")
    a, q = data_loader()
    a, q = np.array(a), np.array(q)
    print('a shape: ', a[0].shape)
    print('q shape: ', q[0].shape)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Cuda available: ", torch.cuda.is_available())
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.98, 0.999))

    # Assuming a and q are numpy arrays and have the same size
    a_tensor = torch.tensor(a, dtype=torch.float32).to(device)
    q_tensor = torch.tensor(q, dtype=torch.float32).to(device)
    dataset = TensorDataset(q_tensor, a_tensor)
    # Splitting dataset into training and testing
    train_indices = list(range(0, 41)) + list(range(61, 120))
    test_indices = list(range(41, 61))
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    # train_size = 100
    # test_size = len(dataset) - train_size
    # train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True)

    model.train()
    for epoch in tqdm(range(350)):  # Assuming 10 epochs for simplicity
        epoch_loss = 0.0
        for i, (q_data, a_data) in enumerate(train_dataloader):
            q_data = q_data.to(device)
            a_data = a_data.to(device)  # a_data is the target for training
            optimizer.zero_grad()
            a_hat, mu, logvar = model(q_data)  # q_data is the input to the encoder
            loss = loss_function(a_hat, a_data, mu, logvar)  # Compute loss between a_data and its reconstruction a_hat
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}, Training Loss: {epoch_loss}")
        
    # save model
    def save_model(model, path):
        torch.save(model.state_dict(), path)
        print(f"Model saved to {path}")
    save_path = '/mnt/e/wsl_projects/diff_pd_public/python/example/vae_starfish_output/vae.pth'
    save_model(model, save_path)


