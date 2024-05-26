import os
import numpy as np 
import pandas as pd
import time 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Check if GPU is available 
is_gpu = torch.cuda.is_available()
device = torch.device("cuda" if is_gpu else "cpu")
#device = torch.device("cpu")
print("Using GPU:", is_gpu)

# Make sure empty memory before running the code 
torch.cuda.empty_cache()

# Input Parameters
domain_size = 100  # Size of the square domain
cell_size = 0.10   # Cell size and particle radius
particle_size = 0.10   # Cell size and particle radius
particle_cell_ratio = particle_size/cell_size


simulation_time = 1
kn = 6000000#0#000  # Normal stiffness of the spring
restitution_coefficient = 0.5  # Normal damping coefficient
friction_coefficient = 0.5  # coefficient of friction
rho_p = 2700 
particle_mass = 4/3*3.1415*cell_size**3*rho_p #4188.7902

K_graph = 2.2*10000000*1.7
S_graph = K_graph * (cell_size / domain_size) ** 2


damping_coefficient_Alpha = -1*math.log(restitution_coefficient)/math.pi
damping_coefficient_Gamma = damping_coefficient_Alpha/math.sqrt(damping_coefficient_Alpha**2+1)
damping_coefficient_Eta   = 2 * damping_coefficient_Gamma * math.sqrt(kn * particle_mass)

print('Damping Coefficient:', damping_coefficient_Eta)

input_shape_global = (1, 1, domain_size, domain_size, domain_size)
input_shape_local  = (2, 1, 1, domain_size, domain_size, domain_size)
# Generate particles x, y, z, mask
x_grid = np.zeros(input_shape_local, dtype=np.float32)  
y_grid = np.zeros(input_shape_local, dtype=np.float32)  
z_grid = np.zeros(input_shape_local, dtype=np.float32)  

x_grid_next = np.zeros(input_shape_local, dtype=np.float32)  
y_grid_next = np.zeros(input_shape_local, dtype=np.float32)  
z_grid_next = np.zeros(input_shape_local, dtype=np.float32) 

mask = np.zeros(input_shape_local, dtype=np.float16)  


print('Number of particles:', np.count_nonzero(mask))

mask = torch.from_numpy(mask).float().to(device)
x_grid = torch.from_numpy(x_grid).float().to(device)
y_grid = torch.from_numpy(y_grid).float().to(device)
z_grid = torch.from_numpy(z_grid).float().to(device)



x_grid_next = torch.from_numpy(x_grid_next).float().to(device)
y_grid_next = torch.from_numpy(y_grid_next).float().to(device)
z_grid_next = torch.from_numpy(z_grid_next).float().to(device)

vx_grid = torch.zeros(input_shape_local, device=device, dtype=torch.float32)  
vy_grid = torch.zeros(input_shape_local, device=device, dtype=torch.float32) 
vz_grid = torch.zeros(input_shape_local, device=device, dtype=torch.float32) 


# Define the AI4DEM model
class AI4DEM(nn.Module):
    """AI4DEM model for particle interaction detection and force calculation"""
    def __init__(self):
        super(AI4DEM, self).__init__()
    def detector(self, grid, i, j, k):
        """Detect neighboring particles and calculate the distance between them"""
        return torch.roll(grid, shifts=(k - 2, j - 2, i - 2), dims=(2, 3, 4))
    def forward(self, x_grid, y_grid, z_grid, vx_grid, vy_grid, vz_grid, mask, cell_size, particle_size, kn, damping_coefficient_Eta, dt, input_shape, filter_size):
        fx_grid_collision = torch.zeros(input_shape, device=device, dtype=torch.float32) 
        fy_grid_collision = torch.zeros(input_shape, device=device, dtype=torch.float32) 
        fz_grid_collision = torch.zeros(input_shape, device=device, dtype=torch.float32) 
                
        # judge whether the particle is colliding the boundaries
        for n in range(2):          
            for i in range(filter_size):
                for j in range(filter_size):
                    for k in range(filter_size):
                            diffx1 = x_grid[n,:,:,:,:,:] - self.detector(x_grid[0,:,:,:,:,:], i, j, k)    
                            diffx2 = x_grid[n,:,:,:,:,:] - self.detector(x_grid[1,:,:,:,:,:], i, j, k)
                            diffy1 = y_grid[n,:,:,:,:,:] - self.detector(y_grid[0,:,:,:,:,:], i, j, k)    
                            diffy2 = y_grid[n,:,:,:,:,:] - self.detector(y_grid[1,:,:,:,:,:], i, j, k)
                            diffz1 = z_grid[n,:,:,:,:,:] - self.detector(z_grid[0,:,:,:,:,:], i, j, k)    
                            diffz2 = z_grid[n,:,:,:,:,:] - self.detector(z_grid[1,:,:,:,:,:], i, j, k)

                            dist1 = torch.sqrt(diffx1**2 + diffy1**2 + diffz1**2)  
                            dist2 = torch.sqrt(diffx2**2 + diffy2**2 + diffz2**2)  

                            diffv_Vn1 = (vx_grid[n,:,:,:,:,:] - self.detector(vx_grid[0,:,:,:,:,:], i, j, k)) * diffx1 / torch.clamp(dist1,1e-04) + \
                                        (vy_grid[n,:,:,:,:,:] - self.detector(vy_grid[0,:,:,:,:,:], i, j, k)) * diffy1 / torch.clamp(dist1,1e-04) + \
                                        (vz_grid[n,:,:,:,:,:] - self.detector(vz_grid[0,:,:,:,:,:], i, j, k)) * diffz1 / torch.clamp(dist1,1e-04)

                            diffv_Vn2 = (vx_grid[n,:,:,:,:,:] - self.detector(vx_grid[1,:,:,:,:,:], i, j, k)) * diffx2 / torch.clamp(dist2,1e-04) + \
                                        (vy_grid[n,:,:,:,:,:] - self.detector(vy_grid[1,:,:,:,:,:], i, j, k)) * diffy2 / torch.clamp(dist2,1e-04) + \
                                        (vz_grid[n,:,:,:,:,:] - self.detector(vz_grid[1,:,:,:,:,:], i, j, k)) * diffz2 / torch.clamp(dist2,1e-04)

                            # calculate collision force between the two particles
                            fx_grid_collision[n,:,:,:,:,:] =  fx_grid_collision[n,:,:,:,:,:] + torch.where(torch.lt(dist1, particle_size), kn * (dist1 - particle_size ) * diffx1 / torch.clamp(dist1,1e-04), zeros) + \
                                                                                               torch.where(torch.lt(dist2, particle_size), kn * (dist2 - particle_size ) * diffx2 / torch.clamp(dist2,1e-04), zeros) + \
                                                                                               torch.where(torch.lt(dist1, particle_size), damping_coefficient_Eta * diffv_Vn1 * diffx1 / torch.clamp(dist1,1e-04), zeros) + \
                                                                                               torch.where(torch.lt(dist2, particle_size), damping_coefficient_Eta * diffv_Vn2 * diffx2 / torch.clamp(dist2,1e-04), zeros)

                            fy_grid_collision[n,:,:,:,:,:] =  fy_grid_collision[n,:,:,:,:,:] + torch.where(torch.lt(dist1, particle_size), kn * (dist1 - particle_size ) * diffy1 / torch.clamp(dist1,1e-04), zeros) + \
                                                                                               torch.where(torch.lt(dist2, particle_size), kn * (dist2 - particle_size ) * diffy2 / torch.clamp(dist2,1e-04), zeros) + \
                                                                                               torch.where(torch.lt(dist1, particle_size), damping_coefficient_Eta * diffv_Vn1 * diffy1 / torch.clamp(dist1,1e-04), zeros) + \
                                                                                               torch.where(torch.lt(dist2, particle_size), damping_coefficient_Eta * diffv_Vn2 * diffy2 / torch.clamp(dist2,1e-04), zeros)                                                             

                            fz_grid_collision[n,:,:,:,:,:] =  fz_grid_collision[n,:,:,:,:,:] + torch.where(torch.lt(dist1, particle_size), kn * (dist1 - particle_size ) * diffz1 / torch.clamp(dist1,1e-04), zeros) + \
                                                                                               torch.where(torch.lt(dist2, particle_size), kn * (dist2 - particle_size ) * diffz2 / torch.clamp(dist2,1e-04), zeros) + \
                                                                                               torch.where(torch.lt(dist1, particle_size), damping_coefficient_Eta * diffv_Vn1 * diffz1 / torch.clamp(dist1,1e-04), zeros) + \
                                                                                               torch.where(torch.lt(dist2, particle_size), damping_coefficient_Eta * diffv_Vn2 * diffz2 / torch.clamp(dist2,1e-04), zeros) 
            
            
            is_left_overlap     = torch.gt(x_grid[n,:,:,:,:,:], particle_size) & torch.lt(x_grid[n,:,:,:,:,:], 1.5*particle_size) # Overlap with bottom wall
            is_right_overlap    = torch.gt(x_grid[n,:,:,:,:,:], domain_size*cell_size - 0.5*particle_size - cell_size)                          # Overlap with bottom wall
            is_bottom_overlap   = torch.gt(y_grid[n,:,:,:,:,:], particle_size) & torch.lt(y_grid[n,:,:,:,:,:], 1.5*particle_size) # Overlap with bottom wall
            is_top_overlap      = torch.gt(y_grid[n,:,:,:,:,:], domain_size*cell_size - 0.5*particle_size - cell_size)                         # Overlap with bottom wall
            is_forward_overlap  = torch.gt(z_grid[n,:,:,:,:,:], particle_size) & torch.lt(z_grid[n,:,:,:,:,:], 1.5*particle_size) # Overlap with bottom wall
            is_backward_overlap = torch.gt(z_grid[n,:,:,:,:,:], domain_size*cell_size - 0.5*particle_size - cell_size)                         # Overlap with bottom wall             

        # calculate the elastic force from the boundaries (512 x 512 x 512)
            fx_grid_boundary = kn * torch.where(is_left_overlap,  torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)) * mask[n,:,:,:,:,:] * (particle_size*1.5 - x_grid[n,:,:,:,:,:]) - \
                               kn * torch.where(is_right_overlap, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)) * mask[n,:,:,:,:,:] * (x_grid[n,:,:,:,:,:] - domain_size*cell_size + cell_size + particle_size*0.5) - \
                               damping_coefficient_Eta * vx_grid[n,:,:,:,:,:] * torch.where(is_left_overlap,  torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)) * mask[n,:,:,:,:,:] - \
                               damping_coefficient_Eta * vx_grid[n,:,:,:,:,:] * torch.where(is_right_overlap, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)) * mask[n,:,:,:,:,:]                                                        

            fy_grid_boundary = kn * torch.where(is_bottom_overlap, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)) * mask[n,:,:,:,:,:] * (particle_size*1.5 - y_grid[n,:,:,:,:,:]) - \
                               kn * torch.where(is_top_overlap,    torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)) * mask[n,:,:,:,:,:] * (y_grid[n,:,:,:,:,:] - domain_size*cell_size + cell_size + particle_size*0.5) - \
                               damping_coefficient_Eta * vy_grid[n,:,:,:,:,:] * torch.where(is_bottom_overlap, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)) * mask[n,:,:,:,:,:] - \
                               damping_coefficient_Eta * vy_grid[n,:,:,:,:,:] * torch.where(is_top_overlap,    torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)) * mask[n,:,:,:,:,:]

            fz_grid_boundary = kn * torch.where(is_forward_overlap, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)) * mask[n,:,:,:,:,:] * (particle_size*1.5 - z_grid[n,:,:,:,:,:]) - \
                               kn * torch.where(is_backward_overlap,torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)) * mask[n,:,:,:,:,:] * (z_grid[n,:,:,:,:,:] - domain_size*cell_size + cell_size + particle_size*0.5) - \
                               damping_coefficient_Eta * vz_grid[n,:,:,:,:,:] * torch.where(is_forward_overlap, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)) * mask[n,:,:,:,:,:] - \
                               damping_coefficient_Eta * vz_grid[n,:,:,:,:,:] * torch.where(is_backward_overlap,torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)) * mask[n,:,:,:,:,:]
                                      
        # calculate the new velocity with acceleration calculated by forces
            vx_grid[n,:,:,:,:,:] = vx_grid[n,:,:,:,:,:] + (dt / particle_mass) * mask[n,:,:,:,:,:] * ( - 0.0 * particle_mass - fx_grid_collision[n,:,:,:,:,:] + fx_grid_boundary) 
            vy_grid[n,:,:,:,:,:] = vy_grid[n,:,:,:,:,:] + (dt / particle_mass) * mask[n,:,:,:,:,:] * ( - 0.0 * particle_mass - fy_grid_collision[n,:,:,:,:,:] + fy_grid_boundary)  
            vz_grid[n,:,:,:,:,:] = vz_grid[n,:,:,:,:,:] + (dt / particle_mass) * mask[n,:,:,:,:,:] * ( - 9.8 * particle_mass - fz_grid_collision[n,:,:,:,:,:] + fz_grid_boundary) 
        
            del fx_grid_boundary, fy_grid_boundary, fz_grid_boundary
            del is_left_overlap, is_right_overlap, is_bottom_overlap, is_top_overlap, is_forward_overlap, is_backward_overlap

            cell_xold = torch.round(x_grid[n,:,:,:,:,:] / cell_size).long()
            cell_yold = torch.round(y_grid[n,:,:,:,:,:] / cell_size).long()
            cell_zold = torch.round(z_grid[n,:,:,:,:,:] / cell_size).long()

            cell_xold = cell_xold[cell_xold!=0]
            cell_yold = cell_yold[cell_yold!=0]
            cell_zold = cell_zold[cell_zold!=0]


            # Update particle coordniates
            x_grid[n,:,:,:,:,:] = x_grid[n,:,:,:,:,:] + dt * vx_grid[n,:,:,:,:,:]
            y_grid[n,:,:,:,:,:] = y_grid[n,:,:,:,:,:] + dt * vy_grid[n,:,:,:,:,:]
            z_grid[n,:,:,:,:,:] = z_grid[n,:,:,:,:,:] + dt * vz_grid[n,:,:,:,:,:]

            cell_x = torch.round(x_grid[n,:,:,:,:,:] / cell_size).long()
            cell_y = torch.round(y_grid[n,:,:,:,:,:] / cell_size).long()
            cell_z = torch.round(z_grid[n,:,:,:,:,:] / cell_size).long()
            
            cell_x = cell_x[cell_x!=0]
            cell_y = cell_y[cell_y!=0]
            cell_z = cell_z[cell_z!=0]            
           
            x_grid_merge = x_grid[n,:,:,:,:,:].clone()
            y_grid_merge = y_grid[n,:,:,:,:,:].clone()
            z_grid_merge = z_grid[n,:,:,:,:,:].clone()
            vx_grid_merge = vx_grid[n,:,:,:,:,:].clone()
            vy_grid_merge = vy_grid[n,:,:,:,:,:].clone()
            vz_grid_merge = vz_grid[n,:,:,:,:,:].clone()

            x_grid[n,0,0,cell_zold.long(),cell_yold.long(),cell_xold.long()] = 0 
            y_grid[n,0,0,cell_zold.long(),cell_yold.long(),cell_xold.long()] = 0 
            z_grid[n,0,0,cell_zold.long(),cell_yold.long(),cell_xold.long()] = 0 
            vx_grid[n,0,0,cell_zold.long(),cell_yold.long(),cell_xold.long()] = 0 
            vy_grid[n,0,0,cell_zold.long(),cell_yold.long(),cell_xold.long()] = 0 
            vz_grid[n,0,0,cell_zold.long(),cell_yold.long(),cell_xold.long()] = 0 
            mask[n,0,0,cell_zold.long(), cell_yold.long(), cell_xold.long()] = 0
            
            
            x_grid[n,0,0,cell_z.long(),cell_y.long(), cell_x.long()] = x_grid_merge[0,0,cell_zold.long(),cell_yold.long(),cell_xold.long()] 
            y_grid[n,0,0,cell_z.long(),cell_y.long(), cell_x.long()] = y_grid_merge[0,0,cell_zold.long(),cell_yold.long(),cell_xold.long()] 
            z_grid[n,0,0,cell_z.long(),cell_y.long(), cell_x.long()] = z_grid_merge[0,0,cell_zold.long(),cell_yold.long(),cell_xold.long()] 
            vx_grid[n,0,0,cell_z.long(),cell_y.long(), cell_x.long()] = vx_grid_merge[0,0,cell_zold.long(),cell_yold.long(),cell_xold.long()] 
            vy_grid[n,0,0,cell_z.long(),cell_y.long(), cell_x.long()] = vy_grid_merge[0,0,cell_zold.long(),cell_yold.long(),cell_xold.long()]
            vz_grid[n,0,0,cell_z.long(),cell_y.long(), cell_x.long()] = vz_grid_merge[0,0,cell_zold.long(),cell_yold.long(),cell_xold.long()]
            mask[n,0,0,cell_z.long(),cell_y.long(), cell_x.long()] = 1


            x_grid_next[n,:,:,:,:,:] = x_grid[n,:,:,:,:,:] + dt * vx_grid[n,:,:,:,:,:]
            y_grid_next[n,:,:,:,:,:] = y_grid[n,:,:,:,:,:] + dt * vy_grid[n,:,:,:,:,:]
            z_grid_next[n,:,:,:,:,:] = z_grid[n,:,:,:,:,:] + dt * vz_grid[n,:,:,:,:,:]
            
            cell_x_next = torch.round(x_grid_next[n,:,:,:,:,:] / cell_size).long()
            cell_y_next = torch.round(y_grid_next[n,:,:,:,:,:] / cell_size).long()
            cell_z_next = torch.round(z_grid_next[n,:,:,:,:,:] / cell_size).long()     
            
            cell_x_next = cell_x_next[cell_x_next!=0]
            cell_y_next = cell_y_next[cell_y_next!=0]
            cell_z_next = cell_z_next[cell_z_next!=0]
            
            combined_coords = torch.stack((cell_x_next, cell_y_next, cell_z_next), dim=1)
            combined_coords = combined_coords.tolist()
            
            # 将坐标转换为张量
            coords_tensor = torch.tensor(combined_coords)

            # 使用 torch.unique 函数找到唯一的坐标及其计数
            unique_coords, counts = torch.unique(coords_tensor, return_counts=True, dim=0)

            # 找到第二次重复出现的坐标及其位置
            duplicate_indices = []
            for coord in unique_coords[counts > 1]:
                positions = (coords_tensor == coord).all(dim=1).nonzero(as_tuple=True)[0].tolist()
                if len(positions) > 1:
                    duplicate_indices.append((coord.tolist(), positions[1]))
            
            if duplicate_indices !=[] and n == 0:  
            
                duplicate_coordinate = [item[0] for item in duplicate_indices]
                duplicate_indices = [item[1] for item in duplicate_indices]

                x_grid[1, 0, 0, cell_z[duplicate_indices].long(),cell_y[duplicate_indices].long(),cell_x[duplicate_indices].long()]  = x_grid[0, 0, 0, cell_z[duplicate_indices].long(),cell_y[duplicate_indices].long(),cell_x[duplicate_indices].long()] 
                y_grid[1, 0, 0, cell_z[duplicate_indices].long(),cell_y[duplicate_indices].long(),cell_x[duplicate_indices].long()]  = y_grid[0, 0, 0, cell_z[duplicate_indices].long(),cell_y[duplicate_indices].long(),cell_x[duplicate_indices].long()] 
                z_grid[1, 0, 0, cell_z[duplicate_indices].long(),cell_y[duplicate_indices].long(),cell_x[duplicate_indices].long()]  = z_grid[0, 0, 0, cell_z[duplicate_indices].long(),cell_y[duplicate_indices].long(),cell_x[duplicate_indices].long()] 
                vx_grid[1, 0, 0, cell_z[duplicate_indices].long(),cell_y[duplicate_indices].long(),cell_x[duplicate_indices].long()]  = vx_grid[0, 0, 0, cell_z[duplicate_indices].long(),cell_y[duplicate_indices].long(),cell_x[duplicate_indices].long()] 
                vy_grid[1, 0, 0, cell_z[duplicate_indices].long(),cell_y[duplicate_indices].long(),cell_x[duplicate_indices].long()]  = vy_grid[0, 0, 0, cell_z[duplicate_indices].long(),cell_y[duplicate_indices].long(),cell_x[duplicate_indices].long()] 
                vz_grid[1, 0, 0, cell_z[duplicate_indices].long(),cell_y[duplicate_indices].long(),cell_x[duplicate_indices].long()]  = vz_grid[0, 0, 0, cell_z[duplicate_indices].long(),cell_y[duplicate_indices].long(),cell_x[duplicate_indices].long()] 
                mask[1, 0, 0, cell_z[duplicate_indices].long(),cell_y[duplicate_indices].long(),cell_x[duplicate_indices].long()]  = 1
                x_grid[0, 0, 0, cell_z[duplicate_indices].long(),cell_y[duplicate_indices].long(),cell_x[duplicate_indices].long()]  = 0
                y_grid[0, 0, 0, cell_z[duplicate_indices].long(),cell_y[duplicate_indices].long(),cell_x[duplicate_indices].long()]  = 0
                z_grid[0, 0, 0, cell_z[duplicate_indices].long(),cell_y[duplicate_indices].long(),cell_x[duplicate_indices].long()]  = 0
                vx_grid[0, 0, 0, cell_z[duplicate_indices].long(),cell_y[duplicate_indices].long(),cell_x[duplicate_indices].long()]  = 0
                vy_grid[0, 0, 0, cell_z[duplicate_indices].long(),cell_y[duplicate_indices].long(),cell_x[duplicate_indices].long()]  = 0
                vz_grid[0, 0, 0, cell_z[duplicate_indices].long(),cell_y[duplicate_indices].long(),cell_x[duplicate_indices].long()]  = 0
                mask[0, 0, 0, cell_z[duplicate_indices].long(),cell_y[duplicate_indices].long(),cell_x[duplicate_indices].long()]  = 0
                
            if duplicate_indices !=[] and n == 1:  
                duplicate_coordinate = [item[0] for item in duplicate_indices]
                duplicate_indices = [item[1] for item in duplicate_indices]
                
                x_grid[0, 0, 0, cell_z[duplicate_indices].long(),cell_y[duplicate_indices],cell_x[duplicate_indices]]  = x_grid[1, 0, 0, cell_z[duplicate_indices].long(),cell_y[duplicate_indices].long(),cell_x[duplicate_indices].long()] 
                y_grid[0, 0, 0, cell_z[duplicate_indices].long(),cell_y[duplicate_indices],cell_x[duplicate_indices]]  = y_grid[1, 0, 0, cell_z[duplicate_indices].long(),cell_y[duplicate_indices].long(),cell_x[duplicate_indices].long()] 
                z_grid[0, 0, 0, cell_z[duplicate_indices].long(),cell_y[duplicate_indices],cell_x[duplicate_indices]]  = z_grid[1, 0, 0, cell_z[duplicate_indices].long(),cell_y[duplicate_indices].long(),cell_x[duplicate_indices].long()] 
                vx_grid[0, 0, 0, cell_z[duplicate_indices].long(),cell_y[duplicate_indices],cell_x[duplicate_indices]]  = vx_grid[1, 0, 0, cell_z[duplicate_indices].long(),cell_y[duplicate_indices].long(),cell_x[duplicate_indices].long()] 
                vy_grid[0, 0, 0, cell_z[duplicate_indices].long(),cell_y[duplicate_indices],cell_x[duplicate_indices]]  = vy_grid[1, 0, 0, cell_z[duplicate_indices].long(),cell_y[duplicate_indices].long(),cell_x[duplicate_indices].long()] 
                vz_grid[0, 0, 0, cell_z[duplicate_indices].long(),cell_y[duplicate_indices],cell_x[duplicate_indices]]  = vz_grid[1, 0, 0, cell_z[duplicate_indices].long(),cell_y[duplicate_indices].long(),cell_x[duplicate_indices].long()] 
                mask[0, 0, 0, cell_z[duplicate_indices].long(),cell_y[duplicate_indices],cell_x[duplicate_indices]]  = 1
                x_grid[1, 0, 0, cell_z[duplicate_indices].long(),cell_y[duplicate_indices].long(),cell_x[duplicate_indices].long()]  = 0
                y_grid[1, 0, 0, cell_z[duplicate_indices].long(),cell_y[duplicate_indices].long(),cell_x[duplicate_indices].long()]  = 0
                z_grid[1, 0, 0, cell_z[duplicate_indices].long(),cell_y[duplicate_indices].long(),cell_x[duplicate_indices].long()]  = 0
                vx_grid[1, 0, 0, cell_z[duplicate_indices].long(),cell_y[duplicate_indices].long(),cell_x[duplicate_indices].long()]  = 0
                vy_grid[1, 0, 0, cell_z[duplicate_indices].long(),cell_y[duplicate_indices].long(),cell_x[duplicate_indices].long()]  = 0
                vz_grid[1, 0, 0, cell_z[duplicate_indices].long(),cell_y[duplicate_indices].long(),cell_x[duplicate_indices].long()]  = 0
                mask[1, 0, 0, cell_z[duplicate_indices].long(),cell_y[duplicate_indices].long(),cell_x[duplicate_indices].long()]  = 0


        del x_grid_merge, y_grid_merge, z_grid_merge

        del vx_grid_merge, vy_grid_merge, vz_grid_merge

        return x_grid, y_grid, z_grid, vx_grid, vy_grid, vz_grid, mask, fx_grid_collision, fy_grid_collision, fz_grid_collision

model = AI4DEM().to(device)
# model = torch.compile(model, mode="reduce-overhead")
t = 0
dt = 0.0001  # 0.0001
ntime = 4000
filter_size = 5
zeros = torch.zeros(input_shape_global, device=device, dtype=torch.float16)

# Main simulation loop

with torch.no_grad():
    start = time.time()
    for itime in range(1, ntime + 1):
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            [x_grid, y_grid, z_grid, vx_grid, vy_grid, vz_grid, mask, Fx, Fy, Fz] = model(x_grid, y_grid, z_grid, vx_grid, vy_grid, vz_grid, mask, cell_size, particle_size, kn, damping_coefficient_Eta, dt, input_shape_local, filter_size)
            save_path = 'DEM_no_friction_2p'

            if itime % 600 == 0:
                np.save(save_path+"/xp"+str(itime), arr=x_grid.cpu().detach().numpy()[:,0,0,:,:])            
                np.save(save_path+"/yp"+str(itime), arr=y_grid.cpu().detach().numpy()[:,0,0,:,:])            
                np.save(save_path+"/zp"+str(itime), arr=z_grid.cpu().detach().numpy()[:,0,0,:,:])            
                np.save(save_path+"/up"+str(itime), arr=vx_grid.cpu().detach().numpy()[:,0,0,:,:])            
                np.save(save_path+"/vp"+str(itime), arr=vy_grid.cpu().detach().numpy()[:,0,0,:,:])            
                np.save(save_path+"/wp"+str(itime), arr=vz_grid.cpu().detach().numpy()[:,0,0,:,:])     
                np.save(save_path+"/Fx"+str(itime), arr=Fx.cpu().detach().numpy()[:,0,0,:,:])            
                np.save(save_path+"/Fy"+str(itime), arr=Fy.cpu().detach().numpy()[:,0,0,:,:])            
                np.save(save_path+"/Fz"+str(itime), arr=Fz.cpu().detach().numpy()[:,0,0,:,:])     

            if itime==1:
                for i in range(3, 77):
                    for j in range(3, 77):
                        x_grid [0,0, 0, domain_size-4, round (j*particle_cell_ratio), round (i*particle_cell_ratio)] = i*particle_size
                        y_grid [0,0, 0, domain_size-4, round (j*particle_cell_ratio), round (i*particle_cell_ratio)] = i*particle_size
                        z_grid [0,0, 0, domain_size-4, round (j*particle_cell_ratio), round (i*particle_cell_ratio)] = i*particle_size
                        vx_grid [0,0, 0, domain_size-4, round (j*particle_cell_ratio), round (i*particle_cell_ratio)] = i*particle_size
                        vy_grid [0,0, 0, domain_size-4, round (j*particle_cell_ratio), round (i*particle_cell_ratio)] = i*particle_size
                        vz_grid [0,0, 0, domain_size-4, round (j*particle_cell_ratio), round (i*particle_cell_ratio)] = i*particle_size
                        mask[0,0, 0, domain_size-4, round (j*particle_size/cell_size), round (i*particle_size/cell_size)] = 1

            if itime % 250 == 0 and itime < 5000 :
                for i in range(3, 77):
                    for j in range(3, 77):
                        x_grid[0,0, 0, domain_size-4, round (j*particle_size/cell_size), round (i*particle_size/cell_size)] = i*particle_size
                        y_grid[0,0, 0, domain_size-4, round (j*particle_size/cell_size), round (i*particle_size/cell_size)] = j*particle_size
                        z_grid[0,0, 0, domain_size-4, round (j*particle_size/cell_size), round (i*particle_size/cell_size)] = (domain_size-4)*cell_size
                        vx_grid[0,0, 0, domain_size-4, round (j*particle_size/cell_size), round (i*particle_size/cell_size)] = 0
                        vy_grid[0,0, 0, domain_size-4, round (j*particle_size/cell_size), round (i*particle_size/cell_size)] = 0
                        vz_grid[0,0, 0, domain_size-4, round (j*particle_size/cell_size), round (i*particle_size/cell_size)] = -0.05
                        mask[0,0, 0, domain_size-4, round (j*particle_size/cell_size), round (i*particle_size/cell_size)] = 1
               

            print('Time step:', itime, 'Number of particles:', torch.count_nonzero(mask).item()) 
    end = time.time()
    print('Elapsed time:', end - start)