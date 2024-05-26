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
domain_size = 200  # Size of the square domain
cell_size = 0.05    # Cell size and particle radius
particle_size = 0.10  # Cell size and particle radius

simulation_time = 1
kn = 600000#0#000  # Normal stiffness of the spring
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
input_shape_local  = (1, 1, domain_size, domain_size, domain_size)
# Generate particles x, y, z, mask
x_grid = np.zeros(input_shape_local, dtype=np.float32)  
y_grid = np.zeros(input_shape_local, dtype=np.float32)  
z_grid = np.zeros(input_shape_local, dtype=np.float32)  

mask = np.zeros(input_shape_local, dtype=np.float16)  


print('Number of particles:', np.count_nonzero(mask))

mask = torch.from_numpy(mask).float().to(device)
x_grid = torch.from_numpy(x_grid).float().to(device)
y_grid = torch.from_numpy(y_grid).float().to(device)
z_grid = torch.from_numpy(z_grid).float().to(device)

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
        for i in range(filter_size):
            for j in range(filter_size):
                for k in range(filter_size):
                        diffx = x_grid - self.detector(x_grid, i, j, k)    
                        diffy = y_grid - self.detector(y_grid, i, j, k)    
                        diffz = z_grid - self.detector(z_grid, i, j, k)    

                        dist = torch.sqrt(diffx**2 + diffy**2 + diffz**2)  

                        diffv_Vn =  (vx_grid - self.detector(vx_grid, i, j, k)) * diffx / torch.clamp(dist,1e-04) + \
                                    (vy_grid - self.detector(vy_grid, i, j, k)) * diffy / torch.clamp(dist,1e-04) + \
                                    (vz_grid - self.detector(vz_grid, i, j, k)) * diffz / torch.clamp(dist,1e-04)


                        # calculate collision force between the two particles
                        fx_grid_collision =  fx_grid_collision + torch.where(torch.lt(dist, particle_size), kn * (dist - particle_size ) * diffx / torch.clamp(dist,1e-04), zeros) + \
                                                                                           torch.where(torch.lt(dist, particle_size), kn * (dist - particle_size ) * diffx / torch.clamp(dist,1e-04), zeros) + \
                                                                                           torch.where(torch.lt(dist, particle_size), damping_coefficient_Eta * diffv_Vn * diffx / torch.clamp(dist,1e-04), zeros) + \
                                                                                           torch.where(torch.lt(dist, particle_size), damping_coefficient_Eta * diffv_Vn * diffx / torch.clamp(dist,1e-04), zeros)

                        fy_grid_collision =  fy_grid_collision + torch.where(torch.lt(dist, particle_size), kn * (dist - particle_size ) * diffy / torch.clamp(dist,1e-04), zeros) + \
                                                                                           torch.where(torch.lt(dist, particle_size), kn * (dist - particle_size ) * diffy / torch.clamp(dist,1e-04), zeros) + \
                                                                                           torch.where(torch.lt(dist, particle_size), damping_coefficient_Eta * diffv_Vn * diffy / torch.clamp(dist,1e-04), zeros) + \
                                                                                           torch.where(torch.lt(dist, particle_size), damping_coefficient_Eta * diffv_Vn * diffy / torch.clamp(dist,1e-04), zeros)                                                             

                        fz_grid_collision =  fz_grid_collision + torch.where(torch.lt(dist, particle_size), kn * (dist - particle_size ) * diffz / torch.clamp(dist,1e-04), zeros) + \
                                                                                           torch.where(torch.lt(dist, particle_size), kn * (dist - particle_size ) * diffz / torch.clamp(dist,1e-04), zeros) + \
                                                                                           torch.where(torch.lt(dist, particle_size), damping_coefficient_Eta * diffv_Vn * diffz / torch.clamp(dist,1e-04), zeros) + \
                                                                                           torch.where(torch.lt(dist, particle_size), damping_coefficient_Eta * diffv_Vn * diffz / torch.clamp(dist,1e-04), zeros) 
            
            
            is_left_overlap     = torch.gt(x_grid, particle_size) & torch.lt(x_grid, 1.5*particle_size) # Overlap with bottom wall
            is_right_overlap    = torch.gt(x_grid, domain_size*cell_size - 0.5*particle_size - cell_size)                          # Overlap with bottom wall
            is_bottom_overlap   = torch.gt(y_grid, particle_size) & torch.lt(y_grid, 1.5*particle_size) # Overlap with bottom wall
            is_top_overlap      = torch.gt(y_grid, domain_size*cell_size - 0.5*particle_size - cell_size)                         # Overlap with bottom wall
            is_forward_overlap  = torch.gt(z_grid, particle_size) & torch.lt(z_grid, 1.5*particle_size) # Overlap with bottom wall
            is_backward_overlap = torch.gt(z_grid, domain_size*cell_size - 0.5*particle_size - cell_size)                         # Overlap with bottom wall             

        # calculate the elastic force from the boundaries (512 x 512 x 512)
            fx_grid_boundary = kn * torch.where(is_left_overlap,  torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)) * mask * (particle_size*1.5 - x_grid) - \
                               kn * torch.where(is_right_overlap, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)) * mask * (x_grid - domain_size*cell_size + cell_size + particle_size*0.5) - \
                               damping_coefficient_Eta * vx_grid * torch.where(is_left_overlap,  torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)) * mask - \
                               damping_coefficient_Eta * vx_grid * torch.where(is_right_overlap, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)) * mask                                                        

            fy_grid_boundary = kn * torch.where(is_bottom_overlap, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)) * mask * (particle_size*1.5 - y_grid) - \
                               kn * torch.where(is_top_overlap,    torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)) * mask * (y_grid - domain_size*cell_size + cell_size + particle_size*0.5) - \
                               damping_coefficient_Eta * vy_grid * torch.where(is_bottom_overlap, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)) * mask - \
                               damping_coefficient_Eta * vy_grid * torch.where(is_top_overlap,    torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)) * mask

            fz_grid_boundary = kn * torch.where(is_forward_overlap, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)) * mask * (particle_size*1.5 - z_grid) - \
                               kn * torch.where(is_backward_overlap,torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)) * mask * (z_grid - domain_size*cell_size + cell_size + particle_size*0.5) - \
                               damping_coefficient_Eta * vz_grid * torch.where(is_forward_overlap, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)) * mask - \
                               damping_coefficient_Eta * vz_grid * torch.where(is_backward_overlap,torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)) * mask
                                      
        # calculate the new velocity with acceleration calculated by forces
            vx_grid = vx_grid + (dt / particle_mass) * mask * ( - 0.0 * particle_mass - fx_grid_collision + fx_grid_boundary) 
            vy_grid = vy_grid + (dt / particle_mass) * mask * ( - 0.0 * particle_mass - fy_grid_collision + fy_grid_boundary)  
            vz_grid = vz_grid + (dt / particle_mass) * mask * ( - 9.8 * particle_mass - fz_grid_collision + fz_grid_boundary) 
        
            del fx_grid_boundary, fy_grid_boundary, fz_grid_boundary
            del is_left_overlap, is_right_overlap, is_bottom_overlap, is_top_overlap, is_forward_overlap, is_backward_overlap

            cell_xold = torch.round(x_grid / cell_size).long()
            cell_yold = torch.round(y_grid / cell_size).long()
            cell_zold = torch.round(z_grid / cell_size).long()

            cell_xold = cell_xold[cell_xold!=0]
            cell_yold = cell_yold[cell_yold!=0]
            cell_zold = cell_zold[cell_zold!=0]


            # Update particle coordniates
            x_grid = x_grid + dt * vx_grid
            y_grid = y_grid + dt * vy_grid
            z_grid = z_grid + dt * vz_grid
           
            cell_x = torch.round(x_grid / cell_size).long()
            cell_y = torch.round(y_grid / cell_size).long()
            cell_z = torch.round(z_grid / cell_size).long()
            
            cell_x = cell_x[cell_x!=0]
            cell_y = cell_y[cell_y!=0]
            cell_z = cell_z[cell_z!=0]            

            x_grid_merge = x_grid.clone()
            y_grid_merge = y_grid.clone()
            z_grid_merge = z_grid.clone()
            vx_grid_merge = vx_grid.clone()
            vy_grid_merge = vy_grid.clone()
            vz_grid_merge = vz_grid.clone()

            x_grid[0,0,cell_zold.long(),cell_yold.long(),cell_xold.long()] = 0 
            y_grid[0,0,cell_zold.long(),cell_yold.long(),cell_xold.long()] = 0 
            z_grid[0,0,cell_zold.long(),cell_yold.long(),cell_xold.long()] = 0 
            vx_grid[0,0,cell_zold.long(),cell_yold.long(),cell_xold.long()] = 0 
            vy_grid[0,0,cell_zold.long(),cell_yold.long(),cell_xold.long()] = 0 
            vz_grid[0,0,cell_zold.long(),cell_yold.long(),cell_xold.long()] = 0 
            mask[0,0,cell_zold.long(), cell_yold.long(), cell_xold.long()] = 0
            
            
            x_grid[0,0,cell_z.long(),cell_y.long(), cell_x.long()] = x_grid_merge[0,0,cell_zold.long(),cell_yold.long(),cell_xold.long()] 
            y_grid[0,0,cell_z.long(),cell_y.long(), cell_x.long()] = y_grid_merge[0,0,cell_zold.long(),cell_yold.long(),cell_xold.long()] 
            z_grid[0,0,cell_z.long(),cell_y.long(), cell_x.long()] = z_grid_merge[0,0,cell_zold.long(),cell_yold.long(),cell_xold.long()] 
            vx_grid[0,0,cell_z.long(),cell_y.long(), cell_x.long()] = vx_grid_merge[0,0,cell_zold.long(),cell_yold.long(),cell_xold.long()] 
            vy_grid[0,0,cell_z.long(),cell_y.long(), cell_x.long()] = vy_grid_merge[0,0,cell_zold.long(),cell_yold.long(),cell_xold.long()]
            vz_grid[0,0,cell_z.long(),cell_y.long(), cell_x.long()] = vz_grid_merge[0,0,cell_zold.long(),cell_yold.long(),cell_xold.long()]
            mask[0,0,cell_z.long(),cell_y.long(), cell_x.long()] = 1

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
            save_path = 'DEM_no_friction_1p'

            if itime % 60 == 0:
                np.save(save_path+"/xp"+str(itime), arr=x_grid.cpu().detach().numpy()[0,0,:,:])            
                np.save(save_path+"/yp"+str(itime), arr=y_grid.cpu().detach().numpy()[0,0,:,:])            
                np.save(save_path+"/zp"+str(itime), arr=z_grid.cpu().detach().numpy()[0,0,:,:])            
                np.save(save_path+"/up"+str(itime), arr=vx_grid.cpu().detach().numpy()[0,0,:,:])            
                np.save(save_path+"/vp"+str(itime), arr=vy_grid.cpu().detach().numpy()[0,0,:,:])            
                np.save(save_path+"/wp"+str(itime), arr=vz_grid.cpu().detach().numpy()[0,0,:,:])     
                np.save(save_path+"/Fx"+str(itime), arr=Fx.cpu().detach().numpy()[0,0,:,:])            
                np.save(save_path+"/Fy"+str(itime), arr=Fy.cpu().detach().numpy()[0,0,:,:])            
                np.save(save_path+"/Fz"+str(itime), arr=Fz.cpu().detach().numpy()[0,0,:,:])     

            if itime==1:
                for i in range(3, 97):
                    for j in range(3, 97):
                        x_grid[0, 0, domain_size-4, j*2, i*2] = i*cell_size*2
                        y_grid[0, 0, domain_size-4, j*2, i*2] = j*cell_size*2
                        z_grid[0, 0, domain_size-4, j*2, i*2] = (domain_size-4)* cell_size
                        vz_grid[0, 0, domain_size-4, j*2, i*2] = -5*0.001
                        vx_grid[0, 0, domain_size-4, j*2, i*2] = 0# random.uniform(-1.0,1.0)*0.001
                        vy_grid[0, 0, domain_size-4, j*2, i*2] = 0# random.uniform(-1.0,1.0)*0.001
                        mask[0, 0, domain_size-4, j*2, i*2] = 1
            if itime % 250 == 0 and itime < 5000 :
                for i in range(3, 97):
                    for j in range(3, 97):
                        x_grid[0, 0, domain_size-4, j*2, i*2] = i*cell_size*2
                        y_grid[0, 0, domain_size-4, j*2, i*2] = j*cell_size*2
                        z_grid[0, 0, domain_size-4, j*2, i*2] = (domain_size-4)* cell_size
                        vz_grid[0, 0, domain_size-4, j*2, i*2] = -5*0.001
                        vx_grid[0, 0, domain_size-4, j*2, i*2] = 0# random.uniform(-1.0,1.0)*0.001
                        vy_grid[0, 0, domain_size-4, j*2, i*2] = 0# random.uniform(-1.0,1.0)*0.001
                        mask[0, 0, domain_size-4, j*2, i*2] = 1
            print('Time step:', itime, 'Number of particles:', torch.count_nonzero(mask).item()) 
    end = time.time()
    print('Elapsed time:', end - start)