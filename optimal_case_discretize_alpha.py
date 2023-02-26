import numpy as np
import math
import multiprocessing
import time
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from robot_models.SingleIntegrator2D import *
from Trajectory_Model import *
from matplotlib.animation import FFMpegWriter
from scipy.stats import norm
from discretize_alpha_helper import *

plt.rcParams.update({'font.size': 15}) #27
# Sim Parameters                  
dt = 0.1
t = 0

# Define Parameters for CLF and CBF
betta = 0.8
d_max = 0.2

# for curved trajectory
tf = 15
num_steps = int(tf/dt)
U_max = 2.0
y_max = 6.0



# Define Trajectory
radius = 5
num_points = 11
trajectory_points = PointsInCircum(radius,(num_points-1)*2)[0:num_points]
trajectory_time = math.pi*radius/(U_max/math.sqrt(2))
trajectory = Trajectory2D(trajectory_points=trajectory_points,tot_time=trajectory_time,poly_degree=5)

# Plot                  
plt.ion()
fig = plt.figure()
ax = plt.axes(xlim=(-6,6),ylim=(-2,8)) 
ax.set_xlabel("X")
ax.set_ylabel("Y")
rect = patches.Rectangle((-5, y_max), 10, 4, linewidth=1, edgecolor='none', facecolor='k')
# Add the patch to the Axes
ax.add_patch(rect)
ax.plot(trajectory_points[:,0],trajectory_points[:,1],'r--')
max_allowed_trajectory = PointsInCircum(radius+d_max,20)[0:11]
min_allowed_trajectory = PointsInCircum(radius-d_max,20)[0:11]
ax.plot(max_allowed_trajectory[:,0],max_allowed_trajectory[:,1],'k')
ax.plot(min_allowed_trajectory[:,0],min_allowed_trajectory[:,1],'k')
fig.canvas.draw()
fig.canvas.flush_events()

# Define Trajectory
radius = 5
num_points = 11
trajectory_points = PointsInCircum(radius,(num_points-1)*2)[0:num_points]
trajectory_time = math.pi*radius/(U_max/math.sqrt(2))
trajectory = Trajectory2D(trajectory_points=trajectory_points,tot_time=trajectory_time,poly_degree=5)

#Define Search Map
control_hash_table = {}
x_min = -6
x_max = 6.0
y_min = -2.0
y_max = 6.0
step = 0.01
x_range = np.arange(start=x_min, stop=x_max, step=step)
x_fliped_range = np.flip(x_range)
y_range = np.arange(start=y_min, stop=y_max, step=step)
y_fliped_range = np.flip(y_range)
u_thresh = 1e-4
feasible_candidates = []
for x in x_fliped_range:
    for y in y_fliped_range:
        if y >= 0 and (x**2+y**2)>(radius-d_max)**2 and (x**2+y**2)<(radius+d_max)**2:
            x0 = np.array([x,y])
            feasible_candidates.append(x0)
# Define alpha_list
#"""
alpha_step = 0.5
alpha_list = np.arange(start=0,stop=50.0+alpha_step,step=alpha_step)
#"""
#alpha_list = np.array([0.8])

with multiprocessing.Pool() as pool:
    for (x0_key, forward_set) in pool.map(forward_cal,feasible_candidates):
        for forward_cell in forward_set:
            if y < 0 or (x**2+y**2)<(radius-d_max)**2 or (x**2+y**2)>(radius+d_max)**2 :
                continue
            else:
                backward_set = control_hash_table.get(forward_cell)
                none_list = np.array([backward_set == None]).reshape(-1,).tolist()
                if any(none_list):
                    backward_set = np.array([x0_key])
                else:
                    backward_set = np.append(backward_set,np.array([x0_key]),axis=0)
                control_hash_table.update({forward_cell,backward_set})


x_success_list = []
y_success_list = []
target_pos = np.array([[-5.0,0.0]]).reshape(-1,2)
for i in range(target_pos.shape[0]):
    pos = target_pos[i,:]
    target_pos_key = str(int((pos[0]-x_min)/step))+","+str(int((pos[1]-y_min)/step))
    if i == 0:
        success_list = np.array([target_pos_key])
    else:
        updated = False
        for pos_key in success_list:
            if pos_key == target_pos:
                updated = True
        if updated == False:
            success_list =  np.append(success_list,np.array([target_pos_key]),axis=0)

pos_in_success_table = {}
while success_list.size > 0:
    current = success_list[0]
    success_list = np.delete(success_list, obj=0, axis=0)
    print(success_list.size)
    x = ""
    for i in range(len(current)):
        a = current[i]
        if a!=',':
            x += a
        else:
            break
    y = current[i+1:]
    x = x_range[int(x)]
    y = y_range[int(y)]
    x_success_list.append(x)
    y_success_list.append(y)
    backward_set = control_hash_table.get(current)
    none_list = np.array([backward_set == None]).reshape(-1,).tolist()
    if any(none_list):
        continue
    filtered_backward_set = None
    for i in range(backward_set.size):
        has_been_pushed = pos_in_success_table.get(backward_set[i])
        if has_been_pushed == None:
            none_list = np.array([filtered_backward_set == None]).reshape(-1,).tolist()
            if any(none_list):
                filtered_backward_set = np.array([backward_set[i]])
            else:
                filtered_backward_set = np.append(filtered_backward_set,np.array([backward_set[i]]),axis=0)                
        pos_in_success_table.update({backward_set[i]: True})

    none_list = np.array([filtered_backward_set == None]).reshape(-1,).tolist()
    if any(none_list):
        continue
    
    if len(success_list)> 0:
        success_list = np.append(success_list,filtered_backward_set,axis=0)
    else:
        success_list = filtered_backward_set
plt.ioff()

print(len(control_hash_table))
plt.plot(x_success_list,y_success_list,'b.')
print(len(x_success_list))
plt.show()
