import numpy as np
import math
import time
import cvxpy as cp
import multiprocessing
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import norm
from robot_models.SingleIntegrator2D import *
from Safe_Set_Series import *
from matplotlib.animation import FFMpegWriter
from Trajectory_Model import *
from discretize_helper import *

plt.rcParams.update({'font.size': 15}) #27
# Sim Parameters                  
dt = 0.05
t = 0
tf = 20
num_steps = int(tf/dt)

# Define Parameters for CLF and CBF
U_max = 2.0
d_max = 0.5


# Plot                  
plt.ion()
x_min = -6
x_max = 6
y_min = -2
y_max = 6
fig = plt.figure()
ax = plt.axes(xlim=(x_min,x_max),ylim=(y_min,y_max+2)) 
ax.set_xlabel("X")
ax.set_ylabel("Y")
# Define Series of Safe Sets
num_points = 21
centroids = PointsInCircum(r=5,n=(num_points-1)*2)[1:num_points]
rect = patches.Rectangle((-5, y_max), 10, 4, linewidth=1, edgecolor='none', facecolor='k')
# Add the patch to the Axes
ax.add_patch(rect)
radii = np.zeros((centroids.shape[0],))+d_max
alpha_list = np.zeros((centroids.shape[0],))+1.0
Safe_Set_Series = Safe_Set_Series2D(centroids=centroids,radii=radii,alpha_list=alpha_list)

for i in range(0,num_points-1):
    circle = patches.Circle(centroids[i,:], radius=radii[i], color='green')
    ax.add_patch(circle)
ax.axis('equal')

metadata = dict(title='Movie Test', artist='Matplotlib',comment='Movie support!')
writer = FFMpegWriter(fps=15, metadata=metadata)
movie_name = 'series_of_safesets_with_disturb_6_0_alpha_downwind.mp4'


#Define Search Map
control_hash_table = {}
step = 0.05
x_range = np.arange(start=x_min, stop=x_max, step=step)
x_fliped_range = np.flip(x_range)
y_range = np.arange(start=y_min, stop=y_max, step=step)
y_fliped_range = np.flip(y_range)
feasible_candidates = []

# Define Disturbance Distribution
disturbance = True
mean = 0
std = 2
disturb_max = -6.0*U_max


for x in x_fliped_range:
    for y in y_fliped_range:
        x0 = np.array([x,y])
        feasible_candidates.append(x0)


# Define Robot
robot = SingleIntegrator2D(x0, dt, ax=ax, id = 0, color='r',palpha=1.0, num_constraints_hard = 0, num_constraints_soft = 0, plot=False)

# Define u_list
u_step = 0.1
u_list = np.arange(start=-U_max,stop=U_max+u_step,step=u_step)
u2d_list = np.zeros(shape=(u_list.shape[0]**2,2))
for i in range(u_list.shape[0]):
    for j in range(u_list.shape[0]):
        if u_list[i]==0 and u_list[j]==0:
            continue
        u = np.array([u_list[i],u_list[j]])
        u /= np.linalg.norm(u)
        u *= U_max
        u2d_list[u_list.shape[0]*i+j,:] = u.reshape(-1,2)
"""
for x0 in feasible_candidates:
    print(x0)
    if disturbance:
        y_disturb = norm.pdf(x0, loc=mean, scale=std)[0] * disturb_max
        u_disturb = np.array([0.0, y_disturb]).reshape(2,1)
    for i in range(u2d_list.shape[0]):
        robot.X = x0.reshape(-1,1)
        u = u2d_list[i,:].reshape(2,1)
        if disturbance:
            u_next = u + u_disturb
        else:
            u_next = u
        robot.step(u_next)
        new_pos = robot.X
        x = new_pos[0]
        y = new_pos[1]
        if y>y_max-0.2 or y<y_min or x>x_max or x<x_min:
            continue
        pos_key = str(int((x-x_min)/step))+","+str(int((y-y_min)/step))
        x0_key = str(int((x0[0]-x_min)/step))+","+str(int((x0[1]-y_min)/step))

        if (control_hash_table.get(pos_key)):
            backward_set, ulist = control_hash_table.get(pos_key)
            backward_set = np.append(backward_set,np.array([x0_key]),axis=0)
            ulist = np.append(ulist,np.array([u]).reshape(-1,1),axis=1)
        else:
            backward_set = np.array([x0_key])
            ulist = np.array([u]).reshape(-1,1)
        control_hash_table.update({pos_key: (backward_set,ulist)})
"""

with multiprocessing.Pool() as pool:
    for (x0_key, forward_set, ulist_ford) in pool.map(discretize_alpha_forward_cal,feasible_candidates):
        for idx, forward_cell in enumerate(forward_set):
            if (control_hash_table.get(forward_cell)):
                backward_set,ulist = control_hash_table.get(forward_cell)
                backward_set = np.append(backward_set,np.array([x0_key]))
                ulist = np.append(ulist,np.array([ulist_ford[:,idx]]).reshape(-1,1),axis=1)
            else:
                backward_set = np.array([x0_key])
                ulist = np.array([ulist_ford[:,idx]]).reshape(-1,1) 
            control_hash_table.update({forward_cell: (backward_set, ulist)})

x0 = np.array([5.0,0.0])
robot = SingleIntegrator2D(x0, dt, ax=ax, id = 0, color='r',palpha=1.0, num_constraints_hard = 0, num_constraints_soft = 0, plot=True)

alpha_step = 0.5
alpha_list = np.arange(start=0,stop=50.0+alpha_step,step=alpha_step)


active_safe_set_id = 0
delta_t = 0.0
with writer.saving(fig, movie_name, 100): 
    for i in range(num_steps):
        current_pos = robot.X
        if disturbance:
            y_disturb = norm.pdf(current_pos, loc=mean, scale=std)[0] * disturb_max
            u_disturb = np.array([0.0, y_disturb[0]]).reshape(2,1)
        current_pos_key = str(int((current_pos[0]-x_min)/step))+","+str(int((current_pos[1]-y_min)/step))
        Safe_Set_Series.id = active_safe_set_id
        centroid = Safe_Set_Series.return_centroid(None)
        r = Safe_Set_Series.return_radius()
        x_target_range = np.arange(start=centroid[0]-r,stop=centroid[0]+r+step,step=step)
        y_target_range = np.arange(start=centroid[1]-r,stop=centroid[1]+r+step,step=step)
        target_pos = np.array([])
        in_target_pos = {}

        for x in x_target_range:
            for y in y_target_range:
                if ((x-centroid[0]**2)+(y-centroid[1]**2)) <= r**2:
                    pos_key = str(int((x-x_min)/step))+","+str(int((y-y_min)/step))
                    target_pos = np.append(target_pos,np.array([pos_key]),axis=0)
                    in_target_pos.update({pos_key: True})

        if False:
            next_target_pos = np.array([])
            in_next_target_pos = {}
            centroid_next = Safe_Set_Series.return_centroid(active_safe_set_id+1)
            x_next_target_range = np.arange(start=centroid_next[0]-r,stop=centroid_next[0]+r+step,step=step)
            y_next_target_range = np.arange(start=centroid_next[1]-r,stop=centroid_next[1]+r+step,step=step)
            for x in x_next_target_range:
                for y in y_next_target_range:
                    if ((x-centroid_next[0]**2)+(y-centroid_next[1]**2)) <= r**2:
                        pos_key = str(int((x-x_min)/step))+","+str(int((y-y_min)/step))
                        next_target_pos = np.append(next_target_pos,np.array([pos_key]),axis=0)
                        in_next_target_pos.update({pos_key: True})            
            target_pos_filtered = np.array([])
            in_target_pos_filtered = {}
            while next_target_pos.size > 0:
                node = next_target_pos[0]
                next_target_pos = np.delete(next_target_pos, obj=0, axis=0)
                if (control_hash_table.get(node)):
                    backward_set,_ = control_hash_table.get(node)
                else:
                    continue
                filtered_backward_set = np.array([])
                for idx,cell in enumerate(backward_set):
                    if (in_next_target_pos.get(cell)):
                        continue
                    else:
                        if (in_target_pos.get(cell)):
                            if in_target_pos_filtered.get(cell):
                                continue
                            else:
                                target_pos_filtered = np.append(target_pos_filtered,np.array([cell]),axis=0)
                                in_target_pos_filtered.update({cell: True})
                        else:
                            filtered_backward_set = np.append(filtered_backward_set,np.array([cell]),axis=0)
                            in_next_target_pos.update({cell: True})
                if np.size(filtered_backward_set)==0:
                    continue
                if np.size(next_target_pos)> 0:
                    next_target_pos = np.append(next_target_pos,filtered_backward_set,axis=0)
                else:
                    next_target_pos = filtered_backward_set
        else:
            target_pos_filtered = target_pos

        possible_u = np.array([])
        while target_pos_filtered.size > 0:
            node = target_pos_filtered[0]
            target_pos_filtered = np.delete(target_pos_filtered, obj=0, axis=0)
            
            if (control_hash_table.get(node)):
                backward_set,ulist = control_hash_table.get(node)
            else:
                continue

            filtered_backward_set = np.array([])
            for idx,cell in enumerate(backward_set):
                if (in_target_pos.get(cell)):
                    continue
                else:
                    if cell == current_pos_key:
                        if np.size(possible_u)==0:
                            possible_u = np.array([ulist[:,idx]]).reshape(-1,1)
                        else:
                            possible_u = np.append(possible_u,np.array([ulist[:,idx]]).reshape(-1,1),axis=1)
                    else:
                        filtered_backward_set = np.append(filtered_backward_set,np.array([cell]),axis=0)
                        in_target_pos.update({cell: True})
                    
            if np.size(filtered_backward_set)==0:
                continue
            if np.size(target_pos)> 0:
                target_pos_filtered = np.append(target_pos_filtered,filtered_backward_set,axis=0)
            else:
                target_pos_filtered = filtered_backward_set

        possible_u_filtered = np.array([])
        h1, dh1_dx =  Safe_Set_Series.safe_set_h(robot)
        h2 = y_max-robot.X[1]
        dh2_dx = -np.array([0,1]).reshape(1,2)
        h1dot_list = np.array([])
        if len(possible_u!=0):
            for idx in range(possible_u.shape[1]):
                u = possible_u[:,idx].reshape(-1,1)
                if disturbance:
                    u_eff = u + u_disturb
                else:
                    u_eff = u
                h1dot = dh1_dx@robot.f() + dh1_dx@robot.g()@u_eff
                h2dot = dh2_dx@robot.f() + dh2_dx@robot.g()@u_eff
                if np.size(possible_u_filtered)==0:
                    possible_u_filtered = u_eff.reshape(-1,1)
                else:
                    possible_u_filtered = np.append(possible_u_filtered,u_eff.reshape(-1,1),axis=1)
                h1dot_list = np.append(h1dot_list,np.array(h1dot).reshape(-1,),axis=0)

        if np.size(possible_u_filtered)==0:
            if active_safe_set_id < num_points-2:
                active_safe_set_id += 1
                print(active_safe_set_id)
                continue
            else:
                break

        
        idx = np.argmax(h1dot_list)
        u_best = possible_u_filtered[:,idx].reshape(-1,1)
        robot.step(u_best)
        robot.render_plot()
        fig.canvas.draw()
        fig.canvas.flush_events()
        print(i)
        print(active_safe_set_id)
        delta_t += dt

        if Safe_Set_Series.sets_reached(robot) or delta_t>=tf/(num_points-2):
            if active_safe_set_id < num_points-2:
                active_safe_set_id += 1
                delta_t = 0
            else:
                break
        
        writer.grab_frame()

        
plt.ioff()
plt.show()
