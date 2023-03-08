import numpy as np
import math
import time
import cvxpy as cp
import copy
import multiprocessing
from queue import PriorityQueue
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
movie_name = 'series_of_safesets_with_medium_wind.mp4'


#Define Search Map
control_hash_table = {}
in_control_hash_table = {}
step = 0.1
x_range = np.arange(start=x_min, stop=x_max+step, step=step)
x_fliped_range = np.flip(x_range)
y_range = np.arange(start=y_min, stop=y_max+step, step=step)
y_fliped_range = np.flip(y_range)
feasible_candidates = []

for x in x_fliped_range:
    for y in y_fliped_range:
        x0 = np.array([x,y])
        feasible_candidates.append(x0)


with multiprocessing.Pool() as pool:
    for (x0_key, forward_set, dist_ford) in pool.map(discretize_u_forward_cal,feasible_candidates):
        for idx, forward_cell in enumerate(forward_set):
            x = ""
            for i in range(len(forward_cell)):
                a = forward_cell[i]
                if a!=',':
                    x += a
                else:
                    break
            y = forward_cell[i+1:]
            x = x_range[int(x)]
            y = y_range[int(y)]
            if y > y_max or y < y_min or x > x_max or x < x_min:
                continue
            if (in_control_hash_table.get(forward_cell)==None):
                backward_set = np.array([x0_key])
                dist_back = np.array([dist_ford[idx]])
            else:
                backward_set, dist_back = control_hash_table.get(forward_cell)
                backward_set = np.append(backward_set,np.array([x0_key]))
                dist_back = np.append(dist_back,np.array([dist_ford[idx]]))
            control_hash_table.update({forward_cell: (backward_set, dist_back)})
            in_control_hash_table.update({forward_cell: True})


x0 = np.array([5.0,0.0])
robot = SingleIntegrator2D(x0, dt, ax=ax, id = 0, color='r',palpha=1.0, num_constraints_hard = 0, num_constraints_soft = 0, plot=True)

final_centroids = Safe_Set_Series.centroids[-1,:]
final_target_centroid = np.array([final_centroids]).reshape(2,1)
r = Safe_Set_Series.radii[-1]
x_final_target_range = np.arange(start=final_target_centroid[0]-r,stop=final_target_centroid[0]+r+step,step=step)
y_final_target_range = np.arange(start=final_target_centroid[1]-r,stop=final_target_centroid[1]+r+step,step=step)

success_list = np.array([])
pos_in_success_table = {}
for x in x_final_target_range:
    for y in y_final_target_range:
        if ((x-final_target_centroid[0])**2 + (y-final_target_centroid[1])**2) <= r**2:
            target_pos = np.array([x,y]).reshape(2,1)
            target_pos_key = str(int((target_pos[0]-x_min)/step))+","+str(int((target_pos[1]-y_min)/step))
            success_list = np.append(success_list,np.array([target_pos_key]))
            pos_in_success_table.update({target_pos_key: True})

while success_list.size > 0:
    current = success_list[0]
    success_list = np.delete(success_list, obj=0, axis=0)
    print(success_list.size)
    if (in_control_hash_table.get(current)==None):
        continue
    else:
        backward_set, _ = control_hash_table.get(current)
    filtered_backward_set = None
    for i in range(backward_set.size):
        has_been_pushed = pos_in_success_table.get(backward_set[i])
        if has_been_pushed==None:
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

x_success_list = []
y_success_list = []
for i, pos in enumerate(pos_in_success_table):
    current = pos
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


active_safe_set_id = 0
delta_t = 0.0
final_path = []
chosen_node = str(int((x0[0]-x_min)/step))+","+str(int((x0[1]-y_min)/step))
with writer.saving(fig, movie_name, 100): 
    for i in range(num_steps):
        current_pos = robot.X
        current_pos_key = chosen_node
        Safe_Set_Series.id = active_safe_set_id
        centroid = Safe_Set_Series.return_centroid(Safe_Set_Series.id)
        if len(final_path) == 0:
            possible_node_list = PriorityQueue()
            in_path_list = {}
            pos_key = str(int((centroid[0]-x_min)/step))+","+str(int((centroid[1]-y_min)/step))
            in_success_table = pos_in_success_table.get(pos_key)
            if (in_success_table):
                possible_node_list.put((0, [pos_key]))
                in_path_list.update({pos_key: True})

            if possible_node_list.empty():
                active_safe_set_id += 1
                continue

            while ~possible_node_list.empty():
                prev_weight, possible_path = possible_node_list.get()
                node = possible_path[-1]

                if node == current_pos_key:
                    final_path = possible_path
                    final_path.pop(-1)
                    break

                if (in_control_hash_table.get(node)==None):
                    continue
                else:
                    backward_set,dist_back = control_hash_table.get(node)

                for idx,cell in enumerate(backward_set):
                    if in_path_list.get(cell) == True:
                        continue
                    new_path = copy.deepcopy(possible_path)
                    new_path.append(cell)
                    weight = dist_back[idx] + prev_weight
                    possible_node_list.put((weight, new_path))
                    in_path_list.update({cell: True})

        chosen_node = final_path.pop(-1)
        chosen_x = ""
        for i in range(len(chosen_node)):
            a = chosen_node[i]
            if a!=',':
                chosen_x += a
            else:
                break
        chosen_y = chosen_node[i+1:]
        chosen_x = x_range[int(chosen_x)] 
        chosen_y = y_range[int(chosen_y)]
        robot.X = np.array([chosen_x,chosen_y]).reshape(-1,1)
        robot.render_plot()
        fig.canvas.draw()
        fig.canvas.flush_events()
        print(active_safe_set_id)
        #delta_t += dt

        if len(final_path)==0 or delta_t>=tf/(num_points-2):
            if active_safe_set_id < num_points-2:
                active_safe_set_id += 1
                delta_t = 0
            else:
                break
        
        writer.grab_frame()


plt.ioff()



print(len(pos_in_success_table))
plt.plot(x_success_list,y_success_list,'b.')
print(len(x_success_list))

plt.show()

