import numpy as np
import math
from scipy.stats import norm
import time
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from robot_models.SingleIntegrator2D import *
from Trajectory_Model import *
from matplotlib.animation import FFMpegWriter


plt.rcParams.update({'font.size': 15}) #27
# Sim Parameters                  
dt = 0.05
t = 0

# Define Parameters for CLF and CBF
alpha = 0.8
betta1 = 0.8
betta2 = 0.8
betta3 = 0.8
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

# Define Disturbance Distribution
disturbance = True
mean = 0
std = 2
disturb_list = np.zeros((num_steps,))
disturb_max = 6*U_max

#Define Search Map
control_hash_table = {}
x_min = -6
x_max = 6
y_min = -2 
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
# Define u_list
u_step = 0.1
u_list = np.arange(start=-U_max,stop=U_max+u_step,step=u_step)
u2d_list = np.zeros(shape=(u_list.shape[0]**2,2))
for i in range(u_list.shape[0]):
    for j in range(u_list.shape[0]):
        u2d_list[u_list.shape[0]*i+j,:] = np.array([u_list[i],u_list[j]])

for x0 in feasible_candidates:
    # Define Robot
    robot = SingleIntegrator2D(x0, dt, ax=ax, id = 0, color='r',palpha=1.0, num_constraints_hard = 0, num_constraints_soft = 0, plot=False)
    print(x0)
    if disturbance:
        y_disturb = norm.pdf(robot.X[0], loc=mean, scale=std)[0] * disturb_max
        u_disturb = np.array([0.0, y_disturb]).reshape(2,1)
    for i in range(u2d_list.shape[0]):
        u = u2d_list[i,:].reshape(2,1)
        u_next = u + u_disturb
        robot.nextU = u_next
        robot.step(robot.nextU)
        new_pos = robot.X
        robot.X = x0.reshape(-1,1)
        x = new_pos[0]
        y = new_pos[1]
        if y < 0 and (x**2+y**2)<(radius-d_max)**2 and (x**2+y**2)>(radius+d_max)**2:
            continue
        pos_key = str(int((x-x_min)/step))+","+str(int((y-y_min)/step))
        backward_set = control_hash_table.get(pos_key)
        none_list = np.array([backward_set == None]).reshape(-1,).tolist()
        x0_key = str(int((x0[0]-x_min)/step))+","+str(int((x0[1]-y_min)/step))
        if any(none_list):
            backward_set = np.array([x0_key])
        else:
            backward_set = np.append(backward_set,np.array([x0_key]),axis=0)

        control_hash_table.update({pos_key: backward_set})
        """
        robot.render_plot()
        fig.canvas.draw()
        fig.canvas.flush_events()
        """
plt.ioff()   


x_fail_list = []
y_fail_list = []
x_success_list = []
y_success_list = []
target_pos = np.array([-5,0]).reshape(2,1)
target_pos_key = str(int((target_pos[0]-x_min)/step))+","+str(int((target_pos[1]-y_min)/step))
success_list = np.array([target_pos_key])
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
    x = int(x)*step+x_min
    y = int(y)*step+y_min
    x_success_list.append(x)
    y_success_list.append(y)
    backward_set = control_hash_table.get(current)
    none_list = np.array([backward_set == None]).reshape(-1,).tolist()
    if any(none_list):
        continue

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

print(len(control_hash_table))
plt.plot(x_success_list,y_success_list,'b.')
print(len(x_success_list))
plt.show()