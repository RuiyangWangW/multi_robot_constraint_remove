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

# for curved trajectory
y_max = 6.0
x0 = np.array([5,0])
tf = 15
num_steps = int(tf/dt)
U_max = 2.0


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


# Define Trajectory
radius = 5
num_points = 11
trajectory_points = PointsInCircum(radius,(num_points-1)*2)[0:num_points]
trajectory_time = math.pi*radius/(U_max/math.sqrt(2))
trajectory = Trajectory2D(trajectory_points=trajectory_points,tot_time=trajectory_time,poly_degree=5)

# Define Disturbance 
u_d = cp.Parameter((2,1), value = np.zeros((2,1)))

# Define Unrelaxed Optimization Problem
u1 = cp.Variable((2,1))
u1_ref = cp.Parameter((2,1), value = np.zeros((2,1)) )
num_constraints_hard1 = 3
num_constraints_soft1 = 1
A1_hard = cp.Parameter((num_constraints_hard1,2),value=np.zeros((num_constraints_hard1,2)))
b1_hard = cp.Parameter((num_constraints_hard1,1),value=np.zeros((num_constraints_hard1,1)))
A1_soft = cp.Parameter((num_constraints_soft1,2),value=np.zeros((num_constraints_soft1,2)))
b1_soft = cp.Parameter((num_constraints_soft1,1),value=np.zeros((num_constraints_soft1,1)))
slack_constraints1 = cp.Variable((num_constraints_soft1,1))
const1 = [A1_hard @ u1 <= b1_hard, A1_soft @ u1 <= b1_soft + slack_constraints1, cp.norm2(u1) <= U_max]
objective1 = cp.Minimize( cp.sum_squares( u1 - u1_ref ) + 1000*cp.sum_squares(slack_constraints1))
constrained_controller = cp.Problem( objective1, const1 ) 

# Define Relaxed Optimization Problem
u2 = cp.Variable((2,1))
u2_ref = cp.Parameter((2,1), value = np.zeros((2,1)) )
num_constraints_hard2 = 3
num_constraints_soft2 = 1
A2_hard = cp.Parameter((num_constraints_hard2,2),value=np.zeros((num_constraints_hard2,2)))
b2_hard = cp.Parameter((num_constraints_hard2,1),value=np.zeros((num_constraints_hard2,1)))
A2_soft = cp.Parameter((num_constraints_soft2,2),value=np.zeros((num_constraints_soft2,2)))
b2_soft = cp.Parameter((num_constraints_soft2,1),value=np.zeros((num_constraints_soft2,1)))

slack_constraints2 = cp.Variable((num_constraints_soft1,1))
const2 = [A2_hard @ u2 <= b2_hard, A2_soft @ u2 <= b2_soft + slack_constraints2, cp.norm2(u2) <= U_max]
objective2 = cp.Minimize( cp.sum_squares( u2 - u2_ref )  + 1000*cp.sum_squares(slack_constraints2))
relaxed_controller = cp.Problem( objective2, const2 ) 

# Define Lists for Plotting
tp = np.arange(start=0,stop=tf,step=dt).reshape((num_steps, ))

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
y_range = np.arange(start=y_min, stop=y_max, step=step)
None_list = np.array([None,None]).reshape(2,1)
u_thresh = 1e-4

for x_0 in x_range:
    for y_0 in y_range:
        travelled_path = []
        x0 = np.array([x_0,y_0])
        # Define Robot
        robot = SingleIntegrator2D(x0, dt, ax=ax, id = 0, color='r',palpha=1.0, num_constraints_hard = num_constraints_hard1, num_constraints_soft = num_constraints_soft1, plot=True)
        t = 0
        for i in range(num_steps):
            travelled_path.append(robot.X)
            pos_key = str(robot.X[0])+","+str(robot.X[1])
            u_next = control_hash_table.get(pos_key)
            u_none = (u_next==None_list)[0][0]
            if u_none:
                if disturbance:
                    y_disturb = norm.pdf(robot.X[0], loc=mean, scale=std)[0] * disturb_max
                    u_d.value = np.array([0.0, y_disturb]).reshape(2,1)
                    disturb_list[i] = y_disturb

                x_r = trajectory.get_current_target(t)
                x_r_dot = trajectory.x_r_dot(t)

                v, dv_dx = robot.lyapunov(x_r) 
                robot.A1_soft[0,:] = dv_dx@robot.g()
                robot.b1_soft[0] = dv_dx@(x_r_dot-robot.f()) - alpha*v - dv_dx@robot.g()@u_d.value

                h1, dh1_dx = robot.static_safe_set(np.zeros((2,1)),radius+d_max)    
                robot.A1_hard[0,:] = -dh1_dx@robot.g()
                robot.b1_hard[0] = dh1_dx@robot.f() + betta1*h1 + dh1_dx@robot.g()@u_d.value

                h2, dh2_dx = robot.static_safe_set(np.zeros((2,1)),radius-d_max)
                h2 = -h2
                dh2_dx = -dh2_dx
                robot.A1_hard[1,:] = -dh2_dx@robot.g()
                robot.b1_hard[1] = dh2_dx@robot.f() + betta2*h2 + dh2_dx@robot.g()@u_d.value

                h3 = y_max-robot.X[1]
                robot.A1_hard[2,:] = np.array([0,1]).reshape(1,2)@robot.g()
                robot.b1_hard[2] = -np.array([0,1]).reshape(1,2)@robot.g()@u_d.value + betta3*h3 - np.array([0,1]).reshape(1,2)@robot.f()

                A1_soft.value = robot.A1_soft
                b1_soft.value = robot.b1_soft
                A1_hard.value = robot.A1_hard
                b1_hard.value = robot.b1_hard

                u1_ref.value = robot.nominal_input(x_r)

                try: 
                    constrained_controller.solve(solver=cp.GUROBI, reoptimize=True)
                    if  constrained_controller.status!="optimal":
                        for pos in travelled_path:
                            pos_key = str(pos[0])+","+str(pos[1])
                            control_hash_table.update({pos_key: "Fail"})
                        break
                    else:
                        u_next = u1.value + u_d.value
                        robot.nextU = u_next
                        robot.step(robot.nextU)
                        #Needs to round robot's position to cell position
                        robot.X[0] = x_range[int((robot.X[0]-x_min)/step)]
                        robot.X[1] = y_range[int((robot.X[1]-y_min)/step)]
                        control_hash_table.update({pos_key: robot.nextU})
                except:
                    for pos in travelled_path:
                        pos_key = str(pos[0])+","+str(pos[1])
                        control_hash_table.update({pos_key: "Fail"})
                    break
            else:
                if u_next == "Fail":
                    for pos in travelled_path:
                        pos_key = str(pos[0])+","+str(pos[1])
                        control_hash_table.update({pos_key: "Fail"})
                    break
                else:
                    robot.nextU = u_next
                    robot.step(robot.nextU)
                    #Needs to round robot's position to cell position
                    robot.X[0] = x_range[int((robot.X[0]-x_min)/step)]
                    robot.X[1] = y_range[int((robot.X[1]-y_min)/step)]
                    continue
            if u_none and u_next!="Fail":
                print(u_next[0])
                if u_next[0]<=u_thresh and u_next[1]<=u_thresh:
                    break
            t = t + dt
            robot.render_plot()
            fig.canvas.draw()
            fig.canvas.flush_events()
            
for x_0 in x_range:
    for y_0 in y_range:
        pos_key = str(x_0)+","+str(y_0)
        u_next = control_hash_table.get(pos_key)
        if u_next == "Fail":
            plt.scatter(x_0,y_0,Color='b')
        else:
            plt.scatter(x_0,y_0,Color='r')

plt.show()