import numpy as np
import math
import time
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from robot_models.SingleIntegrator2D import *
from Safe_Set_Series import *
from matplotlib.animation import FFMpegWriter
from Trajectory_Model import *

plt.rcParams.update({'font.size': 15}) #27
# Sim Parameters                  
dt = 0.05
t = 0
tf = 25
num_steps = int(tf/dt)

# Define Parameters for CLF and CBF
alpha_cbf = 7.0 
num_obstacles = 0
U_max = 3.0

# Define Series of Safe Sets
num_points = 11
centroids = PointsInCircum(10,(num_points-1)*2)[1:num_points]
centroids[6,1] = centroids[6,1] - 1.0
centroids[7,1] = centroids[7,1] + 7;
centroids[8,1] = centroids[8,1] + 9;
centroids[9,1] = centroids[9,1] + 11;
centroids[9,0] = centroids[9,0] - 2;
radii = np.zeros((centroids.shape[0],))+0.7
alpha_list = np.zeros((centroids.shape[0],))+1.0 #0.8
Safe_Set_Series = Safe_Set_Series2D(centroids=centroids,radii=radii,alpha_list=alpha_list)

# Plot                  
plt.ion()
fig = plt.figure()
ax = plt.axes(xlim=(-2,12),ylim=(-7,10)) 
ax.set_xlabel("X")
ax.set_ylabel("Y")

for i in range(0,num_points-1):
    circle = patches.Circle(centroids[i,:], radius=radii[i], color='green')
    ax.add_patch(circle)
ax.axis('equal')

metadata = dict(title='Movie Test', artist='Matplotlib',comment='Movie support!')
writer = FFMpegWriter(fps=15, metadata=metadata)
movie_name = 'series_of_safesets_with_disturb.mp4'


# Define Disturbance 
u_d = cp.Parameter((2,1), value = np.zeros((2,1)))

# Define Unrelaxed Optimization Problem
u1 = cp.Variable((2,1))
u1_ref = cp.Parameter((2,1), value = np.zeros((2,1)) )
num_constraints_hard1 = 1
num_constraints_soft1 = 0
A1_hard = cp.Parameter((num_constraints_hard1,2),value=np.zeros((num_constraints_hard1,2)))
b1_hard = cp.Parameter((num_constraints_hard1,1),value=np.zeros((num_constraints_hard1,1)))
const1 = [A1_hard @ u1 <= b1_hard, cp.norm(u1) <= U_max]
objective1 = cp.Minimize(cp.norm(u1-u1_ref))
constrained_controller = cp.Problem( objective1, const1 ) 

# Define Robot
x0 = np.array([10,0])
robot = SingleIntegrator2D(x0, dt, ax, id = 0, color='r',palpha=1.0, alpha=alpha_cbf, num_constraints_hard = num_constraints_hard1, num_constraints_soft = num_constraints_soft1, num_obstacles=num_obstacles)

disturbance = True

# Define Lists for Plotting
tp = np.arange(start=0,stop=tf,step=dt).reshape((num_steps, ))
u_list = np.zeros((2, num_steps))
u_ref_list = np.zeros((2, num_steps))
x_list = np.zeros((2,num_steps))
x_target_list = np.zeros((2,num_steps))


with writer.saving(fig, movie_name, 100): 

    for i in range(num_steps):

        if disturbance:
            if (t >= 6 and t<=10) :
                u_d.value = np.array([0.0,2.0]).reshape(2,1) #2.5
            else:
                u_d.value = np.zeros((2,1))

        Safe_Set_Series.update_targets(robot = robot, failed=False)
        robot.A1_hard, robot.b1_hard = Safe_Set_Series.safe_set_constraints(robot = robot, u_d=u_d)
        A1_hard.value = robot.A1_hard
        b1_hard.value = robot.b1_hard

        u1_ref.value = robot.nominal_input(Safe_Set_Series.centroids[Safe_Set_Series.id,:].reshape(2,1))
        
        u_ref_list[:,i] = u1_ref.value.reshape(2,)

        constrained_controller.solve(solver=cp.GUROBI, reoptimize=True)
        # print("Au value: ", A1_hard.value@u1.value)
        # print("b value: ", b1_hard.value)
        
        # if constrained_controller.status == "optimal" and A1_hard.value@u1.value > b1_hard.value:
        #     print("ERROR")
        
        while constrained_controller.status != "optimal":# or (A1_hard@u1 - b1_hard).value > 0:
            print("here")
            Safe_Set_Series.update_targets(robot=robot,failed=True)
            robot.A1_hard, robot.b1_hard = Safe_Set_Series.safe_set_constraints(robot = robot, u_d=u_d)
            A1_hard.value = robot.A1_hard
            b1_hard.value = robot.b1_hard
            u1_ref.value = robot.nominal_input(Safe_Set_Series.centroids[Safe_Set_Series.id,:].reshape(2,1))
            u_ref_list[:,i] = u1_ref.value.reshape(2,)
            constrained_controller.solve(solver=cp.GUROBI, reoptimize=True)

        robot.nextU = u1.value + u_d.value
        robot.step(robot.nextU)
        robot.render_plot()

        u_list[:,i] = robot.nextU.reshape(2,)

        t = t + dt
        fig.canvas.draw()
        fig.canvas.flush_events()

        writer.grab_frame()

plt.ioff()   

plt.figure(2)
plt.plot(tp, u_list[0,:])
plt.plot(tp, u_list[1,:])
plt.legend(['u_1', 'u_2'])

plt.figure(3)
plt.plot(tp, u_ref_list[0,:])
plt.plot(tp, u_ref_list[1,:])
plt.legend(['u_1_ref', 'u_2_ref'])

plt.show()

