import numpy as np
import math
import time
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from robot_models.SingleIntegrator2D import *
from Safe_Set_Series import *
from matplotlib.animation import FFMpegWriter

plt.rcParams.update({'font.size': 15}) #27
# Sim Parameters                  
dt = 0.05
t = 0
tf = 15
num_steps = int(tf/dt)

# Define Parameters for CLF and CBF
alpha_cbf = 7.0 
num_obstacles = 0
U_max = 1.0

# Define Series of Safe Sets
centroids = np.array([[5,5],[10,0],[5,-5]]).T
radii = np.array([3,3,3])
alpha_list = np.array([0.8,0.8,0.8])
Safe_Set_Series = Safe_Set_Series2D(centroids=centroids,radii=radii,alpha_list=alpha_list)

# Plot                  
plt.ion()
fig = plt.figure()
ax = plt.axes(xlim=(-2,12),ylim=(-7,7)) 
ax.set_xlabel("X")
ax.set_ylabel("Y")
circle1 = patches.Circle(centroids[:,0], radius=radii[0], color='green')
circle2 = patches.Circle(centroids[:,1], radius=radii[1], color='green')
circle3 = patches.Circle(centroids[:,2], radius=radii[2], color='green')
ax.add_patch(circle1)
ax.add_patch(circle2)
ax.add_patch(circle3)
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
x0 = np.zeros((2,1))
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
                u_d.value = np.array([-1.0,0.0]).reshape(2,1)
            else:
                u_d.value = np.zeros((2,1))

        Safe_Set_Series.update_targets(robot_pos=robot.X,failed=False)
        robot.A1_hard, robot.b1_hard = Safe_Set_Series.safe_set_constraints(robot_pos=robot.X, u_d=u_d, robot_f=robot.f(), robot_g=robot.g())
        A1_hard = robot.A1_hard
        b1_hard = robot.b1_hard

        u1_ref.value = robot.nominal_input(Safe_Set_Series.centroids[:,Safe_Set_Series.id].reshape(2,1))
        
        u_ref_list[:,i] = u1_ref.value.reshape(2,)

        constrained_controller.solve(solver=cp.GUROBI, reoptimize=True)
        
        x_diff = (robot.X-Safe_Set_Series.centroids[:,Safe_Set_Series.id].reshape(2,1)).T
        print("h = ", Safe_Set_Series.evaluate_h(x_diff))
        print("Au = ", A1_hard@u1.value)
        print("b = ", b1_hard)
        
        if constrained_controller.status != "optimal" or A1_hard@u1.value > b1_hard:
            print("here")
            Safe_Set_Series.update_targets(robot_pos=robot.X,failed=True)
            robot.A1_hard, robot.b1_hard = Safe_Set_Series.safe_set_constraints(robot_pos=robot.X, u_d=u_d, robot_f=robot.f(), robot_g=robot.g())
            A1_hard = robot.A1_hard
            b1_hard = robot.b1_hard
            u1_ref.value = robot.nominal_input(Safe_Set_Series.centroids[:,Safe_Set_Series.id].reshape(2,1))
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

