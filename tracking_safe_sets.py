import numpy as np
import math
import time
import cvxpy as cp
import matplotlib.pyplot as plt
from robot_models.SingleIntegrator2D import *
from Safe_Set_Series import *
from matplotlib.animation import FFMpegWriter

plt.rcParams.update({'font.size': 15}) #27
# Sim Parameters                  
dt = 0.05
t = 0
tf = 10
num_steps = int(tf/dt)

# Define Parameters for CLF and CBF
alpha_cbf = 7.0 
num_obstacles = 0
U_max = 1.0


# Plot                  
plt.ion()
fig = plt.figure()
ax = plt.axes(xlim=(-2,12),ylim=(-7,7)) 
ax.set_xlabel("X")
ax.set_ylabel("Y")
metadata = dict(title='Movie Test', artist='Matplotlib',comment='Movie support!')
writer = FFMpegWriter(fps=15, metadata=metadata)
movie_name = 'series_of_safesets_without_disturb.mp4'


# Define Disturbance 
#u_d = cp.Parameter((2,1), value = np.zeros((2,1)))
u_d = np.zeros((2,1))

# Define Unrelaxed Optimization Problem
u1 = cp.Variable((2,1))
num_constraints_hard1 = 1
num_constraints_soft1 = 1
A1_hard = cp.Parameter((num_constraints_hard1,2),value=np.zeros((num_constraints_hard1,2)))
b1_hard = cp.Parameter((num_constraints_hard1,1),value=np.zeros((num_constraints_hard1,1)))
const1 = [A1_hard @ u1 <= b1_hard, cp.norm2(u1) <= U_max]
objective1 = cp.Minimize(cp.norm2(u1))
constrained_controller = cp.Problem( objective1, const1 ) 

# Define Robot
x0 = np.zeros((2,1))
robot = SingleIntegrator2D(x0, dt, ax, id = 0, color='r',palpha=1.0, alpha=alpha_cbf, num_constraints_hard = num_constraints_hard1, num_constraints_soft = num_constraints_soft1, num_obstacles=num_obstacles)

# Define Series of Safe Sets
centroids = np.array([[5,5],[10,0],[5,-5]]).T
radii = np.array([3,3,3])
alpha_list = np.array([0.8,0.8,0.8])
Safe_Set_Series = Safe_Set_Series2D(centroids=centroids,radii=radii,alpha_list=alpha_list)

disturbance = False

with writer.saving(fig, movie_name, 100): 

    for i in range(num_steps):

        if disturbance:
            if (robot.X[0] >= -2 and robot.X[0]<=4) :
                u_d.value = np.array([0.0,1.2]).reshape(2,1)
            else:
                u_d.value = np.zeros((2,1))

        Safe_Set_Series.update_targets(robot_pos=robot.X,failed=False)
        robot.A1_hard, robot.b1_hard = Safe_Set_Series.safe_set_constraints(robot_pos=robot.X, u_d=u_d, robot_f=robot.f(), robot_g=robot.g())
        A1_hard = robot.A1_hard
        b1_hard = robot.b1_hard

        constrained_controller.solve(solver=cp.GUROBI, reoptimize=True)
        while constrained_controller.status != "optimal":
            Safe_Set_Series.update_targets(robot_pos=robot.X,failed=True)

        print(robot.A1_hard@u1.value)
        print(robot.b1_hard)
        
        robot.nextU = u1.value + u_d
        robot.step(robot.nextU)
        robot.render_plot()

        t = t + dt
        fig.canvas.draw()
        fig.canvas.flush_events()

        writer.grab_frame()



