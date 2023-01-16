import numpy as np
import math
import time
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from robot_models.SingleIntegrator2D import *
from robot_models.Unicycle2D import *
from Trajectory_Model import *
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

plt.rcParams.update({'font.size': 15}) #27
# Sim Parameters                  
dt = 0.05
t = 0

# Define Parameters for CLF and CBF
alpha = 0.8
betta1 = 0.8
betta2 = 0.8
d_max = 0.2
alpha_cbf = 7.0 

num_obstacles = 0

# for curved trajectory
y_max = 6.0
tf = 25
num_steps = int(tf/dt)
U_max = 1.0


# Define Trajectory
radius = 5
num_points = 11
trajectory_points = PointsInCircum_with_theta(radius,(num_points-1)*2)[0:num_points]
trajectory_time = math.pi*radius/(U_max/math.sqrt(2))
trajectory = Trajectory2D(trajectory_points=trajectory_points,tot_time=trajectory_time,poly_degree=5,type="Unicycle")

# figure
plt.ion()
fig = plt.figure()#(dpi=100)
# fig.set_size_inches(33, 15)
ax = plt.axes(projection ="3d",xlim=(-6,6),ylim=(-2,8), zlim=(0,4.0))
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_box_aspect([1,1,4.0/10])

# Set Bounding Box
x_start = -5
x_end = 5
y_start = y_max
y_end = 8
z_start = 0
z_end = 4

x_ = np.linspace(start=x_start, stop=x_end, num=10)
y_ = np.linspace(start=y_start, stop=y_end, num=10)
z_ = np.linspace(start=z_start, stop=z_end, num=10)
x, y = np.meshgrid(x_, y_)
z = np.zeros(shape=x.shape)+z_start
ax.plot_surface(x, y, z, color='black')
z = np.zeros(shape=x.shape)+z_end
ax.plot_surface(x, y, z, color='black')
x, z = np.meshgrid(x_,z_)
y = np.zeros(x.shape)+y_start
ax.plot_surface(x, y ,z, color='black')
y = np.zeros(x.shape)+y_end
ax.plot_surface(x, y ,z, color='black')
y, z = np.meshgrid(y_,z_)
x = np.zeros(y.shape)+x_start
ax.plot_surface(x, y ,z, color='black')
x = np.zeros(y.shape)+x_end
ax.plot_surface(x, y ,z, color='black')


ax.plot(trajectory_points[:,0],trajectory_points[:,1],'r--')
max_allowed_trajectory = PointsInCircum(5+d_max,20)[0:11]
min_allowed_trajectory = PointsInCircum(5-d_max,20)[0:11]
ax.plot(max_allowed_trajectory[:,0],max_allowed_trajectory[:,1],'k')
ax.plot(min_allowed_trajectory[:,0],min_allowed_trajectory[:,1],'k')

movie_name = 'curved_trajectory_with_disturb_3D.mp4'

metadata = dict(title='Movie Test', artist='Matplotlib',comment='Movie support!')
writer = FFMpegWriter(fps=15, metadata=metadata)

# Define Disturbance 
u_d = cp.Parameter((4,1), value = np.zeros((4,1)))

# Define Unrelaxed Optimization Problem
u1 = cp.Variable((2,1))
u1_ref = cp.Parameter((2,1), value = np.zeros((2,1)) )
num_constraints_hard1 = 2
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
num_constraints_hard2 = 2
num_constraints_soft2 = 1
A2_hard = cp.Parameter((num_constraints_hard2,2),value=np.zeros((num_constraints_hard2,2)))
b2_hard = cp.Parameter((num_constraints_hard2,1),value=np.zeros((num_constraints_hard2,1)))
A2_soft = cp.Parameter((num_constraints_soft2,2),value=np.zeros((num_constraints_soft2,2)))
b2_soft = cp.Parameter((num_constraints_soft2,1),value=np.zeros((num_constraints_soft2,1)))

slack_constraints2 = cp.Variable((num_constraints_soft1,1))
const2 = [A2_hard @ u2 <= b2_hard, A2_soft @ u2 <= b2_soft + slack_constraints2, cp.norm2(u2) <= U_max]
objective2 = cp.Minimize( cp.sum_squares( u2 - u2_ref )  + 1000*cp.sum_squares(slack_constraints2))
relaxed_controller = cp.Problem( objective2, const2 ) 

# Define Robot
robot = Unicycle2D(np.array([5,0,0,np.pi/2.0]), dt, ax, id = 0, color='r',palpha=1.0, alpha=alpha_cbf, num_constraints_hard = num_constraints_hard1, num_constraints_soft = num_constraints_soft1)

# Define Lists for Plotting
tp = np.arange(start=0,stop=tf,step=dt).reshape((num_steps, ))
u_list = np.zeros((2, num_steps))
u_ref_list = np.zeros((2, num_steps))
x_list = np.zeros((4,num_steps))
x_target_list = np.zeros((4,num_steps))

disturbance = True

with writer.saving(fig, movie_name, 100): 

    for i in range(num_steps):

        if disturbance:
            if (t >= 6 and t<=12) :
                u_d.value = np.array([0.0,1.5,0.0,0.0]).reshape(4,1)
            else:
                u_d.value = np.zeros((4,1))

        x_r = trajectory.get_current_target(t)
        x_target_list[:,i] = x_r.reshape(4,)
        x_list[:,i] = robot.X.reshape(4,)
        x_r_dot = trajectory.x_r_dot(t)

        v, dv_dx = robot.lyapunov(x_r) 
        robot.A1_soft[0,:] = dv_dx@robot.g()
        robot.b1_soft[0] = dv_dx@(x_r_dot-robot.f()) - alpha*v - dv_dx@u_d.value

        h1, dh1_dx = robot.static_safe_set(x_r,d_max)    
        robot.A1_hard[0,:] = -dh1_dx@robot.g()
        robot.b1_hard[0] = -dh1_dx@(x_r_dot-robot.f()) + betta1*h1 + dh1_dx@u_d.value

        
        h2 = robot.X[1]-y_max
        robot.A1_hard[1,:] = np.array([0,1,0,0]).reshape(1,4)@robot.g()
        robot.b1_hard[1] = -np.array([0,1,0,0]).reshape(1,4)@u_d.value - betta2*h2 - np.array([0,1,0,0]).reshape(1,4)@robot.f()   

        A1_soft.value = robot.A1_soft
        b1_soft.value = robot.b1_soft
        A1_hard.value = robot.A1_hard
        b1_hard.value = robot.b1_hard

        u1_ref.value = robot.nominal_input(x_r)
        
        u_ref_list[:,i] = u1_ref.value.reshape(2,)
        try:
            constrained_controller.solve(solver=cp.GUROBI, reoptimize=True)
        except:
            print("Constrained Controller Failed")
            robot.A1_hard[0,:] = np.zeros((1,2))
            robot.b1_hard[0] = 0
            """
            robot.A1_hard[1,:] = np.zeros((1,2))
            robot.b1_hard[1] = 0
            """
            print("here")
            A2_hard.value = robot.A1_hard
            b2_hard.value = robot.b1_hard
            A2_soft.value = robot.A1_soft
            b2_soft.value = robot.b1_soft

            u2_ref.value = u1_ref.value
            relaxed_controller.solve(solver=cp.GUROBI, reoptimize=True)
            robot.nextU = u2.value
            
        if constrained_controller.status != "optimal":
            robot.A1_hard[0,:] = np.zeros((1,2))
            robot.b1_hard[0] = 0
            """
            robot.A1_hard[1,:] = np.zeros((1,2))
            robot.b1_hard[1] = 0
            """
            print("here")
            A2_hard.value = robot.A1_hard
            b2_hard.value = robot.b1_hard
            A2_soft.value = robot.A1_soft
            b2_soft.value = robot.b1_soft

            u2_ref.value = u1_ref.value
            relaxed_controller.solve(solver=cp.GUROBI, reoptimize=True)
            robot.nextU = u2.value
        else:
            robot.nextU = u1.value

        u_list[:,i] = robot.nextU.reshape(2,)
        robot.step(robot.nextU, u_d.value)
        robot.render_plot()

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
plt.legend(['u_1', 'u_2'])

plt.figure(4)
plt.plot(x_list[0,:], x_list[1,:])
plt.plot(x_target_list[0,:], x_target_list[1,:])
plt.legend(['x', 'x_target'])
plt.show()