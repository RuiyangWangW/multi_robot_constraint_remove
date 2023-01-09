import numpy as np
import time
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from robot_models.SingleIntegrator2D import *
from trajectory_model import *
from matplotlib.animation import FFMpegWriter


plt.rcParams.update({'font.size': 15}) #27
# Sim Parameters                  
dt = 0.05
tf = 12.0
num_steps = int(tf/dt)
t = 0

# Define Parameters for CLF and CBF
alpha = 0.8
betta1 = 0.8
betta2 = 0.8
d_max = 0.2
y_max = 1.0
alpha_cbf = 7.0 

num_obstacles = 0

# Plot                  
plt.ion()
fig = plt.figure()
ax = plt.axes(xlim=(-2,12),ylim=(-5,5)) 
ax.set_xlabel("X")
ax.set_ylabel("Y")
rect = patches.Rectangle((0, y_max), 10, 4, linewidth=1, edgecolor='none', facecolor='k')
# Add the patch to the Axes
ax.add_patch(rect)

ax.hlines(0, 0, 10, colors='r', linestyles='dashed')
ax.hlines(d_max, 0, 10, 'k')
ax.hlines(-d_max, 0, 10, 'k')
movie_name = 'straight_line_trajectory_without_disturb.mp4'
       
metadata = dict(title='Movie Test', artist='Matplotlib',comment='Movie support!')
writer = FFMpegWriter(fps=15, metadata=metadata)

# Define Disturbance 
u_d = cp.Parameter((2,1), value = np.zeros((2,1)))

# Define Unrelaxed Optimization Problem
u1 = cp.Variable((2,1))
u1_ref = cp.Parameter((2,1), value = np.zeros((2,1)) )
num_constraints_hard1 = 6
num_constraints_soft1 = 1
A1_hard = cp.Parameter((num_constraints_hard1,2),value=np.zeros((num_constraints_hard1,2)))
b1_hard = cp.Parameter((num_constraints_hard1,1),value=np.zeros((num_constraints_hard1,1)))
A1_soft = cp.Parameter((num_constraints_soft1,2),value=np.zeros((num_constraints_soft1,2)))
b1_soft = cp.Parameter((num_constraints_soft1,1),value=np.zeros((num_constraints_soft1,1)))
slack_constraints1 = cp.Variable((num_constraints_soft1,1))
const1 = [A1_hard @ u1 <= b1_hard, A1_soft @ u1 <= b1_soft + slack_constraints1]
objective1 = cp.Minimize( cp.sum_squares( u1 - u1_ref ) + 1000*cp.sum_squares(slack_constraints1))
constrained_controller = cp.Problem( objective1, const1 ) 

# Define Relaxed Optimization Problem
u2 = cp.Variable((2,1))
u2_ref = cp.Parameter((2,1), value = np.zeros((2,1)) )
num_constraints_hard2 = 6
num_constraints_soft2 = 1
A2_hard = cp.Parameter((num_constraints_hard2,2),value=np.zeros((num_constraints_hard2,2)))
b2_hard = cp.Parameter((num_constraints_hard2,1),value=np.zeros((num_constraints_hard2,1)))
A2_soft = cp.Parameter((num_constraints_soft2,2),value=np.zeros((num_constraints_soft2,2)))
b2_soft = cp.Parameter((num_constraints_soft2,1),value=np.zeros((num_constraints_soft2,1)))

slack_constraints2 = cp.Variable((num_constraints_soft1,1))
const2 = [A2_hard @ u2 <= b2_hard, A2_soft @ u2 <= b2_soft + slack_constraints2]
objective2 = cp.Minimize( cp.sum_squares( u2 - u2_ref )  + 1000*cp.sum_squares(slack_constraints2))
relaxed_controller = cp.Problem( objective2, const2 ) 

# Define Robot
robot = SingleIntegrator2D(np.array([0,0]), dt, ax, id = 0, color='r',palpha=1.0, alpha=alpha_cbf, num_constraints_hard = num_constraints_hard1, num_constraints_soft = num_constraints_soft1, num_obstacles=num_obstacles)
U_max = 1.2
tol = 0.01

# Define Trajectory
trajectory_points = np.array([[0,0],[1,0],[2,0],[3,0],[4,0],[5,0],[6,0],[7,0],[8,0],[9,0],[10,0]])
num_points = trajectory_points.shape[0]
trajectory_time = 10
trajectory = Trajectory2D(trajectory_points=trajectory_points,tot_time=trajectory_time,poly_degree=3)


# Define Lists for Plotting
tp = np.arange(start=0,stop=tf,step=dt).reshape((num_steps, ))
u_list = np.zeros((2, num_steps))
u_ref_list = np.zeros((2, num_steps))
x_list = np.zeros((2,num_steps))
x_target_list = np.zeros((2,num_steps))

disturbance = False
with writer.saving(fig, movie_name, 100): 

    for i in range(num_steps):

        if disturbance:
            if (t >= 3 and t<=7) :
                u_d.value = np.array([0.0,1.5]).reshape(2,1)
            else:
                u_d.value = np.zeros((2,1))

        curr_target = trajectory.get_current_target(t)
        x_target_list[:,i] = curr_target.reshape(2,)
        x_list[:,i] = robot.X.reshape(2,)
        x_r_dot = trajectory.x_r_dot(t)
        x_diff = (robot.X-curr_target).reshape(1,2)[0]
        v = np.linalg.norm(x_diff)**2
        robot.A1_soft[0,:] = 2*(x_diff)@robot.g()
        robot.b1_soft[0] = 2*(x_diff)@(x_r_dot-robot.f())-alpha*v-2*(x_diff)@(robot.f()+robot.g()@u_d.value)

        h1 = np.linalg.norm(x_diff)**2 - d_max
        robot.A1_hard[0,:] = 2*(x_diff)@robot.g()
        robot.b1_hard[0] = 2*(x_diff)@(x_r_dot-robot.f())-betta1*h1-2*(x_diff)@(robot.f()+robot.g()@u_d.value)

        h2 = robot.X[1]-y_max
        robot.A1_hard[1,:] = np.array([0,1]).reshape(1,2)@robot.g()
        robot.b1_hard[1] = -np.array([0,1]).reshape(1,2)@robot.g()@u_d.value - betta2*h2 - np.array([0,1]).reshape(1,2)@robot.f()
    

        robot.A1_hard[2:4,:] = np.eye(2)
        robot.A1_hard[4:,:] = -np.eye(2)
        robot.b1_hard[2:] = np.array([U_max,U_max,U_max,U_max]).reshape(4,1)

        A1_soft.value = robot.A1_soft
        b1_soft.value = robot.b1_soft
        A1_hard.value = robot.A1_hard
        b1_hard.value = robot.b1_hard
        u1_ref.value = robot.find_u_nominal(x_diff=x_diff,U_max=U_max,tol=tol)
        u_ref_list[:,i] = u1_ref.value.reshape(2,)

        constrained_controller.solve(solver=cp.GUROBI, reoptimize=True)
        if constrained_controller.status != "optimal":
            robot.A1_hard[0,:] = np.zeros((1,2))
            robot.b1_hard[0] = 0
        
            #robot.A1_hard[1,:] = np.zeros((1,2))
            #robot.b1_hard[1] = 0
            print("here")
            A2_hard.value = robot.A1_hard
            b2_hard.value = robot.b1_hard
            A2_soft.value = robot.A1_soft
            b2_soft.value = robot.b1_soft

            u2_ref.value = u1_ref.value
            relaxed_controller.solve(solver=cp.GUROBI, reoptimize=True)
            robot.nextU = u2.value + u_d.value
        else:
            robot.nextU = u1.value + u_d.value

        u_list[:,i] = robot.nextU.reshape(2,)
        robot.step(robot.nextU)
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
plt.plot(tp, x_list[0,:])
plt.plot(tp, x_target_list[0,:])
plt.legend(['x', 'x_target'])
plt.show()
