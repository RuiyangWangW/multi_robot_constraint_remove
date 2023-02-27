import numpy as np
from scipy.stats import norm
import cvxpy as cp
from robot_models.SingleIntegrator2D import *


def discretize_alpha_forward_cal(x0):
    #Define Constants
    dt = 0.1
    U_max = 2.0

    #Define Disturbance
    disturbance = True
    mean = 0.0
    std = 2.0
    disturb_max = -6.0 * U_max

    #Define Grid
    radius = 5.0
    d_max = 0.2
    y_max = 6.0
    y_min = -2.0
    x_min = -6.0
    step = 0.05

    #Define Alpha
    alpha_step = 0.5
    alpha_list = np.arange(start=0,stop=10.0+alpha_step,step=alpha_step)
    betta = 0.8

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

    robot = SingleIntegrator2D(x0, dt, ax=None, id = 0, color='r',palpha=1.0, num_constraints_hard = num_constraints_hard1, num_constraints_soft = num_constraints_soft1, plot=False)
    forward_set = np.array([])
    x0_key = str(int((x0[0]-x_min)/step))+","+str(int((x0[1]-y_min)/step))
    if disturbance:
        y_disturb = norm.pdf(x0, loc=mean, scale=std)[0] * disturb_max
    else:
        y_disturb = 0.0
    u_disturb = np.array([0.0, y_disturb]).reshape(2,1)

    has_been_added = {}
    ulist = np.array([])
    for i in range(alpha_list.shape[0]):
        robot.X = x0.reshape(-1,1)
        alpha = alpha_list[i]
        x_r = np.zeros((2,1))
        dtheta = 0.1
        intercept_theta = np.arctan2(robot.X[1],robot.X[0])
        x_r_theta = intercept_theta+dtheta
        x_r[0] = radius*np.cos(x_r_theta)
        x_r[1] = radius*np.sin(x_r_theta)
        if x_r[0] < -5:
            x_r[0] = -5
        if x_r[1] < 0:
            x_r[1] = 0
        v, dv_dx = robot.lyapunov(x_r) 
        robot.A1_soft[0,:] = dv_dx@robot.g()
        robot.b1_soft[0] = -dv_dx@robot.f() - betta*v - dv_dx@robot.g()@u_disturb
        
        h1, dh1_dx = robot.static_safe_set(np.zeros((2,1)),radius+d_max)    
        robot.A1_hard[0,:] = -dh1_dx@robot.g()
        robot.b1_hard[0] = dh1_dx@robot.f() + alpha*h1 + dh1_dx@robot.g()@u_disturb

        h2, dh2_dx = robot.static_safe_set(np.zeros((2,1)),radius-d_max)
        h2 = -h2
        dh2_dx = -dh2_dx
        robot.A1_hard[1,:] = -dh2_dx@robot.g()
        robot.b1_hard[1] = dh2_dx@robot.f() + alpha*h2 + dh2_dx@robot.g()@u_disturb

        h3 = y_max-robot.X[1]
        robot.A1_hard[2,:] = np.array([0,1]).reshape(1,2)@robot.g()
        robot.b1_hard[2] = -np.array([0,1]).reshape(1,2)@robot.g()@u_disturb + alpha*h3 - np.array([0,1]).reshape(1,2)@robot.f()
        A1_soft.value = robot.A1_soft
        b1_soft.value = robot.b1_soft
        A1_hard.value = robot.A1_hard
        b1_hard.value = robot.b1_hard
        u1_ref.value = robot.nominal_input(x_r)
        try: 
            constrained_controller.solve(solver=cp.GUROBI, reoptimize=True)
            if constrained_controller.status!="optimal":
                continue
        except:
            continue
        u_next = u1.value + u_disturb
        robot.step(u_next)
        new_pos = robot.X
        x = new_pos[0]        
        y = new_pos[1]

        if y < 0 and (x**2+y**2)<(radius-d_max)**2 and (x**2+y**2)>(radius+d_max)**2:
            continue
        pos_key = str(int((x-x_min)/step))+","+str(int((y-y_min)/step))
        added = has_been_added.get(pos_key)
        if added:
            continue
        forward_set = np.append(forward_set,np.array([pos_key]),axis=0)
        has_been_added.update({pos_key: True})
        if np.size(ulist) == 0:
            ulist = np.array([u1.value]).reshape(-1,1) 
        else:
            ulist = np.append(ulist,np.array([u1.value]).reshape(-1,1),axis=1)
    return x0_key, forward_set, ulist

def discretize_u_forward_cal(x0):
    #Define Constants
    dt = 0.1
    U_max = 2.0

    #Define Disturbance
    disturbance = True
    mean = 0.0
    std = 2.0
    disturb_max = -6.0 * U_max

    #Define Grid
    y_max = 6.0
    y_min = -2.0
    x_min = -6.0
    x_max = 6

    step = 0.05


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

    robot = SingleIntegrator2D(x0, dt, ax=None, id = 0, color='r',palpha=1.0, num_constraints_hard = 0, num_constraints_soft = 0, plot=False)
    forward_set = np.array([])
    ulist = np.array([])
    if disturbance:
        y_disturb = norm.pdf(x0, loc=mean, scale=std)[0] * disturb_max
        u_disturb = np.array([0.0, y_disturb]).reshape(2,1)
    
    x0_key = str(int((x0[0]-x_min)/step))+","+str(int((x0[1]-y_min)/step))
    has_been_added = {}
    for i in range(u2d_list.shape[0]):
        robot.X = x0.reshape(-1,1)
        u = u2d_list[i,:].reshape(2,1)
        if disturbance:
            u_next = u + u_disturb
        else:
            u_next = u
        robot.nextU = u_next
        robot.step(robot.nextU)
        new_pos = robot.X
        x = new_pos[0]
        y = new_pos[1]
        if y>y_max or y<y_min or x>x_max or x<x_min:
            continue
        pos_key = str(int((x-x_min)/step))+","+str(int((y-y_min)/step))
        added = has_been_added.get(pos_key)
        if added:
            continue
        forward_set = np.append(forward_set,np.array([pos_key]),axis=0)
        if np.size(ulist) == 0:
            ulist = np.array([u]).reshape(-1,1) 
        else:
            ulist = np.append(ulist,np.array([u]).reshape(-1,1),axis=1)
        has_been_added.update({pos_key: True})

    return x0_key, forward_set, ulist