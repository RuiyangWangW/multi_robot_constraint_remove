import numpy as np

class Trajectory2D:
    def __init__(self, trajectory_points, tot_time, poly_degree):
        self.type = 'Trajectory2D'
        self.trajectory_points = trajectory_points
        self.num_points = trajectory_points.shape[0]
        self.time_step = tot_time/float(self.num_points-1)
        self.time_list = np.arange(start=0,stop=tot_time+self.time_step,step=self.time_step)
        self.poly_fit_x = np.polyfit(self.time_list, self.trajectory_points[:,0], deg=poly_degree)
        self.poly_fit_y = np.polyfit(self.time_list, self.trajectory_points[:,1], deg=poly_degree)
        self.tot_time = tot_time


    def get_current_target(self, time):
        target = np.zeros((2,1))
        target[0] = np.polyval(self.poly_fit_x, time)
        target[1] = np.polyval(self.poly_fit_y, time)
        if(time>self.tot_time):
            target = self.trajectory_points[-1,:]
        return target.reshape(2,1)

    def x_r_dot(self, time):
        dx_r_dt = np.zeros((2,1))
        poly_dx_dt = np.polyder(self.poly_fit_x)
        dx_r_dt[0] = np.polyval(poly_dx_dt, time)
        poly_dy_dt = np.polyder(self.poly_fit_y)
        dx_r_dt[1] = np.polyval(poly_dy_dt, time)
        return dx_r_dt.reshape(2,1)