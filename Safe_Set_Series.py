import numpy as np


class Safe_Set_Series2D:
    
    def __init__(self,centroids,radii,alpha_list):
        self.centroids = centroids
        self.radii = radii
        self.alpha_list = alpha_list
        self.id = 0
        self.num_sets = len(centroids)

    def evaluate_h(self, x_diff):
        h = np.linalg.norm(x_diff)-self.radii[self.id]
        return h

    def safe_set_constraints(self,robot_pos,u_d,robot_f,robot_g):
        x_diff = (robot_pos-self.centroids[:,self.id].reshape(2,1)).T
        A = 2*x_diff@robot_g
        h = self.evaluate_h(x_diff)
        b = -self.alpha_list[self.id]*h-2*x_diff@(robot_f+robot_g@u_d.value)
        return A, b
    
    def sets_reached(self,robot_pos):
        x_diff = (robot_pos-self.centroids[:,self.id].reshape(2,1)).T
        h = self.evaluate_h(x_diff)
        if h < 0:
            return True
        else:
            return False
    
    def update_targets(self, robot_pos, failed):
        if self.sets_reached(robot_pos) or failed:
            if self.id < self.num_sets:
                self.id += 1
    
        

