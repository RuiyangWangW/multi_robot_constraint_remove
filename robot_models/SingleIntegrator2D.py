import numpy as np

class SingleIntegrator2D:
    
    def __init__(self,X0,dt,ax,id,num_robots=1,num_adversaries = 1, num_obstacles = 0, alpha=0.8, num_constraints_hard = 1, num_constraints_soft = 1, color='r',palpha=1.0,plot=True, num_connectivity = 1, num_eigen_connectivity = 0):
        '''
        X0: iniytial state
        dt: simulation time step
        ax: plot axis handle
        id: robot id
        '''
        
        self.type = 'SingleIntegrator2D'        
        
        X0 = X0.reshape(-1,1)
        self.X = X0
        self.dt = dt
        self.id = id
        self.color = color
        self.palpha = palpha

        self.U = np.array([0,0]).reshape(-1,1)
        self.U_nominal = np.array([0,0]).reshape(-1,1)
        self.nextU = self.U

        # Plot handles
        self.plot = plot
        if self.plot:
            self.body = ax.scatter([],[],c=color,alpha=palpha,s=10)
            self.render_plot()
        
        """
         # for Trust computation
        self.eigen_alpha = eigen_alpha
        self.obs_alpha =  alpha*np.ones((1,num_obstacles))#
        self.robot_alpha = alpha*np.ones((1,num_robots))
        self.robot_h = np.ones((1,num_robots))
        self.obs_h = np.ones((1,num_obstacles))
        self.robot_connectivity_objective = 0
        self.robot_connectivity_alpha = alpha*np.ones((1,1)) 
        """

        #num_constraints1  = num_robots - 1 + num_obstacles + num_connectivity + num_eigen_connectivity
        
        self.A1_hard = np.zeros((num_constraints_hard,2))
        self.b1_hard = np.zeros((num_constraints_hard,1))
        self.A1_soft = np.zeros((num_constraints_soft,2))
        self.b1_soft = np.zeros((num_constraints_soft,1))
        self.slack_constraint = np.zeros((num_constraints_soft,1))
        
        self.Xs = X0.reshape(-1,1)
        self.Us = np.array([0,0]).reshape(-1,1)        
        
    def f(self):
        return np.array([0,0]).reshape(-1,1)
    
    def g(self):
        return np.array([ [1, 0],[0, 1] ])
        
    def step(self,U): #Just holonomic X,T acceleration

        self.U = U.reshape(-1,1)
        self.X = self.X + ( self.f() + self.g() @ self.U )*self.dt
        self.render_plot()
        self.Xs = np.append(self.Xs,self.X,axis=1)
        self.Us = np.append(self.Us,self.U,axis=1)
        return self.X

    def render_plot(self):
        if self.plot:
            x = np.array([self.X[0,0],self.X[1,0]])

            # scatter plot update
            self.body.set_offsets([x[0],x[1]])

    def lyapunov(self, G):
        V = np.linalg.norm( self.X - G[0:2] )**2
        dV_dx = 2*( self.X - G[0:2] ).T
        return V, dV_dx
    
    def nominal_input(self,G):
        V, dV_dx = self.lyapunov(G.X)
        return - 5.0 * dV_dx.reshape(-1,1)
    
    def agent_barrier(self,agent,d_min):
        h = d_min**2 - np.linalg.norm(self.X - agent.X[0:2])**2
        dh_dxi = -2*( self.X - agent.X[0:2] ).T
        
        if agent.type=='SingleIntegrator2D':
            dh_dxj = 2*( self.X - agent.X[0:2] ).T
        elif agent.type=='Unicycle':
            dh_dxj = np.append( -2*( self.X - agent.X[0:2] ).T, [[0]], axis=1 )
        else:
            dh_dxj = 2*( self.X - agent.X[0:2] ).T
        return h, dh_dxi, dh_dxj
    
    def connectivity_barrier( self, agent, d_max ):
        h = np.linalg.norm( self.X[0:2] - agent.X[0:2] )**2 - d_max**2
        
        dh_dxi = 2*( self.X[0:2] - agent.X[0:2] ).T
        if agent.type == 'SingleIntegrator2D':
            dh_dxj = -2*( self.X[0:2] - agent.X[0:2] ).T
        elif agent.type == 'Unicycle':
            dh_dxj = np.append( -2*( self.X[0:2] - agent.X[0:2] ).T, np.array([[0]]), axis = 1 )
        else:
            dh_dxj = -2*( self.X[0:2] - agent.X[0:2] ).T
        return h.reshape(-1,1), dh_dxi, dh_dxj        
    
    def find_u_nominal(self,x_diff,U_max,tol):
        theta = np.arctan2(x_diff[1],x_diff[0])
        u_nominal = np.zeros((2,1))
        x_diff = -x_diff
        if ((abs(x_diff[0])>tol) and (abs(x_diff[1]>tol))):
            u_nominal = np.array([U_max*np.cos(theta),U_max*np.sin(theta)])
        elif ((abs(x_diff[0])<tol) and (abs(x_diff[1]>tol))):
            if(abs(x_diff[1])>U_max):
                if x_diff[1] > 0:
                    u_nominal = np.array([0,U_max])
                else:
                    u_nominal = np.array([0,-U_max])
            else:
                u_nominal = np.array([0,x_diff[1]])
        elif ((abs(x_diff[0])>tol) and (abs(x_diff[1]<tol))):
            if(abs(x_diff[0])>U_max):
                if x_diff[0] > 0:
                    u_nominal = np.array([U_max,0])
                else:
                    u_nominal = np.array([-U_max,0])
            else:
                u_nominal = np.array([x_diff[0],0])

        return u_nominal.reshape(2,1)