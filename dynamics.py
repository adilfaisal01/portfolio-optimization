import numpy as np
import osqp
from scipy import sparse
from scipy.sparse import block_diag

class MPCPLanner:
    def __init__(self, n_assets,wmax,N_horizon,trans_cost):
        self.n=n_assets
        self.wmax=wmax
        self.N=N_horizon
        self.R_val=trans_cost
        self.state_dim=self.n+1
 # initial conditions, initial wealth and weights, assume equal distribution
    def _dynamic_matrices(self,r_hat):
        I=np.identity(self.n)
        # constructing according to MPC formulation, x_{k+1}=Ax_k+Bu_k
        B_top=np.zeros(self.n)
        B_bottom=I
        B=np.vstack((B_top,B_bottom)) #input matrix
        assert B.shape==(self.state_dim,self.n)
        A_list=[]
        for i in range(0,self.N):
            A_top_row=np.hstack((np.ones((1,1)),r_hat[i].reshape(1,-1)))
            A_bottom_row=np.hstack((np.zeros((self.n,1)),I))
            A=np.vstack((A_top_row,A_bottom_row))
            assert A.shape==(self.state_dim,self.state_dim)
            A_list.append(A)
        
        T_list=[]
        current_A=np.eye(self.n+1)
        
        for a in A_list:
            current_A=a@current_A
            T_list.append(current_A)
        
        self.T_bar=np.vstack(T_list) ##Tx_0
        
        self.S_bar=np.zeros((self.N*(self.n+1),self.N*self.n))
        for j in range(self.N):
            current_impact=B
            self.S_bar[j*(self.n+1):(j+1)*(self.n+1), j*self.n:(j+1)*(self.n)]=current_impact
            for i in range(j+1,self.N):
                current_impact=A_list[i]@current_impact
                self.S_bar[i*(self.n+1) : (i+1)*(self.n+1), j*self.n : (j+1)*self.n] = current_impact
        return self.T_bar, self.S_bar
    
    def cost_matrices(self,r_hat_cov,x_0):
        Q_top=np.zeros((1,self.n+1))
        Q_bottom=np.hstack((np.zeros((self.n,1)),r_hat_cov))
        Q=np.vstack((Q_top,Q_bottom))
        
        print(Q.shape)    
        assert Q.shape==(self.n+1,self.n+1) #making the cost matrices from the covariances of the returns
        # print(f'Q={Q}')
        # stacking the Q matrices over time horizon, N, since only the diagonals have the Q matrices and the rest are zero, the Q_bar is a sparse block diagonal
        Q_bar=block_diag([Q for _ in range(self.N)],format='csc').toarray()
        # print(f'Q_bar:{Q_bar}') # Q_bar is the block sparse matrix for the full cost vector
        
        print(f'shape of Q_bar={Q_bar.shape}')
        
        ## Transaction costs, R, for simplicity assume identity matrix
        R=np.eye(self.n)*self.R_val
        R_bar=block_diag([R for _ in range(self.N)],format='csc').toarray()
        # print(f'R_bar:{R_bar}')
        
        print(f'R_bar shape:{R_bar.shape}')
        # finding the Nessian
        H=2*(R_bar+self.S_bar.T@Q_bar@self.S_bar)
        F_t=2*(self.T_bar.T@Q_bar@self.S_bar)
        print(f'F shape{F_t.shape}')
        q_risk=F_t.T@x_0 #risk associated term
        
        terminal_wealth_index=(self.N-1)*self.state_dim
        c=np.zeros(self.N*self.state_dim)
        c[terminal_wealth_index]=1.0
        q_goal=-1*(c.T@self.S_bar).reshape(-1,1)
        
        q_total= q_goal+q_risk #linear term
        return H, q_total
        
    def constraint_definition(self,x_0):
        leftm_s=np.zeros((self.n,1))
        rightm_s=np.eye(self.n)
        m_s=np.hstack((leftm_s,rightm_s))
        
        M=block_diag([m_s for _ in range(self.N)],format='csc').toarray() #0<w_i<0.20
        print(M.shape)
        # print(M)
        
        l_wi=0-M@self.T_bar@x_0
        u_wi=self.wmax-M@self.T_bar@x_0
        
        # sum constraints, sum(w_i)=1
        m_row=np.hstack((0, np.ones(self.n)))
        M_sum=block_diag([m_row.reshape(1,-1) for _ in range(self.N)], format='csc').toarray()
        print(M_sum.shape)
        l_ws=1-M_sum@self.T_bar@x_0
        u_ws=l_ws
        
        Aineq=sparse.csc_matrix(M@self.S_bar)
        Aeq=sparse.csc_matrix(M_sum@self.S_bar)
        A_cons=sparse.vstack([Aineq,Aeq],format='csc')
        l_cons=np.hstack([l_wi.flatten(),l_ws.flatten()])
        u_cons=np.hstack([u_wi.flatten(),u_ws.flatten()])
        
        return A_cons, l_cons, u_cons
    
    def solver(self,x_0,r_hat,r_hat_cov):
        self._dynamic_matrices(r_hat)
        H,q=self.cost_matrices(r_hat_cov,x_0)
        A_cons,l_cons,u_cons=self.constraint_definition(x_0)
        
        prob=osqp.OSQP()
        prob.setup(sparse.csc_matrix(H),q.flatten(),A_cons,l_cons,u_cons,warm_starting=True)
        
        # solve the osqp problem
        res = prob.solve()
        u_today=None
        # Check if it worked
        if res.info.status == 'solved':
            # z is our plan: [u0, u1, ... u29]
            z_optimal = res.x
            # We only execute the first move (MPC principle)
            u_today = z_optimal[:self.n] 
            print("Optimal trades for today:", u_today)
            # print(np.sum(u_today))
        else:
            print("Solver failed to find a solution!")
        
        return u_today
        
        