import numpy as np
from numpy._core.shape_base import block
import osqp
from scipy.sparse import block_diag

n=2 #number of assets
N=3 #how many days am I looking ahead
state_dim=n+1 # how many states I have, assets plus cash (hence n assets, plus 1 wealth tracker)

#system dynamics setup
I=np.identity(n)
x_0=[10000]

for w in range(0,n):
    x_0.append(1/n)

x_0=np.array(x_0).reshape(-1,1) # initial conditions, initial wealth and weights, assume equal distribution

# constructing according to MPC formulation, x_{k+1}=Ax_k+Bu_k
B_top=np.zeros(n)
B_bottom=I
B=np.vstack((B_top,B_bottom)) #input matrix
assert B.shape==(n+1,n)

#constructing the state dynamics
r_hat=np.random.rand(N,n)
r_hat_cov=np.cov(r_hat,rowvar=False)
# A_top_row=np.hstack((np.ones((1,1)),r_hat.T))
# A_bottom_row=np.hstack((np.zeros((n,1)),I))
# A=np.vstack((A_top_row,A_bottom_row)) #state dynamics matrix, A
# assert A.shape==(n+1,n+1)

A_list=[]

for i in range(0,N):
     A_top_row=np.hstack((np.ones((1,1)),r_hat[i].reshape(1,-1)))
     A_bottom_row=np.hstack((np.zeros((n,1)),I))
     A=np.vstack((A_top_row,A_bottom_row))
     assert A.shape==(n+1,n+1)
     A_list.append(A)
     
T_list=[]
current_A=np.eye(n+1)

for a in A_list:
    current_A=a@current_A
    T_list.append(current_A)

T_bar=np.vstack(T_list) ##Tx_0

S_bar=np.zeros((N*(n+1),N*n))
for j in range(N):
    current_impact=B
    S_bar[j*(n+1):(j+1)*(n+1), j*n:(j+1)*(n)]=current_impact
    for i in range(j+1,N):
        current_impact=A_list[i]@current_impact
        S_bar[i*(n+1) : (i+1)*(n+1), j*n : (j+1)*n] = current_impact


print(f'Sbar shape: {S_bar.shape}')
print(f'T-bar shape:{T_bar.shape}')

Q_top=np.zeros((1,n+1))
Q_bottom=np.hstack((np.zeros((n,1)),r_hat_cov))
Q=np.vstack((Q_top,Q_bottom))

print(Q.shape)    
assert Q.shape==(n+1,n+1) #making the cost matrices from the covariances of the returns
# print(f'Q={Q}')
# stacking the Q matrices over time horizon, N, since only the diagonals have the Q matrices and the rest are zero, the Q_bar is a sparse block diagonal
Q_bar=block_diag([Q for _ in range(N)],format='csc').toarray()
# print(f'Q_bar:{Q_bar}') # Q_bar is the block sparse matrix for the full cost vector

print(f'shape of Q_bar={Q_bar.shape}')

## Transaction costs, R, for simplicity assume identity matrix
R=np.eye(n)
R_bar=block_diag([R for _ in range(N)],format='csc').toarray()
# print(f'R_bar:{R_bar}')

print(f'R_bar shape:{R_bar.shape}')
# finding the Nessian
H=2*(R_bar+S_bar.T@Q_bar@S_bar)

# finding the linear term

F_t=2*(T_bar.T@Q_bar@S_bar)
print(f'F shape{F_t.shape}')
q_risk=F_t.T@x_0 #risk associated term

terminal_wealth_index=(N-1)*state_dim
c=np.zeros(N*state_dim)
c[terminal_wealth_index]=1.0
q_goal=-1*(c.T@S_bar).reshape(-1,1)

q_total= q_goal+q_risk #linear term









