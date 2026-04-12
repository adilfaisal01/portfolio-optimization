import numpy as np
from numpy._core.shape_base import block
import osqp
from scipy import sparse
from scipy.sparse import block_diag

n=7 #number of assets
N=30 #how many days am I looking ahead
state_dim=n+1 # how many states I have, assets plus cash (hence n assets, plus 1 wealth tracker)
w_max=0.20 # maximum weight in a single asset

#system dynamics setup
I=np.identity(n)
x_0=[np.log(10000)]

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
print(f'Qshape:{q_total.shape}')

# constrain handling
leftm_s=np.zeros((n,1))
rightm_s=np.eye(n)
m_s=np.hstack((leftm_s,rightm_s))

M=block_diag([m_s for _ in range(N)],format='csc').toarray() #0<w_i<0.20
print(M.shape)
# print(M)

l_wi=0-M@T_bar@x_0
u_wi=w_max-M@T_bar@x_0

# sum constraints, sum(w_i)=1
m_row=np.hstack((0, np.ones(n)))
M_sum=block_diag([m_row.reshape(1,-1) for _ in range(N)], format='csc').toarray()
print(M_sum.shape)
l_ws=1-M_sum@T_bar@x_0
u_ws=l_ws

Aineq=sparse.csc_matrix(M@S_bar)
Aeq=sparse.csc_matrix(M_sum@S_bar)
A_cons=sparse.vstack([Aineq,Aeq],format='csc')
l_cons=np.hstack([l_wi.flatten(),l_ws.flatten()])
u_cons=np.hstack([u_wi.flatten(),u_ws.flatten()])


prob=osqp.OSQP()
prob.setup(sparse.csc_matrix(H),q_total.flatten(),A_cons,l_cons,u_cons,warm_starting=True)

# solve the osqp problem
res = prob.solve()

# Check if it worked
if res.info.status == 'solved':
    # z is our plan: [u0, u1, ... u29]
    z_optimal = res.x
    # We only execute the first move (MPC principle)
    u_today = z_optimal[:n] 
    print("Optimal trades for today:", u_today)
    # print(np.sum(u_today))
else:
    print("Solver failed to find a solution!")
    
# moving through the dynamics
print(u_today.shape)
print(x_0.shape)
u_today=u_today.reshape(n,1)
w_updated=x_0[1:n+1]+u_today
o=np.array([0.50]*2+[-0.01]*3+[0.10]*2).reshape(n,1)
print(o.shape)
wealth_next_log=x_0[0]+w_updated.T@o
print(wealth_next_log.item())

weight_update=w_updated*np.exp(o)
print(weight_update/np.sum(weight_update))