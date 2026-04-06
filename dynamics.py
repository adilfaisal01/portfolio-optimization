import numpy as np
import osqp

n=2 #number of assets
H=3 #how many days am I looking ahead

#system dynamics setup
I=np.identity(n)

# constructing according to MPC formulation, x_{k+1}=Ax_k+Bu_k
B_top=np.zeros(n)
B_bottom=I
B=np.vstack((B_top,B_bottom)) #input matrix
assert B.shape==(n+1,n)

#constructing the state dynamics
r_hat=np.random.rand(H,n)
r_hat_cov=np.cov(r_hat,rowvar=False)
# A_top_row=np.hstack((np.ones((1,1)),r_hat.T))
# A_bottom_row=np.hstack((np.zeros((n,1)),I))
# A=np.vstack((A_top_row,A_bottom_row)) #state dynamics matrix, A
# assert A.shape==(n+1,n+1)

A_list=[]

for i in range(0,H):
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

S_bar=np.zeros((H*(n+1),H*n))
for j in range(H):
    current_impact=B
    S_bar[j*(n+1):(j+1)*(n+1), j*n:(j+1)*(n)]=current_impact
    for i in range(j+1,H):
        current_impact=A_list[i]@current_impact
        S_bar[i*(n+1) : (i+1)*(n+1), j*n : (j+1)*n] = current_impact


# print(S_bar.shape)
# print(T_bar.shape)

Q_bar_top=np.zeros((1,n+1))
Q_bar_bottom=np.hstack((np.zeros((n,1)),r_hat_cov))
Q_bar=np.vstack((Q_bar_top,Q_bar_bottom))

print(Q_bar.shape)    
assert Q_bar.shape==(n+1,n+1) #making the cost matrices from the covariances of the returns



