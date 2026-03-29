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

T_list=[A_list[0]]



    




    

