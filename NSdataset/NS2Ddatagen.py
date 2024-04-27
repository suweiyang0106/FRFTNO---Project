import numpy as np
import torch
from scipy import signal
import matplotlib.pyplot as plt
import scipy


#Ref: 2D Poisson equation, finite difference.ipynb
#Ref link: https://github.com/gilbertfrancois/partial-differential-equations?tab=readme-ov-file
#Ref(nonstationary): Exploring a New Class of Non-stationary Spatial Gaussian Random Fields with Varying Local Anisotropy

#2D poission data gent
def poisson_plot():
    # nx, ny denotes the number of nodes, including the boundaries.
    # Set these values to 5, to follow the written text above. However, you can refine the resolution by setting e.g. nx = xy = 128.
    # Also try asymmetric discretization, e.g. nx = 64, ny = 32.
    nx = 128
    ny = 128
    dx = 1/(nx-1)
    dy = 1/(ny-1)
    print(f"dx={dx:.3f}, dy={dy:.3f}")
    # Create block for one row i, all columns j.
    # u_{i,j}
    diag_block = np.eye(nx-2)*(-2/dx**2 + -2/dy**2)
    # u_{i+1, j}
    diag_block = diag_block + np.diag(np.ones(shape=(nx-3,))*1/dx**2, 1)
    # u_{i-1,j}
    diag_block = diag_block + np.diag(np.ones(shape=(nx-3,))*1/dx**2, -1)
    # Create blocks for all rows
    A = np.kron(np.eye(ny-2), diag_block)
    # u_{i, j+1}
    A = A + np.diag(np.ones((nx-2)*(ny-3),)*1/dy**2, nx-2)
    # u_{i, j-1}
    A = A + np.diag(np.ones((nx-2)*(ny-3),)*1/dy**2, -(nx-2))
    print(A)
    print(f"\nA shape (rows, cols): {A.shape}")
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    xx, yy = np.meshgrid(x, y)
    def fn(x, y):
        fbase=20
        return -np.sin(x*x*fbase)+np.cos(y*y)
    # Compute the values of F(x,y) for all nodes, including boundary nodes.
    # F = fn(xx, yy)
    F = NSGRF(nx,ny)

    # Take the inner nodes of F(x,y) for solving the PDE.
    b = -F[1:-1,1:-1]
    # Reshape b into a 1D vector to match the dimensions of matrix A 
    b = b.reshape(-1)
    # Solve the PDE
    u = np.linalg.solve(A, b)    
    # Reshape the solution u to 2D to match the inner nodes of Omega and add boundary values to u.
    u = u.reshape(ny-2, nx-2) 

    # Add boundary values to solution u, (Omega_b = 0)
    ub = np.zeros(shape=(ny, nx))
    ub[1:ny-1,1:nx-1] = u
    u = ub
    # 3D plots

    fig, ax = plt.subplots(1, 2, subplot_kw={"projection": "3d"}, figsize=(14, 7))
    ax[0].plot_surface(xx, yy, F, cmap="inferno", linewidth=1, antialiased=True)
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    ax[0].view_init(20, -60)
    ax[0].set_xlim(0, 1)
    ax[0].set_ylim(0, 1)

    ax[0].set_zlim(-1, 0)
    ax[0].set_title("$F(x, y)$")
    ax[0].set_xlabel("X")
    ax[0].set_ylabel("Y")

    ax[1].plot_surface(xx, yy, u, cmap="inferno", linewidth=1, antialiased=True)
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    ax[1].view_init(20, -60)
    ax[1].set_xlim(0, 1)
    ax[1].set_ylim(0, 1)
    ax[1].set_title("$u(x, y)$")
    ax[1].set_xlabel("X")
    ax[1].set_ylabel("Y")

    plt.show()
    plt.savefig("FRFT/Output/Poisson3D.png")
    plt.clf()
# 2D plots

    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    axs[0].imshow(F, cmap="inferno", origin="lower", aspect="auto", extent=[0, 1, 0, 1])
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].set_title("$F(x,y)$")
    axs[1].imshow(ub, cmap="inferno", origin="lower", aspect="auto", extent=[0, 1, 0, 1])
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")
    axs[1].set_title("$u(x,y)$")
    plt.show()
    plt.savefig("FRFT/Output/Poisson2D.png")
    return


def Poision2Dgen(numx,numy):
    # nx, ny denotes the number of nodes, including the boundaries.
    # Set these values to 5, to follow the written text above. However, you can refine the resolution by setting e.g. nx = xy = 128.
    # Also try asymmetric discretization, e.g. nx = 64, ny = 32.
    nx = numx
    ny = numy
    dx = 1/(nx-1)
    dy = 1/(ny-1)
    # Create block for one row i, all columns j.
    # u_{i,j}
    diag_block = np.eye(nx-2)*(-2/dx**2 + -2/dy**2)
    # u_{i+1, j}
    diag_block = diag_block + np.diag(np.ones(shape=(nx-3,))*1/dx**2, 1)
    # u_{i-1,j}
    diag_block = diag_block + np.diag(np.ones(shape=(nx-3,))*1/dx**2, -1)
    # Create blocks for all rows
    A = np.kron(np.eye(ny-2), diag_block)
    # u_{i, j+1}
    A = A + np.diag(np.ones((nx-2)*(ny-3),)*1/dy**2, nx-2)
    # u_{i, j-1}
    A = A + np.diag(np.ones((nx-2)*(ny-3),)*1/dy**2, -(nx-2))
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    xx, yy = np.meshgrid(x, y)
    def fn(x, y):
        fbase=50
        return -np.sin(x*x*fbase)+np.cos(y*y)
    # Compute the values of F(x,y) for all nodes, including boundary nodes.
    # F = fn(xx, yy)
    F = NSGRF(numx, numy)

    # Take the inner nodes of F(x,y) for solving the PDE.
    b = -F[1:-1,1:-1]
    # Reshape b into a 1D vector to match the dimensions of matrix A 
    b = b.reshape(-1)
    # Solve the PDE
    u = np.linalg.solve(A, b)    
    # Reshape the solution u to 2D to match the inner nodes of Omega and add boundary values to u.
    u = u.reshape(ny-2, nx-2) 

    # Add boundary values to solution u, (Omega_b = 0)
    ub = np.zeros(shape=(ny, nx))
    ub[1:ny-1,1:nx-1] = u
    u = ub    
    return torch.from_numpy(F), torch.from_numpy(u)    

def datagen():
    numx, numy = 64, 64
    trainsize = 200
    f_train = torch.zeros(trainsize,numx,numy)
    u_train = torch.zeros(trainsize,numx,numy)
    
    for idx in range(trainsize):
        f_train[idx,:,:], u_train[idx,:,:] = Poision2Dgen(numx, numy)
    scipy.io.savemat('/data/suwei/poisson2D/poissonnorm_trainsize200_64x64.mat', mdict={'a': f_train.cpu().numpy(), 'u': u_train.cpu().numpy()})    
    return f_train, u_train

def dataread():
    data = scipy.io.loadmat('/data/suwei/poisson2D/poissonnorm_trainsize200_64x64.mat')
    return torch.from_numpy(data['a']), torch.from_numpy(data['u'])    

#Nonstationary Gaussian random field generate

def H_s(s1,s2):
    def vs(s1,s2):
        #vector field, the place creates nonstationary property
        v1 = 1.2*torch.sin(torch.tensor(2*torch.pi*10*s2*s2)*torch.cos(torch.tensor(s1*s2)))#1.2*torch.sin(torch.tensor(2*torch.pi*10*s2*s2)*torch.cos(torch.tensor(s1*s1)))
        v2 = 1.4*torch.sin(torch.tensor(2*torch.pi*5*s2*s2))+torch.cos(torch.tensor(s1*s2))#1.4*torch.sin(torch.tensor(2*torch.pi*5*s2*s2))+torch.cos(torch.tensor(s1*s2))
        v = torch.tensor([v1,v2])       
        return v
        
    #H(s) = r*I2 + B*V(s)V(s)
    r, B = 1, 100
    I, V = torch.eye(2), vs(s1,s2)
    H = r*I +B*V.reshape(2,1) @ V.reshape(1,2)
    return H

def A_H(M,N,hx,hy):
    ah = torch.eye(M*N)
    #fill uij in each row jM+i
    for j in range(N):
        for i in range(M):
            # A cell, Eij, is limited at [i*hx+(i+1)*hx]X[j*hy+(j+1)*hy]
            centroid_i, centroid_j = i*hx+0.5*hx, j*hy+0.5*hy #sij
            b_nb_i, b_nb_j = centroid_i , (centroid_j-0.5*hy)                                   #bottom in cell
            t_nb_i, t_nb_j = centroid_i , (centroid_j+0.5*hy) if (centroid_j+0.5*hy) < N else 0 #top in cell
            r_nb_i, r_nb_j = (centroid_i+0.5*hx) if(centroid_i+0.5*hx) < M else 0 , centroid_j  #right in cell
            l_nb_i, l_nb_j = (centroid_i-0.5*hx) , centroid_j                                   #left in cell

            #Start filling ah matrix
            h11 = H_s(r_nb_i,r_nb_j)[0,0]+H_s(l_nb_i,l_nb_j)[0,0]
            h22 = H_s(t_nb_i,t_nb_j)[1,1]+H_s(b_nb_i,b_nb_j)[1,1]
            ah[j*M+i,j*M+i]=-hy/hx * h11 -hx/hy*h22                                             #fill diag elements in ah, aka uij itself
            
            l_ah_i, l_ah_j = j*M+i, j*M+(i-1) #if(j*M+(i-1)) >=0 else M*N-1                      #left point in AH matrix
            r_ah_i, r_ah_j = j*M+i, j*M+(i+1) if(j*M+(i+1)) <= M*N-1 else 0                     #right point in AH matrix
            t_ah_i, t_ah_j = j*M+i, (j+1)*M+i if((j+1)*M+i) <= M*N-1 else (i+1)                 #above point in AH matrix
            b_ah_i, b_ah_j = j*M+i, (j-1)*M+i #if((j-1)*M+i) >=0 else (-M+i)                     #below point in AH matrix
            lb_ah_i, lb_ah_j = j*M+i, (j-1)*M+(i-1) #if(j-1)*M+(i-1) >= 0 else -M+i              #left-below point in AH matrix
            rb_ah_i, rb_ah_j = j*M+i, (j-1)*M+(i+1) #if(j-1)*M+(i+1) >= 0 else -M+i              #right-below point in AH matrix
            lt_ah_i, lt_ah_j = j*M+i, (j+1)*M+(i-1) if(j+1)*M+(i-1) <= M*N-1 else i             #left-top point in AH matrix
            rt_ah_i, rt_ah_j = j*M+i, (j+1)*M+(i+1) if(j+1)*M+(i+1) <= M*N-1 else (i+2)         #right-top point in AH matrix

            h11 = H_s(l_nb_i,l_nb_j)[0,0] 
            h12 = H_s(t_nb_i,t_nb_j)[0,1] - H_s(b_nb_i,b_nb_j)[0,1]
            ah[l_ah_i,l_ah_j] = hy/hx*h11 - 0.25*h12
            h11 = H_s(r_nb_i,r_nb_j)[0,0]
            h12 = H_s(t_nb_i,t_nb_j)[0,1] - H_s(b_nb_i,b_nb_j)[0,1]
            ah[r_ah_i,r_ah_j] = hy/hx*h11 + 0.25*h12
            h22 = H_s(t_nb_i,t_nb_j)[1,1]
            h21 = H_s(r_nb_i,r_nb_j)[1,0] - H_s(l_nb_i,l_nb_j)[1,0]
            ah[t_ah_i,t_ah_j] = hx/hy*h22 + 0.25*h21
            h22 = H_s(b_nb_i,b_nb_j)[1,1]
            h21 = H_s(r_nb_i,r_nb_j)[1,0] - H_s(l_nb_i,l_nb_j)[1,0]
            ah[b_ah_i,b_ah_j] = hx/hy*h22 - 0.25*h21
            h12 , h21 = H_s(b_nb_i,b_nb_j)[0,1] , H_s(l_nb_i,l_nb_j)[1,0]
            ah[lb_ah_i,lb_ah_j] = 0.25*(h12+h21)
            h12 , h21 = H_s(b_nb_i,b_nb_j)[0,1] , H_s(r_nb_i,r_nb_j)[1,0]
            ah[rb_ah_i,rb_ah_j] = -0.25*(h12+h21)
            h12 , h21 = H_s(t_nb_i,t_nb_j)[0,1] , H_s(l_nb_i,l_nb_j)[1,0]
            ah[lt_ah_i,lt_ah_j] = -0.25*(h12+h21)
            h12 , h21 = H_s(t_nb_i,t_nb_j)[0,1] , H_s(r_nb_i,r_nb_j)[1,0]
            ah[rt_ah_i,rt_ah_j] = 0.25*(h12+h21)


    return ah

def precision_matrix(numx, numy):
    #Set grid range [0,A]X[0,B] where A=1, B=1
    #Set cell numbers where M = 200, N = 200
    A, B = 1, 1
    M, N = numx, numy
    hx, hy = A/M, B/N
    Dv = hx*hy*torch.eye(M*N)
    Dk = 10 * torch.eye(M*N) # here we set k^2 = 1
    ah = A_H(M,N,hx,hy)
    A = torch.matmul(Dv,Dk)-ah
    Q1 =torch.matmul(A,Dv)
    Q = torch.matmul(Q1,torch.transpose(A, 0, 1))
    return torch.zeros(M*N) , Q

def NSGRF(numx, numy):
    u, Q = precision_matrix(numx, numy)
    Q = Q/np.linalg.norm(Q)#suwei add not test yet
    ns = np.random.multivariate_normal(mean = u, cov = Q)#suwei
    return ns.reshape(numx,numy)

if __name__ == "__main__":
    # u0 = NSGRF(128,128)
    # F, u = Poision2Dgen(64,64)
    # f, u = datagen()
    # poisson_plot()
    datagen()
    dataread()
    print("haha")