import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint
from scipy.linalg import toeplitz
import torch
import scipy

#Ref: 1D Burgers' equation, finite volume, upwind scheme
#Ref link: https://github.com/gilbertfrancois/partial-differential-equations?tab=readme-ov-file

def dudt_upwind(u_bar, t, dx):
    
    n = u_bar.shape[0]
    
    # u_bar denotes state in cell centers (in the formulas written as \bar{u}_i)
    # f_bar denotes f(u_bar) in cell centers, using Burgers' equation f(u) = u^2/2
    f_bar = u_bar**2 / 2
    
    # Flux at the cell boundaries from f_{0-1/2} to f_{n+1/2}
    f_interface = np.zeros(shape=(n+1,))
    assert f_interface.shape[0] == u_bar.shape[0] + 1
    
    # Shock speed at the cell boundaries from f_{0-1/2} to f_{n+1/2}, 
    # setting the outer boundaries of the domain to s=0, because they are unused.
    s = np.zeros_like(f_interface)
    s[1:n] = (u_bar[0:n-1] + u_bar[1:n]) / 2
    
    # left boundary condition: f_{0-1/2}=0
    f_interface[0] = 0
    
    for i in range(1, n):
        if s[i] > 0:
            f_interface[i] = f_bar[i-1]
        else:
             f_interface[i] = f_bar[i]
    
    # right boundary condition: f_{n+1/2}=0
    f_interface[n] = 0
    
    # Compute the time derivative as the difference of flux directed into the cell.
    dudt = (f_interface[0:n] - f_interface[1:n+1])/dx
    return dudt

def Burger(n, timestep):
    # number of cells in 1D space
    # n = 1024#200

    # space
    x0 = 0
    xn = 1
    dx = (xn - x0)/n
    x_interface = np.linspace(x0, xn, n+1)
    x = x_interface[0:n] + dx/2

    # time
    t0 = 0
    tn = .5
    t_steps = timestep#200
    t = np.linspace(t0, tn, t_steps)

    # Initial condition
    # f_base = 20
    # u_init = np.sin(2*np.pi*f_base*x*np.sin(2*np.pi*x))
    u_init = NSGRF(n,2)

    #Plot inital condition
    # plt.figure(figsize=(7,7))
    # plt.plot(x, u_init, label="$\bar{u}$")
    # plt.xlabel("$x$")
    # plt.ylabel("$u$")
    # plt.title("Initial condition of $u$")
    # plt.show()
    # plt.savefig("FRFT/Output/BurgerIC_nonstationary.png")
    # plt.clf()

    #integral from initial condition
    u = odeint(dudt_upwind, u_init, t, args=(dx,))

    #Plot u(x,t)
    # fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    # # axs[0].set_ylim(min(U0, Un),max(U0,Un))
    # axs[0].plot(x, u[0], label=f"t={0/t_steps*(tn-t0):.3f}")
    # axs[0].plot(x, u[49], label=f"t={50/t_steps*(tn-t0):.3f}")
    # axs[0].plot(x, u[99], label=f"t={100/t_steps*(tn-t0):.3f}")
    # axs[0].plot(x, u[149], label=f"t={150/t_steps*(tn-t0):.3f}")
    # axs[0].plot(x, u[199], label=f"t={200/t_steps*(tn-t0):.3f}")
    # axs[0].legend()
    # axs[0].set_ylabel("u")
    # axs[0].set_xlabel("x")
    # axs[1].imshow(u, aspect="auto", cmap="inferno", origin="lower", extent=[0, 1, t0, tn])
    # axs[1].set_xlabel(f"x")
    # axs[1].set_ylabel("time")
    # plt.show()    
    # plt.savefig("FRFT/Output/Burger_uxt_nonstationary.png")
    # plt.clf()
    return torch.from_numpy(u_init), torch.from_numpy(u)

def Burger_datagen():
    trainsize = 30
    xnum = 1024
    tnum = 1000
    a = torch.zeros(trainsize,xnum)
    u = torch.zeros(trainsize,tnum,xnum)
    for i in range(trainsize):
        a[i,:], u[i,:,:] = Burger(xnum, tnum)
    scipy.io.savemat('/data/suwei/burger1D/Burgernorm2_trainsize30_x1024_t1000.mat', mdict={'a': a.cpu().numpy(), 'u': u.cpu().numpy()})
    return

def Burger_dataread():
    data = scipy.io.loadmat('/data/suwei/burger1D/Burgernorm2_trainsize200_x1024_t100.mat')
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
    # u, Q = precision_matrix(numx, numy)
    # ns = np.random.multivariate_normal(mean = u, cov = Q)
    # norm_ns = 2*(ns[:]-ns.min())/(ns.max()-ns.min())-1 #normalize for burger
    # return norm_ns[:numx]#only get one dimention sample
    # return ns[:numx]#only get one dimention sample

    # suwei: norm is not tested it yet.
    u, Q = precision_matrix(numx, numy)
    normQ = Q/np.linalg.norm(Q)
    ns = np.random.multivariate_normal(mean = u, cov = normQ)    
    return ns[::numy]
    # suwei: norm is not tested it yet.

if __name__ == "__main__":
    # u0 = NSGRF(125,125)
    # Burger(1024,1000)
    Burger_datagen()
    print("haha")