#Ref: https://github.com/sachabinder/Burgers_equation_simulation/blob/main/Burgers_solver_SP.py
""" 
This file was built to solve numerically 1D Burgers' equation wave equation with the FFT. The equation corresponds to :

$\dfrac{\partial u}{\partial t} + \mu u\dfrac{\partial u}{\partial x} = \nu \dfrac{\partial^2 u}{\partial x^2}$
 
where
 - u represent the signal
 - x represent the position
 - t represent the time
 - nu and mu are constants to balance the non-linear and diffusion terms.

Copyright - © SACHA BINDER - 2021
"""

############## MODULES IMPORTATION ###############
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import torch
import scipy

############## SET-UP THE PROBLEM ###############
def Burger(xnum, tnum):
    mu = 1
    nu = 0.000001 #kinematic viscosity coefficient
        
    #Spatial mesh
    L_x = 10 #Range of the domain according to x [m]
    dx = 0.000001 #Infinitesimal distance
    N_x = xnum#int(L_x/dx) #Points number of the spatial mesh
    X = np.linspace(0,L_x,N_x) #Spatial array

    #Temporal mesh
    L_t = 8 #Duration of simulation [s]
    dt = 0.025  #Infinitesimal time
    N_t = tnum#int(L_t/dt) #Points number of the temporal mesh
    T = np.linspace(0,L_t,N_t) #Temporal array

    #Wave number discretization
    k = 2*np.pi*np.fft.fftfreq(N_x, d = dx)


    #Def of the initial condition    
    u0 = NSGRF(N_x,2)#np.exp(-(X-3)**2/2) #Single space variable fonction that represent the wave form at t = 0
    # viz_tools.plot_a_frame_1D(X,u0,0,L_x,0,1.2,'Initial condition')
    #PDE resolution (ODE system resolution)
    U = odeint(burg_system, u0, T, args=(k,mu,nu,),mxstep=5000, hmin=1e-50)#odeint(burg_system, u0, T, args=(k,mu,nu,), mxstep=5000).T
    
    #Plot inital condition
    # plt.figure(figsize=(7,7))
    # plt.plot(X, u0, label="$\bar{u}$")
    # plt.xlabel("$x$")
    # plt.ylabel("$u$")
    # plt.title("Initial condition of $u$")
    # plt.show()
    # plt.savefig("FRFT/Output/BurgerIC_nonstationary_nu0001.png")
    # plt.clf()
    #Plot u(x,t)
    # fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    # # axs[0].set_ylim(min(U0, Un),max(U0,Un))
    # axs[0].plot(X, U[:,0], label=f"t={0/N_t*(L_t-0):.3f}")
    # axs[0].plot(X, U[:,49], label=f"t={50/N_t*(L_t-0):.3f}")
    # axs[0].plot(X, U[:,99], label=f"t={100/N_t*(L_t-0):.3f}")
    # axs[0].plot(X, U[:,149], label=f"t={150/N_t*(L_t-0):.3f}")
    # axs[0].plot(X, U[:,199], label=f"t={200/N_t*(L_t-0):.3f}")
    # axs[0].legend()
    # axs[0].set_ylabel("u")
    # axs[0].set_xlabel("x")
    # axs[1].imshow(U, aspect="auto", cmap="inferno", origin="lower", extent=[0, 1, 0, L_t])
    # axs[1].set_xlabel(f"x")
    # axs[1].set_ylabel("time")
    # plt.show()    
    # plt.savefig("FRFT/Output/Burger_uxt_nonstationary0001.png")
    # plt.clf()    
    
    return torch.from_numpy(u0), torch.from_numpy(U)

############## EQUATION SOLVING ###############

#Definition of ODE system (PDE ---(FFT)---> ODE system)
def burg_system(u,t,k,mu,nu):
    #Spatial derivative in the Fourier domain
    u_hat = np.fft.fft(u)
    u_hat_x = 1j*k*u_hat
    u_hat_xx = -k**2*u_hat
    
    #Switching in the spatial domain
    u_x = np.fft.ifft(u_hat_x)
    u_xx = np.fft.ifft(u_hat_xx)
    
    #ODE resolution
    u_t = -mu*u*u_x + nu*u_xx
    return u_t.real

def Burger_datagen():
    trainsize = 200
    xnum = 1024
    tnum = 1000
    a = torch.zeros(trainsize,xnum)
    u = torch.zeros(trainsize,tnum,xnum)
    for i in range(trainsize):
        a[i,:], u[i,:,:] = Burger(xnum, tnum)
    scipy.io.savemat('/data/suwei/burger1D/BurgerV1e-6_trainsize200_x1024_t1000.mat', mdict={'a': a.cpu().numpy(), 'u': u.cpu().numpy()})
    return

def Burger_dataread():
    data = scipy.io.loadmat('/data/suwei/burger1D/BurgerV0001_trainsize200_x1024_t1000.mat')
    # data = scipy.io.loadmat('/data/suwei/burger1D/Burgernorm2_trainsize200_x1024_t100.mat')
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
    #Gnerate buurger with vicostiy
    Burger_datagen()
    print("haha")
