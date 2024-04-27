
import numpy as np
import torch
from scipy import signal
from scipy.fft import fftshift
import matplotlib.pyplot as plt
from torch_frft.frft_module import  frft, frft_shifted
from torch_frft.dfrft_module import dfrft
import random
import scipy
from scipy import signal
from scipy.fft import fftshift


#Ref: 2D_WAVE-EQ_variable-velocity 
#https://github.com/sachabinder/wave_equation_simulations/blob/main/2D_WAVE-EQ_variable-velocity.py
#Ref: Exploring a New Class of Non-stationary Spatial Gaussian Random Fields with Varying Local Anisotropy

def wave_plot():
    u0, u = wave_gen()
    fig, axs = plt.subplots(2, 2, figsize=(14, 7))
    axs[0,0].imshow(u0[:,:,0], cmap="inferno", origin="lower", aspect="auto", extent=[0, 1, 0, 1])
    axs[0,0].set_xlabel("x")
    axs[0,0].set_ylabel("y")
    axs[0,0].set_title("$u0$")
    axs[0,1].imshow(u0[:,:,50], cmap="inferno", origin="lower", aspect="auto", extent=[0, 1, 0, 1])
    axs[0,1].set_xlabel("x")
    axs[0,1].set_ylabel("y")
    axs[0,1].set_title("$u, t=50$")
    axs[1,0].imshow(u[:,:,0], cmap="inferno", origin="lower", aspect="auto", extent=[0, 1, 0, 1])
    axs[1,0].set_xlabel("x")
    axs[1,0].set_ylabel("y")
    axs[1,0].set_title("$u t=100$")
    axs[1,1].imshow(u[:,:,50], cmap="inferno", origin="lower", aspect="auto", extent=[0, 1, 0, 1])
    axs[1,1].set_xlabel("x")
    axs[1,1].set_ylabel("y")
    axs[1,1].set_title("$u t=200$")
    plt.show()
    plt.savefig("FRFT/Output/Wave2D.png")    
    return

def wave_gen():
    #Def of the initial condition   
    def I(x,y):
        """
        two space variables depending function 
        that represent the wave form at t = 0
        """
        return 1.1*np.sin(100*np.pi*x*x)+1.2*np.sin(100*np.pi*y*y)#0.2*np.exp(-((x-1)**2/0.1 + (y-1)**2/0.1))

    def V(x,y):
        """
        initial vertical speed of the wave
        """
        return 0.001#0
    ############## SET-UP THE PROBLEM ###############

    #Def of velocity (spatial scalar field)
    def celer(x,y):
        """
        constant velocity
        """
        return 1
    loop_exec = 1 # Processing loop execution flag

    bound_cond = 2  #Boundary cond 1 : Dirichlet, 2 : Neumann, 3 Mur

    if bound_cond not in [1,2,3]:
        loop_exec = 0
        print("Please choose a correct boundary condition")

    #Spatial mesh - i indices
    L_x = 4#5 #Range of the domain according to x [m]
    dx = 0.0625#0.05 #Infinitesimal distance in the x direction
    N_x = int(L_x/dx) #Points number of the spatial mesh in the x direction
    X = np.linspace(0,L_x,N_x+1) #Spatial array in the x direction

    #Spatial mesh - j indices
    L_y = 4#5 #Range of the domain according to y [m]
    dy = 0.0625#0.05 #Infinitesimal distance in the x direction
    N_y = int(L_y/dy) #Points number of the spatial mesh in the y direction
    Y = np.linspace(0,L_y,N_y+1) #Spatial array in the y direction

    #Temporal mesh with CFL < 1 - n indices
    L_t = 4 #Duration of simulation [s]
    dt = dt = 0.1*min(dx, dy)   #Infinitesimal time with CFL (Courant–Friedrichs–Lewy condition)
    N_t = int(L_t/dt) #Points number of the temporal mesh
    T = np.linspace(0,L_t,N_t+1) #Temporal array

    #Velocity array for calculation (finite elements)
    c = np.zeros((N_x+1,N_y+1), float)
    for i in range(0,N_x+1):
        for j in range(0,N_y+1):
            c[i,j] = celer(X[i],Y[j])    
    ############## CALCULATION CONSTANTS ###############
    Cx2 = (dt/dx)**2
    Cy2 = (dt/dy)**2 
    CFL_1 = dt/dy*c[:,0]
    CFL_2 = dt/dy*c[:,N_y]
    CFL_3 = dt/dx*c[0,:]
    CFL_4 =dt/dx*c[N_x,:]
    ############## PROCESSING LOOP ###############

    if loop_exec:
        # $\forall i \in {0,...,N_x}$
        U = np.zeros((N_x+1,N_x+1,N_t+1),float) #Tableau de stockage de la solution

        u_nm1 = np.zeros((N_x+1,N_y+1),float)   #Vector array u_{i,j}^{n-1}
        u_n = np.zeros((N_x+1,N_y+1),float)     #Vector array u_{i,j}^{n}
        u_np1 = np.zeros((N_x+1,N_y+1),float)  #Vector array u_{i,j}^{n+1}
        V_init = np.zeros((N_x+1,N_y+1),float)
        q = np.zeros((N_x+1, N_y+1), float)
        
        #init cond - at t = 0
        for i in range(0, N_x+1):
            for j in range(0, N_y+1):
                q[i,j] = c[i,j]**2
        
        # for i in range(0, N_x+1):
        #     for j in range(0, N_y+1):
        #         u_n[i,j] = I(X[i],Y[j])
        u_n = NSGRF()
                
        for i in range(0, N_x+1):
            for j in range(0, N_y+1):
                V_init[i,j] = V(X[i],Y[j])
        
        U[:,:,0] = u_n.copy()

    

        #init cond - at t = 1
        #without boundary cond
        u_np1[1:N_x,1:N_y] = 2*u_n[1:N_x,1:N_y] - (u_n[1:N_x,1:N_y] - 2*dt*V_init[1:N_x,1:N_y]) + Cx2*(  0.5*(q[1:N_x,1:N_y] + q[2:N_x+1,1:N_y ])*(u_n[2:N_x+1,1:N_y] - u_n[1:N_x,1:N_y])  - 0.5*(q[0:N_x -1,1:N_y] + q[1:N_x,1:N_y ])*(u_n[1:N_x,1:N_y] - u_n[0:N_x -1,1:N_y]) ) + Cy2*(  0.5*(q[1:N_x,1:N_y] + q[1:N_x ,2:N_y+1])*(u_n[1:N_x,2:N_y+1] - u_n[1:N_x,1:N_y])  - 0.5*(q[1:N_x,0:N_y -1] + q[1:N_x ,1:N_y])*(u_n[1:N_x,1:N_y] - u_n[1:N_x,0:N_y -1]) )


        #boundary conditions
        if bound_cond == 1:
            #Dirichlet bound cond
            u_np1[0,:] = 0
            u_np1[-1,:] = 0
            u_np1[:,0] = 0
            u_np1[:,-1] = 0



        elif bound_cond == 2:
            #Nuemann bound cond
            i,j = 0,0
            u_np1[i,j] = 2*u_n[i,j] - (u_n[i,j] - 2*dt*V_init[i,j]) + Cx2*(q[i,j] + q[i+1,j])*(u_n[i+1,j] - u_n[i,j]) + Cy2*(q[i,j] + q[i,j+1])*(u_n[i,j+1] - u_n[i,j])
            
            i,j = 0,N_y
            u_np1[i,j] = 2*u_n[i,j] - (u_n[i,j] - 2*dt*V_init[i,j]) + Cx2*(q[i,j] + q[i+1,j])*(u_n[i+1,j] - u_n[i,j]) + Cy2*(q[i,j] + q[i,j-1])*(u_n[i,j-1] - u_n[i,j])
                        
            i,j = N_x,0
            u_np1[i,j] = 2*u_n[i,j] - (u_n[i,j] - 2*dt*V_init[i,j]) + Cx2*(q[i,j] + q[i-1,j])*(u_n[i-1,j] - u_n[i,j]) + Cy2*(q[i,j] + q[i,j+1])*(u_n[i,j+1] - u_n[i,j])
                    
            i,j = N_x,N_y
            u_np1[i,j] = 2*u_n[i,j] - (u_n[i,j] - 2*dt*V_init[i,j]) + Cx2*(q[i,j] + q[i-1,j])*(u_n[i-1,j] - u_n[i,j]) + Cy2*(q[i,j] + q[i,j-1])*(u_n[i,j-1] - u_n[i,j])      
            
            i = 0
            u_np1[i,1:N_y -1] = 2*u_n[i,1:N_y -1] - (u_n[i,1:N_y -1] - 2*dt*V_init[i,1:N_y -1]) + Cx2*(q[i,1:N_y -1] + q[i+1,1:N_y -1])*(u_n[i+1,1:N_y -1] - u_n[i,1:N_y -1]) + Cy2*(  0.5*(q[i,1:N_y -1] + q[i,2:N_y])*(u_n[i,2:N_y] - u_n[i,1:N_y -1])  - 0.5*(q[i,0:N_y -2] + q[i,1:N_y -1])*(u_n[i,1:N_y -1] - u_n[i,0:N_y -2]) )
                
            j = 0
            u_np1[1:N_x -1,j] = 2*u_n[1:N_x -1,j] - (u_n[1:N_x -1,j] - 2*dt*V_init[1:N_x -1,j]) + Cx2*(  0.5*(q[1:N_x -1,j] + q[2:N_x,j])*(u_n[2:N_x,j] - u_n[1:N_x -1,j])  - 0.5*(q[0:N_x -2,j] + q[1:N_x -1,j])*(u_n[1:N_x -1,j] - u_n[0:N_x -2,j]) ) + Cy2*(q[1:N_x -1,j] + q[1:N_x -1,j+1])*(u_n[1:N_x -1,j+1] - u_n[1:N_x -1,j])
        
            i = N_x
            u_np1[i,1:N_y -1] = 2*u_n[i,1:N_y -1] - (u_n[i,1:N_y -1] - 2*dt*V_init[i,1:N_y -1]) + Cx2*(q[i,1:N_y -1] + q[i-1,1:N_y -1])*(u_n[i-1,1:N_y -1] - u_n[i,1:N_y -1]) + Cy2*(  0.5*(q[i,1:N_y -1] + q[i,2:N_y])*(u_n[i,2:N_y] - u_n[i,1:N_y -1])  - 0.5*(q[i,0:N_y -2] + q[i,1:N_y -1])*(u_n[i,1:N_y -1] - u_n[i,0:N_y -2]) )
                
            j = N_y
            u_np1[1:N_x -1,j] = 2*u_n[1:N_x -1,j] - (u_n[1:N_x -1,j] - 2*dt*V_init[1:N_x -1,j]) + Cx2*(  0.5*(q[1:N_x -1,j] + q[2:N_x,j])*(u_n[2:N_x,j] - u_n[1:N_x -1,j])  - 0.5*(q[0:N_x -2,j] + q[1:N_x -1,j])*(u_n[1:N_x -1,j] - u_n[0:N_x -2,j]) ) + Cy2*(q[1:N_x -1,j] + q[1:N_x -1,j-1])*(u_n[1:N_x -1,j-1] - u_n[1:N_x -1,j])
               
               
        elif bound_cond == 3:
            #Nuemann bound cond
            i = 0
            u_np1[i,:] = u_n[i+1,:] + (CFL_3 - 1)/(CFL_3 + 1)*(u_np1[i+1,:] - u_n[i,:])
            
            j = 0
            u_np1[:,j] = u_n[:,j+1] + (CFL_1 - 1)/(CFL_1 + 1)*(u_np1[:,j+1] - u_n[:,j])
            
            i = N_x
            u_np1[i,:] = u_n[i-1,:] + (CFL_4 - 1)/(CFL_4 + 1)*(u_np1[i-1,:] - u_n[i,:])
            
            j = N_y
            u_np1[:,j] = u_n[:,j-1] + (CFL_2 - 1)/(CFL_2 + 1)*(u_np1[:,j-1] - u_n[:,j])
        
        
    
    
        u_nm1 = u_n.copy()
        u_n = u_np1.copy()
        U[:,:,1] = u_n.copy()

    
        #Process loop (on time mesh)
        for n in range(2, N_t):
            
            #calculation at step j+1  
            #without boundary cond           
            u_np1[1:N_x,1:N_y] = 2*u_n[1:N_x,1:N_y] - u_nm1[1:N_x,1:N_y] + Cx2*(  0.5*(q[1:N_x,1:N_y] + q[2:N_x+1,1:N_y])*(u_n[2:N_x+1,1:N_y] - u_n[1:N_x,1:N_y])  - 0.5*(q[0:N_x - 1,1:N_y] + q[1:N_x,1:N_y])*(u_n[1:N_x,1:N_y] - u_n[0:N_x - 1,1:N_y]) ) + Cy2*(  0.5*(q[1:N_x ,1:N_y] + q[1:N_x,2:N_y+1])*(u_n[1:N_x,2:N_y+1] - u_n[1:N_x,1:N_y])  - 0.5*(q[1:N_x,0:N_y - 1] + q[1:N_x,1:N_y])*(u_n[1:N_x,1:N_y] - u_n[1:N_x,0:N_y - 1]) )
                
                
                
            #bound conditions
            if bound_cond == 1:
                #Dirichlet bound cond
                u_np1[0,:] = 0
                u_np1[-1,:] = 0
                u_np1[:,0] = 0
                u_np1[:,-1] = 0
                
            
            elif bound_cond == 2:
                #Nuemann bound cond
                i,j = 0,0
                u_np1[i,j] = 2*u_n[i,j] - u_nm1[i,j] + Cx2*(q[i,j] + q[i+1,j])*(u_n[i+1,j] - u_n[i,j]) + Cy2*(q[i,j] + q[i,j+1])*(u_n[i,j+1] - u_n[i,j])
                            
                i,j = 0,N_y
                u_np1[i,j] = 2*u_n[i,j] - u_nm1[i,j] + Cx2*(q[i,j] + q[i+1,j])*(u_n[i+1,j] - u_n[i,j]) + Cy2*(q[i,j] + q[i,j-1])*(u_n[i,j-1] - u_n[i,j])
                                
                i,j = N_x,0
                u_np1[i,j] = 2*u_n[i,j] - u_nm1[i,j] + Cx2*(q[i,j] + q[i-1,j])*(u_n[i-1,j] - u_n[i,j]) + Cy2*(q[i,j] + q[i,j-1])*(u_n[i,j-1] - u_n[i,j])
                        
                i,j = N_x,N_y
                u_np1[i,j] = 2*u_n[i,j] - u_nm1[i,j] + Cx2*(q[i,j] + q[i-1,j])*(u_n[i-1,j] - u_n[i,j]) + Cy2*(q[i,j] + q[i,j-1])*(u_n[i,j-1] - u_n[i,j])
                        
                        
                i = 0
                u_np1[i,1:N_y -1] = 2*u_n[i,1:N_y -1] - u_nm1[i,1:N_y -1] + Cx2*(q[i,1:N_y -1] + q[i+1,1:N_y -1])*(u_n[i+1,1:N_y -1] - u_n[i,1:N_y -1]) + Cy2*(  0.5*(q[i,1:N_y -1] + q[i,2:N_y])*(u_n[i,2:N_y] - u_n[i,1:N_y -1])  - 0.5*(q[i,0:N_y -2] + q[i,j])*(u_n[i,1:N_y -1] - u_n[i,0:N_y -2]) )
                            
                j = 0
                u_np1[1:N_x - 1,j] = 2*u_n[1:N_x - 1,j] - u_nm1[1:N_x - 1,j] + Cx2*(  0.5*(q[1:N_x - 1,j] + q[2:N_x,j])*(u_n[2:N_x,j] - u_n[1:N_x - 1,j])  - 0.5*(q[0:N_x - 2,j] + q[1:N_x - 1,j])*(u_n[1:N_x - 1,j] - u_n[0:N_x - 2,j]) ) + Cy2*(q[1:N_x - 1,j] + q[1:N_x - 1,j+1])*(u_n[1:N_x - 1,j+1] - u_n[1:N_x - 1,j])
                        
                i = N_x
                u_np1[i,1:N_y -1] = 2*u_n[i,1:N_y -1] - u_nm1[i,1:N_y -1] + Cx2*(q[i,1:N_y -1] + q[i-1,1:N_y -1])*(u_n[i-1,1:N_y -1] - u_n[i,1:N_y -1]) + Cy2*(  0.5*(q[i,1:N_y -1] + q[i,2:N_y])*(u_n[i,2:N_y] - u_n[i,1:N_y -1])  - 0.5*(q[i,0:N_y -2] + q[i,1:N_y -1])*(u_n[i,1:N_y -1] - u_n[i,0:N_y -2]) )
                        
                j = N_y
                u_np1[1:N_x - 1,j] = 2*u_n[1:N_x - 1,j] - u_nm1[1:N_x - 1,j] + Cx2*(  0.5*(q[1:N_x - 1,j] + q[2:N_x,j])*(u_n[2:N_x,j] - u_n[1:N_x - 1,j])  - 0.5*(q[0:N_x - 2,j] + q[1:N_x - 1,j])*(u_n[1:N_x - 1,j] - u_n[0:N_x - 2,j]) ) + Cy2*(q[1:N_x - 1,j] + q[1:N_x - 1,j-1])*(u_n[1:N_x - 1,j-1] - u_n[1:N_x - 1,j])
                    

            elif bound_cond == 3:
                #Mur bound cond
                i = 0
                u_np1[i,:] = u_n[i+1,:] + (CFL_3 - 1)/(CFL_3 + 1)*(u_np1[i+1,:] - u_n[i,:])
                
                j = 0
                u_np1[:,j] = u_n[:,j+1] + (CFL_1 - 1)/(CFL_1 + 1)*(u_np1[:,j+1] - u_n[:,j])
                
                i = N_x
                u_np1[i,:] = u_n[i-1,:] + (CFL_4 - 1)/(CFL_4 + 1)*(u_np1[i-1,:] - u_n[i,:])
                
                j = N_y
                u_np1[:,j] = u_n[:,j-1] + (CFL_2 - 1)/(CFL_2 + 1)*(u_np1[:,j-1] - u_n[:,j])
        
        
            u_nm1 = u_n.copy()      
            u_n = u_np1.copy() 
            U[:,:,n] = u_n.copy()


    return torch.from_numpy(U[0:-1,0:-1,0:100]), torch.from_numpy(U[0:-1,0:-1,100:200])#torch.from_numpy(u_n[0:-1,0:-1]), torch.from_numpy(U[0:-1,0:-1,300])

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

def precision_matrix():
    #Set grid range [0,A]X[0,B] where A=1, B=1
    #Set cell numbers where M = 200, N = 200
    A, B = 1, 1
    M, N = 65, 65 # 64, 64 #64x64/128x128 not happend postive-semidefinite warning.(I dont know why now)
    hx, hy = A/M, B/N
    Dv = hx*hy*torch.eye(M*N)
    Dk = 10 * torch.eye(M*N) # here we set k^2 = 1
    ah = A_H(M,N,hx,hy)
    A = torch.matmul(Dv,Dk)-ah
    Q1 =torch.matmul(A,Dv)
    Q = torch.matmul(Q1,torch.transpose(A, 0, 1))
    return torch.zeros(M*N) , Q

def NSGRF():
    u, Q = precision_matrix()
    Q = Q/np.linalg.norm(Q)#suwei add not test yet
    ns = np.random.multivariate_normal(mean = u,cov = Q)
    return ns.reshape(65,65)

def data_gen():
    trainsize = 200
    a = torch.zeros(trainsize,64,64,100)# suwei: elements are train size, X size, Y size, time elpased, respectively.
    u = torch.zeros(trainsize,64,64,100)# suwei: elements are train size, X size, Y size, time elpased, respectively.
    for i in range(trainsize):
        a[i,:,:,:], u[i,:,:,:] = wave_gen()
    scipy.io.savemat('/data/suwei/wave3D/waveequationnorm_spd0001_trainsize200_64x64.mat', mdict={'a': a.cpu().numpy(), 'u': u.cpu().numpy()})

    # a, u = wave_gen()
    return a, u

def data_read():
    # data = scipy.io.loadmat('FRFT/FRFTNO/NSdataset/waveequation_trainsize30.mat')
    # data = scipy.io.loadmat('/data/suwei/wave3D/waveequationnorm_trainsize200_64x64.mat')
    data = scipy.io.loadmat('/data/suwei/wave3D/waveequationnorm_spd0001_trainsize200_64x64.mat')
    return torch.from_numpy(data['a']), torch.from_numpy(data['u'])

#Analysis why original FNO not works here
def data_analysis():
    a, u =wave_gen()
    fs = 10e2#10e3
    # x = np.zeros(64000)
    # for i in range(1000):
    #     x[i*64:((i+1)*64)]=a[:,0,0].reshape(64)
    f, t, Sxx = signal.spectrogram(a[:,:,3].reshape(64*64), fs)
    plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('X+Y')
    plt.show()  
    plt.savefig("FRFT/Output/NSanalysis2.png") 
    plt.clf()

    # rng = np.random.default_rng()
    # fs = 10e3
    # N = 1e5
    # amp = 2 * np.sqrt(2)
    # noise_power = 0.01 * fs / 2
    # time = np.arange(N) / float(fs)
    # mod = 500*np.cos(2*np.pi*0.25*time)
    # carrier = amp * np.sin(2*np.pi*3e3*time + mod)
    # noise = rng.normal(scale=np.sqrt(noise_power), size=time.shape)
    # noise *= np.exp(-time/5)
    # x = carrier + noise    
    # f, t, Sxx = signal.spectrogram(x, fs)
    # plt.pcolormesh(t, f, Sxx, shading='gouraud')
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.show()  
    # plt.savefig("FRFT/Output/NSanalysis.png")  
    return

if __name__ == "__main__":
    # a, u = wave_gen()
    # scipy.io.savemat('waveequation.mat', mdict={'a': a.cpu().numpy(), 'u': u.cpu().numpy()})
    # wave_plot()
    # data_analysis()
    data_gen()
    # data_read()
    print("haha")