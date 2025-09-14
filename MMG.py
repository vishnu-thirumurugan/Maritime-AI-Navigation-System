import numpy as np
import os, scipy
import scipy.io
from scipy import interpolate
###############################################
############# Wave Data Import ################
###############################################
try:
    X_wave = np.array(scipy.io.loadmat(os.getcwd()+"\\"+"Environment"+"\\"+"Wave_Data"+"\\" +"DX_SALV_REV1.mat")['DX'])
    Y_wave = np.array(scipy.io.loadmat(os.getcwd()+"\\"+"Environment"+"\\"+"Wave_Data"+"\\" +"DY_SALV_REV1.mat")['DY'])
    N_wave = np.array(scipy.io.loadmat(os.getcwd()+"\\"+"Environment"+"\\"+"Wave_Data"+"\\" +"DN_SALV_REV1.mat")['DN'])
except:
    X_wave = np.array(scipy.io.loadmat(os.getcwd()+"\\"+"Wave_Data"+"\\" +"DX_SALV_REV1.mat")['DX'])
    Y_wave = np.array(scipy.io.loadmat(os.getcwd()+"\\"+"Wave_Data"+"\\" +"DY_SALV_REV1.mat")['DY'])
    N_wave = np.array(scipy.io.loadmat(os.getcwd()+"\\"+"Wave_Data"+"\\" +"DN_SALV_REV1.mat")['DN'])



###############################################
############### MMG Model Group ###############
###############################################


def MMG_Time_Derivative(u,v,r,x,y,psi,delta,rps,WHA,w_flag):
    """
    ----------
    ip : current state of the ship [u,v,r,x,y,psi]
    delta : commanded rudder angle
    rpm : commanded propeller speed

    Returns
    -------
    Time derivative of ship state[u_dot,v_dot,r_dot,U,V,r]

    """
    ### Assertion ###
    if u == 0 or u < 0:
        u = 0.00001
    if rps == 0 or rps < 0:
        rps = 1
    ### ###
    U             = np.sqrt((u**2)+(v**2))   # Resultant speed
    beta          = np.arctan(-v/u)          # Hull drift angle at midship
    Np            = rps                      #just for notation
    
    ################################################
    ###### Priniciple Particulars of the ship ########
    ################################################
    Lpp     = 2.902        # length between perpenticular
    d       = 0.189        # ship draft
    xG      = 0.102        # Longitudinal coordinate of center of gravity of ship
    rho     = 1025         # Water density
    Volume  = 0.235        # Displacement volume of ship
    Dp      = 0.090        # Propeller diameter
    HR      = 0.144        # Rudder span length
    AR      = 0.00928      # Profile area of movable part of mariner rudder
    
    ######################################################
    #### Mass, Added Mass and Added Moment of inertia ####
    ######################################################
    mx,my,jz = 0.022,0.223,0.011
    
    ### Non DiMensionalizing(ndm) parameters ###
    ndm_force   = 0.5*rho*Lpp*d*(U**2)
    ndm_moment  = 0.5*rho*(Lpp**2)*d*(U**2)
    ndm_mass    = 0.5*rho*d*(Lpp**2)
    ndm_massMoI = 0.5*rho*d*(Lpp**4)
    
    ### Dimensional mass ###
    mx = mx*ndm_mass
    my = my*ndm_mass
    jz = jz*ndm_massMoI
    m  = Volume*rho
    IzG = m * ((0.25*Lpp)**2) #Moment of inertia of ship around center of gravity
    #####################################################
    ####### Governing Equation (Equation 4) #############
    #####################################################
    
    ### Mass Matrix ### (u_dot,v_dot,r_dot)
    M = np.array([[(m+mx),0,0],[0,(m+my),m*xG],[0,m*xG,(jz+((xG**2)*m)+IzG)]])
    
    ### LHS ### (Other than u_dot,v_dot,r_dot) #which should be added on right side
    LHS_r = np.array([[-(m+my)*v*r-xG*m*(r**2)], [(m+mx)*u*r], [m*xG*u*r]])
    
    #####################################################
    ####### Hydrodynamic Force on Hull(F_hull)  #########
    #####################################################
    ### Non Dimensional Velocity ###
    v_ndm = v/U
    r_ndm = r*Lpp/U
    
    R0 = 0.022 # Ship resistance coefficient in straight moving
    
    ### Non Dimensional Hydrodynamic Derivatives ###
    X_vv, X_vr ,X_rr ,X_vvvv = -0.04, 0.002, 0.011, 0.771
    Y_v, Y_R, Y_vvv, Y_vvr, Y_vrr, Y_rrr = -0.315, 0.083, -1.607, 0.379, -0.391, 0.008
    N_v, N_R, N_vvv, N_vvr, N_vrr, N_rrr = -0.137, -0.049, -0.03, -0.294, 0.055, -0.013
    
    ### Force and Moment ###
    X_hull =  -R0 + (X_vv*(v_ndm**2)) + (X_vr*v_ndm*r_ndm) + (X_rr*(r_ndm**2)) + (X_vvvv*(v_ndm**4))
                
    Y_hull =  (Y_v*v_ndm) + (Y_R*r_ndm) + (Y_vvv*(v_ndm**3)) + (Y_vvr*(v_ndm**2)*r_ndm) + (Y_vrr*v_ndm*(r_ndm**2)) + (Y_rrr*(r_ndm**3))
                
    N_hull = (N_v*v_ndm) + (N_R*r_ndm) + (N_vvv*(v_ndm**3)) + (N_vvr*(v_ndm**2)*r_ndm) + (N_vrr*v_ndm*(r_ndm**2)) + (N_rrr*(r_ndm**3))
    
    ### Equation 6 ###
    X_hull = ndm_force * X_hull
    Y_hull = ndm_force * Y_hull
    N_hull = ndm_moment* N_hull
    
    ### Hull Force ###
    F_hull = np.array([[X_hull],[Y_hull],[N_hull]])
    
    #####################################################
    ##### Propellar Force on Ship(F_propellar)  #########
    #####################################################
    tp          = 0.220                    # Thrust deduction factor
    wp0         = 0.40                     # Effective wake in straight moving
    k0,k1,k2    = 0.2931, -0.2753, -0.1385 # 2nd order polynomial function coefficient
    xP          = -0.48                    # Longitudinal coordinate of propellerposition
    beta_P      = beta - (xP * r_ndm)      # geometrical inflow angle 
    
    
    # C1,C2       = 2, 1.6 if beta_P > 0 else 1.2 # wake  change characteristic coefficents
    # wp          = (1 - wp0)* (1 + ((1 - np.exp(-C1*abs(beta_P)))*(C2 - 1)))
    #                                             Equation 16
    wp          =  wp0*np.exp(-4*(beta_P**2))   # Equation 12
    Jp          =  u*(1-wp)/(Np*Dp)             # Equation 11
    KT          =  k0 + k1*Jp + k2*(Jp**2)      # Equation 10
    T           =  rho*(Np**2)*(Dp**4)*KT       # Equation 9
    X_propellar =  (1 - tp)*T                   # Equation 8
    
    ### Propellar Force ###
    F_propellar = np.array([[X_propellar],[0],[0]])
    
    #####################################################
    ######### Rudder Force on Ship(F_rudder)  ###########
    #####################################################
    
    ### Equation 46 ###
    epsilon    = 1.09                   # Ratio of wake fraction
    k          = 0.5                    # An experimental constant forexpressing uR
    eta        = Dp/HR                  # Ratio of propeller diameter to rudderspan 
    uP         = (1 - wp)*u             # propeller inflow velocity
    
    uR1   = np.sqrt(1 + ((8*KT)/(np.pi*(np.square(Jp)))))
    uR2   = np.square(1 + k*(uR1 -1))
    uR    = epsilon * uP * np.sqrt( (eta*uR2) + (1-eta))
    
    ### Equation 23 ###
    lR          = -0.710                        # Effective longitudinal coordinate
    beta_R      =  beta -(lR*r_ndm)             # Effective inflow angle to rudder in maneuvering motions
    gamma_r     = 0.396 if beta_R < 0 else 0.64 # Flow straightening coefficient
    
    vR    =  U * gamma_r * beta_R
   
    ### Equation 19 ###
    f_alpha     = 2.747                     # Rudder lift gradient coefficient
    # print(type(delta))
    # print(type(vR))
    temp_2 = np.arctan(vR/uR)
    temp_2 = temp_2.numpy()
    # print(type(temp_2))
    alpha_R     = delta - temp_2  # Effective inflow angle to rudder 
    Ur          = np.sqrt((uR**2)+(vR**2))  # Resultant inflow velocity to rudder
    
    F_normal = 0.5 * rho * AR * (Ur**2) * f_alpha * np.sin(alpha_R)
    
    ### Equation 18 ###
    tR         = 0.387                      # Steering resistance deduction factor
    aH         = 0.312                      # Rudder force increase factor
    xH         = -0.464   # -0.45* Lpp      # Longitudinal coordinate of acting point of the additional lateral force
    xR         = -0.5 * Lpp                 # Longitudinal coordinate of rudder position 
    
    X_rudder =  -(1 - tR) * F_normal * np.sin(delta)
    Y_rudder =  -(1 + aH) * F_normal * np.cos(delta)
    N_rudder =  -(xR + (aH*xH)) * F_normal * np.cos(delta)
    
    ### Rudder Force ###
    F_rudder = np.array([[X_rudder],[Y_rudder],[N_rudder]])
    #####################################################
    ####### Wave Drift Force on Ship(F_Drift)  ##########
    #####################################################
    if w_flag == True:
        Lamda              = 320/Lpp
        FS_force           = Lamda**3
        FS_moment          = Lamda**4
        
        Amplitude          = 3.2
        wave_frequency     = 0.439*(Lamda)**0.5 #0.3584
        heading_angle      = WHA 
        
        A1  =  np.real(X_wave[::,1:]) * (Amplitude**2)/2
        A2  = -np.real(Y_wave[::,1:]) * (Amplitude**2)/2
        A3  = -np.real(N_wave[::,1:]) * (Amplitude**2)/2
    
        wave_drift     = np.real(2*np.pi*X_wave[::,0])*(Lamda)**0.5
        hangle_drift   = -np.arange(-np.pi,np.pi+0.01,np.deg2rad(15))
        hangle_desire1  = heading_angle - psi
      
        hangle_desire = (hangle_desire1 + np.pi) % (2 * np.pi) - np.pi
            
        fA1 = interpolate.interp2d(hangle_drift, wave_drift,A1) # kind='cubic'(generally linear)
        fA2 = interpolate.interp2d(hangle_drift, wave_drift,A2) # kind='cubic'(generally linear)
        fA3 = interpolate.interp2d(hangle_drift, wave_drift,A3) # kind='cubic'(generally linear)
        
        X_Drift =  fA1(hangle_desire,wave_frequency)  / FS_force
        Y_Drift =   fA2(hangle_desire,wave_frequency) / FS_force  
        N_Drift =   fA3(hangle_desire,wave_frequency) / FS_moment
            
        F_Drift = np.array([[X_Drift[0]],[Y_Drift[0]],[N_Drift[0]]])

    elif w_flag == False :
        F_Drift = np.array([[0.0],[0.0],[0.0]])
    #####################################################
    ##### Solution of Governing Equation (A.X = b)  #####
    #####################################################
    A_inverse       = np.linalg.inv(M) 
    b               = F_hull + F_propellar + F_rudder - LHS_r + F_Drift
    
    ### u_dot,v_dot,r_dot ###
    X   = A_inverse.dot(b)
    #####################################################
    ############## Kinetics of the Ship   ###############
    #####################################################
    ### Rotation Matrix ###
    R = np.array( [[np.cos(psi), -np.sin(psi), 0], [np.sin(psi),  np.cos(psi), 0], [0, 0, 1]])
    ### Velocity and Moment ###
    Vel_Mom = R.dot(np.array([u,v,r]))
    
    ### time derivatives ###
    TD = [X[0].item(), X[1].item(), X[2].item(),Vel_Mom[0].item(), Vel_Mom[1].item(), Vel_Mom[2].item(),]
    
    return np.array(TD)

delta_set = []
def activate(ip,RuPos,delta,rps,WHA,w_flag):
    """
    ----------
    ip    : current state of the ship [u,v,r,x,y,psi]
    RuPos : Rudder Position
    delta : commanded rudder angle
    rps   : commanded propeller speed
    WHA   : Wave Head Angle
    W_Flag: Wave Force Inclusion
    Returns
    -------
    Time derivative of ship state[u_dot,v_dot,r_dot,U,V,r]

    """
    delta_set.append(delta)
    h = 3 # time step as increment
    u,v,r,x,y,psi = ip[0],ip[1],ip[2],ip[3],ip[4],ip[5]
    if abs((RuPos - delta)/h) > (np.deg2rad(20)):
        delta = RuPos + (np.sign(delta)*np.deg2rad(20)*h)
    else :
        delta = delta
    if abs(delta) > np.deg2rad(35):
        delta = np.sign(delta) * np.deg2rad(35)
    #########################################
    RuPos_new = delta
    ###############################    
    ### RK4's K1,K2,K3,K4 Value ###
    ###############################
    K   = np.zeros(8)
    K0  = np.array([u,v,r,x,y,psi])
  
    K1_ = MMG_Time_Derivative(u,v,r,x,y,psi,delta,rps,WHA,w_flag)
    K1  = h*K1_[:3] 
    k1  = h*K0[:3]
   
    K2_ = MMG_Time_Derivative(u+(K1[0]/2), v+(K1[1]/2), r+(K1[2]/2), x+(k1[0]/2), y+(k1[1]/2), psi+(k1[2]/2),delta,rps,WHA,w_flag)
    K2  = h*K2_[:3]
    k2  = h*(K0[:3] + 0.5*K1)
        
    K3_ = MMG_Time_Derivative(u+(K2[0]/2), v+(K2[1]/2), r+(K2[2]/2),x+(k2[0]/2), y+(k2[1]/2), psi+(k2[2]/2),delta,rps,WHA,w_flag)
    K3  = h*K3_[:3]
    k3  = h*(K0[:3] + 0.5*K2)
   
    K4_ = MMG_Time_Derivative(u+(K3[0]), v+(K3[1]), r+(K3[2]), x+(k3[0]), y+(k3[1]), psi+(k3[2]),delta,rps,WHA,w_flag)
    K4  = h*K4_[:3]
    k4  = h*(K0[:3] + 0.5*K3)
   
   
    K[:3]  =  (1/6) * (K1 + (2*K2) + (2*K3) + K4)
   
    R      = np.array( [[np.cos(psi), -np.sin(psi), 0], [np.sin(psi),  np.cos(psi), 0], [0, 0, 1]])
    del_k  =  ((1/6) * (k1 + (2*k2) + (2*k3) + k4)).reshape(3,1)
    K[3:6] = (R.dot(del_k)).reshape(1,3)
    
    nxt_state = np.array(ip) + K
   
    return nxt_state.tolist(), RuPos_new, delta_set


# ####################################################
# ############ To check ##############################
# ####################################################
# import matplotlib
# matplotlib.rc('xtick', labelsize=12) 
# matplotlib.rc('ytick', labelsize=12) 
# font = {'family' : 'normal',
#         'size'   : 12}
# matplotlib.rc('font', **font)
# import matplotlib.pyplot as plt

# Lpp   = 2.902
# rps   = 18.78
# N     = 200
# WHA   = np.pi
# FF    = False # True 



# ip1         = [0.77,0,0,0,0,0,0,0]
# ip2         = [0.77,0,0,0,0,0,0,0]

# print(ip1)
# data= list()
# data1 = list()
# RuPos  = 0.0
# RuPos1  = 0.0
# data.append(ip1)
# data1.append(ip2)

# x1,y1 = [ip1[2]],[ip1[3]]
# x2,y2 = [ip1[2]],[ip1[3]]

# Y = []

# for i in range(N):
#     temp,RuPos_New, delta_set = activate(data[-1],RuPos,np.deg2rad(35),rps,WHA,w_flag =FF) 
#     RuPos           = RuPos_New
#     data.append(temp)
#     x1.append(temp[3]/Lpp)
#     y1.append(temp[4]/Lpp)
#     Y.append(temp[2])
    
# for j in range(N):
#     temp1,RuPos_New, delta_set = activate(data1[-1],RuPos1,-np.deg2rad(35),rps,WHA,w_flag =FF) 
#     RuPos1           = RuPos_New
#     data1.append(temp1)
#     x2.append(temp1[3]/Lpp)
#     y2.append(temp1[4]/Lpp)
    

    
# plt.figure(figsize=(9,6))
# plt.plot(y1,x1,'green',linewidth=3,label = " Numerical Model ")
# plt.plot(y2,x2,'green',linewidth=3,)
# plt.scatter(0,0,marker="P",label = "starting point ")
# plt.xlabel("Transfer (x/L)")
# plt.ylabel("Advance (x/L)")
# plt.axhline(y=0,color="red",alpha=0.5)
# plt.title("L3 : KVLCC2 simulation for 120 seconds for n= 18.2 rps with u_0 = 0.76 m/s ")
# plt.legend(loc="best")
# plt.grid()
# # plt.savefig("Model_Match.jpg",dpi=480)
# plt.show()

# plt.figure(figsize=(9,6))
# plt.plot(Y)
# plt.grid()
# #####################################################
# #####################################################
         
    
    
    






