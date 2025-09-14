# -*- coding: utf-8 -*-
"""

@author: Vishnu Thirumurugan
"""
import numpy as np

def Goal_monitor(ip,op,G):
    """
    Returns
    -------
    flag : when agent moves towards the goal, flag is True

    """
    Flag = False
    x_d0 = np.square(G[0] - ip[3])
    y_d0 = np.square(G[1] - ip[4])
    D0 = np.sqrt(x_d0+y_d0)
    
    x_d1 = np.square(G[0] - op[3])
    y_d1 = np.square(G[1] - op[4])
    D1 = np.sqrt(x_d1 + y_d1)
    
    if D0 >= D1 :
        Flag = True 
   
    return Flag, D1

def CTE(y_e):
    if 0 <= abs(y_e) <= 2:
                R =100
    elif 2 < abs(y_e) <= 3.5:
        R = 20
    else:
        c0 = abs(y_e)/49
        c1 = 1 - c0
        R  = 20 * c1 
    return R

      
def get(ip,op,HE,y_e,G, rp_diff):
    """
    Parameters
    ----------
    ip          : input state
    op          : output state
    y_e         : cross track error
    y_e_old     : previous cross track error
    HE          : heading error
    HE_old      : previous heading error
    G           : goal
    T_i         : Tolerence index for 
    i_episode   : episode number

    Returns
    -------
    Rf : Reward

    """
    
    R = 1 - (abs(HE)*abs(y_e))
    
    # print(HE)
    # goal_target = Goal_monitor(ip,op,G)
    # Flag, D_goal1 = goal_target
    # # if Flag == True:
    # #     R+= 0.5* abs(R)
    
    # if D_goal1 < 3:
    #     R+= 10 
        
    ################################
    ########### Assertion ##########
    ################################
   
    Rf = np.array([R])
        
    return Rf
    

########################################
############# To check #################
# ########################################
# ip = [7.75,0,0,15,15,0]
# op = [7.75,0,0,16,16,0]
# G = [300,300]
# ss = get(ip,op,0,0,G)
# print(ss)
########################################
########################################

