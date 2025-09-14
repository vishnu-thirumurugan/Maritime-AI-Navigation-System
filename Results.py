# -*- coding: utf-8 -*-
"""

@author: Vishnu Thirumurugan
"""

import numpy as np
import matplotlib.pyplot as plt
import os,shutil


def create_op_folder():
    """
    Returns
    -------
    Folder Creation for training plots 

    """
    
    R           = "Outputs"
    F1          = "Weights"
    F2          = "Reward_Plots"
    F3          = "Error_Plots"
    F4          = "Others"
    F5          = "Actor_Weights"
    F6          = "Critic_Weights"
    
    parent = os.getcwd()
    path0 = os.path.join(parent,R)
    path1 = os.path.join(path0, F1)
    path2 = os.path.join(path0, F2)
    path3 = os.path.join(path0, F3)
    path4 = os.path.join(path0, F4)
    path5 = os.path.join(path1, F5)
    path6 = os.path.join(path1, F6)
    
    
    try :
        os.mkdir(path0)
    except FileExistsError:
        shutil.rmtree(path0)
        os.mkdir(path0)
    
    try :
        os.mkdir(path1)
    except FileExistsError:
        os.rmdir(path1)
        os.mkdir(path1)
        
    try :
        os.mkdir(path2)
    except FileExistsError:
        os.rmdir(path2)
        os.mkdir(path2)
        
    try :
        os.mkdir(path3)
    except FileExistsError:
        os.rmdir(path3)
        os.mkdir(path3)
        
    try :
        os.mkdir(path4)
    except FileExistsError:
        os.rmdir(path4)
        os.mkdir(path4)

    try :
        os.mkdir(path5)
    except FileExistsError:
        os.rmdir(path5)
        os.mkdir(path5)
    
    try :
        os.mkdir(path6)
    except FileExistsError:
        os.rmdir(path6)
        os.mkdir(path6)        
        
def plot_1(Episode_number, Cumulative_reward,path,N="End"):
    plt.figure(figsize=(9,12))
    #############################
    plt.subplot(2,1,1)
    N = len(Cumulative_reward)
    running_avg1 = np.empty(N)
    for t in range(N):
            running_avg1[t] = np.mean(Cumulative_reward[max(0, t-100):(t+1)])
    plt.plot(Cumulative_reward,label="Cumulative Reward",color="c",alpha=0.2)
    plt.plot(running_avg1,color='c',label="Running Average")
    plt.title("TD3 Training Results : Average Score & Episode Durations")
    plt.xlabel("Training frames")
    plt.ylabel("Average score")
    plt.legend(loc="best")
    plt.grid()
    ##############################
    plt.subplot(2,1,2)
    N = len(Episode_number)
    running_avg2 = np.empty(N)
    for t in range(N):
            running_avg2[t] = np.mean(Episode_number[max(0, t-100):(t+1)])
    plt.plot(Episode_number,color="m",label = "Episode Durations",alpha=0.2)
    plt.plot(running_avg2,color='m',label="Running Average" )
    plt.xlabel("Training frames")
    plt.ylabel("Episode duration")
    plt.legend(loc="best")
    plt.grid()
    plt.savefig(os.path.join(path,"Rewards"+str(N)+".jpg" ),dpi=1200)
    plt.close()
    
    
def plot_2(HEs,MSE,path,N="End"):
    plt.figure(figsize=(9,12))
    ##############################
    plt.subplot(2,1,1)
    N = len(MSE)
    running_avg1 = np.empty(N)
    for t in range(N):
            running_avg1[t] = np.mean(MSE[max(0, t-100):(t+1)])
    plt.plot(MSE,color="k",label="Mean Square Error",alpha=0.2)
    plt.plot(running_avg1,color='k',label="Running Average")
    plt.title(" DQN Training Resuts : Mean Squared Loss , Heading Error ")
    plt.xlabel("Training frames")
    plt.ylabel("Mean Square Error")
    plt.legend(loc="best")
    plt.grid()
    
    
    plt.subplot(2,1,2)
    N = len(HEs)
    running_avg2 = np.empty(N)
    for t in range(N):
            running_avg2[t] = np.mean(HEs[max(0, t-100):(t+1)])
    plt.plot(HEs,color="y",label = "Cumulative Heading Error",alpha=0.2)
    plt.plot(running_avg2,color='y',label="Running Average" )
    plt.xlabel("Training frames")
    plt.ylabel("Heading Error")
    plt.legend(loc="best")
    plt.grid()
    plt.savefig(os.path.join(path,"Errors"+str(N)+".jpg" ),dpi=1200)
    plt.close()
