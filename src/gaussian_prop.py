# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from scipy import special
import numba
import scipy.linalg as la
from pylab import *
from hackaton_project_2 import pathfinder

tabnn = loadtxt('data/water_n.txt')
#tabn = np.array([1.01,1.11,1.12,1.5])
tabn = tabnn[:,500]

def Mprop(d):
    M = np.array([[1,d],[0,1]])
    return M
    
def Minttan(theta1,n1,n2):
    theta2 = np.arcsin(n1*np.sin(theta1)/n2)
    M = np.diag([np.cos(theta2)/np.cos(theta1),(n1*np.cos(theta1))/(n2*np.cos(theta2))])
    return M

def Mintsag(n1,n2):
    M = np.diag([1,n1/n2])
    return M


def qfinal(tabni, thetai):
    
    Mtan = np.matmul(Mprop(1),Minttan(0,1,1))
    Msag = np.matmul(Mprop(1),Mintsag(1,1))
    #thetai = np.deg2rad(45)
    print(len(tabnnn))
    for i in range(0,len(tabnnn)-2):
        
        Mtan = np.matmul(Minttan(thetai[i],tabni[i],tabni[i+1]),Mtan)
        Mtan = np.matmul(Mprop(0.1),Mtan)  
        
        Msag = np.matmul(Mintsag(tabni[i],tabni[i+1]),Msag)
        Msag = np.matmul(Mprop(0.1),Msag) 
    
    Atan = Mtan[0][0]
    Btan = Mtan[0][1]
    Ctan = Mtan[1][0]
    Dtan = Mtan[1][1]
    
    Asag = Msag[0][0]
    Bsag = Msag[0][1]
    Csag = Msag[1][0]
    Dsag = Msag[1][1]
    
    z0tan2 = imag((Atan*z0tan1*1j+Btan)/(Ctan*z0tan1*1j+Dtan))
    z0sag2 = imag((Asag*z0sag1*1j+Bsag)/(Csag*z0sag1*1j+Dsag))
    
    lamda = 500e-9
    W0tan2 = np.sqrt(lamda*z0tan2/np.pi)
    W0sag2 = np.sqrt(lamda*z0sag2/np.pi)
    
    return W0tan2,W0sag2





