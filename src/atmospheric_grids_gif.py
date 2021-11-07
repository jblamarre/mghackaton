# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 12:37:31 2021

@author: Me
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import random
import numba
#%%

@numba.jit
def xy_move(t, vx_, vy_, x_, y_, rho, grid_dim):
    #pressure = np.zeros(grid_dim)
    #for i in range(len(pressure)):
    #    pressure[i] = len(np.where(y_ <= y[i])[0])
    #ay = X.copy()
    #for k in range(len(y_)):
    #    for j in range(len(y_[k])):
    #        press = pressure.max() - pressure[np.where(y <= y_[j,k])[0][-1]]
    #        #print(y_[j,k], np.where(y <= y_[j,k])[0][-1], press)
    #        ay[j,k] = -g + press

    ay = -g + 2.72**(-y_)
    new_x, new_y = vx_*t + x_, vy_*t + y_# + 0.5*ay*t**2
    y_index_1 = np.where(new_y < ground)
    y_index_2 = np.where(new_y > atmo)

    x_index_1 = np.where(new_x < 0)
    x_index_2 = np.where(new_x >= grid_dim - 1)

    new_y[y_index_1] = y_[y_index_1]#abs(vy_[y_index_1])*t + y_[y_index_1] + 0.5*ay[y_index_1]*t**2
    new_y[y_index_2] = y_[y_index_2]#-abs(vy_[y_index_2])*t + y_[y_index_2] + 0.5*ay[y_index_2]*t**2

    new_x[x_index_1] = (new_x[x_index_1]/new_x[x_index_1])*grid_dim - 1
    new_x[x_index_2] = new_x[x_index_2]*0
    return new_x, new_y

@numba.jit
def animate_gif(ite, ax_gif, vx, vy, x_, y_, grid_dim, dt):
    # Nettoyage de la figure pour le .gif
    ax_gif.clear()
    
    # Dust Coordinate Update
    x_arr[:,:], y_arr[:,:] = xy_move(dt, vx_arr, vy_arr, x_arr, y_arr, 1, grid_dim)

    # Affichage du .gif
    ax_gif.set_title("t = " + str(round(ite*dt, 2)))
    ax_gif.pcolormesh(X, Y, Z, cmap = 'gray')
    ax_gif.scatter(x_arr, y_arr, marker = '.', s = 1, color = 'red', alpha = 0.5)
    ax_gif.set_xlim(0, grid_dim)
    ax_gif.set_ylim(0, grid_dim)
    ax_gif.axis("square")
    ax_gif.set_xlabel("$X$")
    ax_gif.set_ylabel("$Y$")
    return 

def atmospheric_grids_gif():
    # Initial Settings
    g = 9.8 # m s^-2
    vx0, vx_rdm = 0, 1e-2 # Amplitude of initial velocities
    vy0, vy_rdm = 0, 1e-2
    ground = 10
    atmo = 90
    dt = 1 # Time increment
    grid_dim = 101 # Square grid side size
    N = 50

    # Initial Positions & Velocities in 2D
    x_arr = np.asarray([[np.random.uniform(0, grid_dim) for i in range(N)] for j in range(N)], dtype = 'object')
    y_arr = np.asarray([[np.random.uniform(ground, atmo) for i in range(N)] for j in range(N)], dtype = 'object')

    vx_arr = vx0 + np.asarray([[np.random.uniform(-vx_rdm, vx_rdm) for i in range(N)] for j in range(N)], dtype = 'object')
    vy_arr = vy0 + np.asarray([[np.random.uniform(-vy_rdm, vy_rdm) for i in range(N)] for j in range(N)], dtype = 'object')

    # 1D Coordinates
    x = np.arange(0, grid_dim, 1)
    y = x.copy()

    # Physical Grid
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((len(x), len(y)))

    # Segmenting Atmosphere
    i = (Y[:,0] >= ground) & (Y[:,0] < atmo)
    Z[i] = 10000

    #%%
    # Making & Saving Animation (GIF)
    fig_gif, ax_gif = plt.subplots()
    animation = FuncAnimation(fig_gif, animate_gif, frames =\
                   300, fargs = (ax_gif, vx_arr, vy_arr, x_arr, y_arr, grid_dim, dt))
    animation.save('animation.gif', writer = PillowWriter(fps = 24))
