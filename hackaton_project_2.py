# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 18:14:52 2021

@author: Me
"""
import numpy as np
import matplotlib.pyplot as plt
from photutils.datasets import make_noise_image
from perlin_numpy import generate_perlin_noise_2d
import numba
from scipy.stats import multivariate_normal
from scipy.optimize import curve_fit


from im_pro import im_processing
#%%
g = 9.8 # m s^-2
k = 1.38064852e-23 # m^2 kg s^-2 K^-1
y = np.arange(0, 12, 0.01) # km

save = False

gas_names, gas_conc = ['H20', 'CO2', 'CH4'], [(1.24/100, 0.01/100), (0.5, 0.5), (0.5 - 1.24/100, 0.5 - 0.01/100)]
#%%
def gas_fct(P0):
    H = 8.5 # km, Isothermal-barotropic approximation of earth atmosphere
    pressure = P0*np.exp(-y/H)
    return pressure

def partial_pressure_fct(gas_conc_tab_, pressure_arr_):
    #conc = np.arange(gas_conc_tab_[0], gas_conc_tab_[1], len(pressure_arr_))
    return gas_conc_tab_*pressure_arr_

def partial_refract_index_fct(temp_tab_, lamb, pressure_arr_, partial_pressure_):
    return 1 + (77.6*1e-6)/temp_tab_*(1 + (7.52/10**3)/lamb**2)*(pressure_arr_/100 + 4810*partial_pressure_/temp_tab_/100)

@numba.jit
def light_trave_fct(n_arr, gas_conc_tab, transmission_coef):
    F = n_arr.copy()/n_arr.copy() # Total flux initially
    term_profile = (n_arr.copy()/n_arr.copy())[0]*0
    for l in range(len(gas_conc_tab)):
        conc_profile = np.linspace(gas_conc_tab[l][0], gas_conc_tab[l][1], len(n_arr))
        term_profile += conc_profile*transmission_coef[l]
    
    for m in range(len(n_arr)):
        for l in range(len(n_arr[m]) - 1):
            F[l + 1,m] = F[l,m]*term_profile[m]
    return F

@numba.jit
def map_fct(gas_name_tab, gas_conc_tab, pressure_arr, lamb):
    partial_pressure_tab = []
    refract_index_tab = []
    temp_profile = np.linspace(17, -55.5, len(pressure_arr)) + 273.15 # K
    for i in range(len(gas_name_tab)):
        conc_profile = np.linspace(gas_conc_tab[i][0], gas_conc_tab[i][1], len(pressure_arr))
        partial_pressure = pressure_arr.copy()
        refract_index = pressure_arr.copy()
        for j in range(len(pressure_arr)):
            for k in range(len(pressure_arr[j])):
                T = temp_profile[k]
                conc = conc_profile[k]
                partial_pressure[k,j] = partial_pressure_fct(conc, pressure_arr[k,j])
                refract_index[k,j] = partial_refract_index_fct(T, lamb, pressure_arr[k,j], partial_pressure[k,j])
        partial_pressure_tab.append(partial_pressure)
        refract_index_tab.append(refract_index)
    return partial_pressure_tab, refract_index_tab
#%%
total_pressure = gas_fct(101300) # Pa

fig1, ax1 = plt.subplots()
ax1.plot(total_pressure, y, lw = 2, color = 'orange', label = 'Isothermal-Barotropic Model')
ax1.set_xscale('log')
ax1.set_ylabel('Altitude (km)')
ax1.set_xlabel(r'Pressure(altitude) (Pa)')
ax1.set_xlim(min(total_pressure), max(total_pressure))
plt.gca().invert_xaxis()
ax1.legend()
fig1.show()

Y = np.reshape(np.repeat(y, len(y)), (len(y), len(y)))
X = Y.copy().T
P = np.reshape(np.repeat(total_pressure, len(y)), (len(y), len(y)))
noise = make_noise_image((len(y), len(y)), distribution = 'gaussian', mean = 0., stddev = 0.4)
P += noise*10000 + 0.1*generate_perlin_noise_2d((len(y), len(y)), (10, 10))*101300

partial_P, partial_n = map_fct(gas_names, gas_conc, P, 5982.9e-6)
final_F = light_trave_fct(partial_n[0], gas_conc, [0.99, 0.99, 0.99])

fig2, ax2 = plt.subplots()
imshow2 = ax2.pcolormesh(X, Y, P, cmap = 'Reds')
ax2.set_ylabel('Altitude (km)')
ax2.set_xlabel('$X$ (a.u.)')
ax2.axis('square')
fig2.colorbar(imshow2, label = 'P + noise (Pa)')
fig2.show()

#%%
if save:
    np.savetxt('water_n.txt', partial_n[0])
    
def pathfinder(n_arr, t1, i, l):

    x = [i]
    y = [0]
    
    t = []
    n = []
    j = 1

    dir = np.zeros(2)
    dir[1] = 1

    dl = 0.5

    t2 = np.arcsin(n_arr[j - 1, i] / n_arr[j,i] * np.sin(t1))

    while True:

        p = np.tan(t2) * l

        if p > (l - dl):
            dir[:] = dir[::-1]
            dl = np.tan(np.pi - t2) * (l - dl)
        else:
            dl = dl + p

        t1 = np.abs(np.abs(np.pi * dir[0]) - t2)
        t2 = np.abs(np.arcsin(n_arr[j, i] / n_arr[j + int(dir[1]),i + int(dir[0])] * np.sin(t1)))

        if t1 > np.arcsin(n_arr[j + int(dir[1]),i + int(dir[0])]/ n_arr[j, i]) + 100:
            dir[0] = dir[0] * -1
            t2 = t1
            continue
        else:
            i += int(dir[0])
            j += int(dir[1])

            x.append(i)
            y.append(j)
            t.append(t1)
            n.append(np.arcsin(n_arr[j, i]))

        if j == (np.shape(n_arr)[1] - 1):
            break
        if i == np.shape(n_arr)[0] - 1 or i == -1:
            break

    return x, y, n, t
#%%
fig3, ax3 = plt.subplots()

def func(x, m ,b):
    return m*x + b

x_init = [50, 100, 150, 201, 250, 300]
coef = []
for i in range(len(x_init)):
    x_i, y_i, n, t = pathfinder(partial_n[0], 0.5, x_init[i], 12)

    x_i = np.array(x_i, dtype = float)/100
    y_i = (1200 - np.array(y_i, dtype = float))/100

    popt, pcov = curve_fit(func, x_i, y_i)

    coef.append(np.abs(np.arctan(popt[0])) / 0.5)

    ax3.plot(x_i, y_i, lw = 1, color = 'red')
    #ax3.plot(x_i, func(x_i, *popt), lw = 1, color = 'red')

print(coef)

#im_processing(len(x_init), coef)

imshow3 = ax3.pcolormesh(X, Y, partial_n[0], cmap = 'Blues')
ax3.set_ylabel('Altitude (km)')
ax3.set_xlabel('$X$ (a.u.)')
ax3.set_xlim(0, 12)
ax3.set_ylim(0, 12)
ax3.axis('square')
fig3.colorbar(imshow3, label = 'n$_{air}$')
fig3.show()

plt.show()

'''
from scipy import signal
k = 100
M = np.array([[1, k],
              [0, 1]])
sigma_x = 1
sigma_y = 2

mu_x = 1
mu_y = 1

d = np.sqrt((X - mu_x)**2 + (Y - mu_y)**2)
g = np.exp(-((X-mu_x)**2/(2.0*sigma_x**2 ))*((Y-mu_y)**2/(2.0*sigma_y**2 )))

#rot = signal.convolve2d(g, M, boundary='symm', mode='same')
k, h = 3, 2
XX = k*X
YY = h*Y

plt.pcolormesh(X, Y, g)
#plt.pcolormesh(X, Y, rot)
plt.xlim(len(X), len(X))
plt.axis('square')
plt.show()
'''