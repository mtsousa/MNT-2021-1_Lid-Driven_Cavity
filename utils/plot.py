"""
@Author: Matheus Teixeira de Sousa (mtsousa14@gmail.com)

Implement the functions to plot the pressure, velocity and vorticity
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

def plot_psi_stream(u_plot, v_plot, x, y, Nx, Lx, Ly, output_path, obs_i, obs_j, L, obs=False, dont_show=False, dont_save=False):
    
    fig, ax = plt.subplots(1, 1, figsize=(4, 4),
                            constrained_layout=True,
                            sharex=True, sharey=True)

    # output_path dos eixos
    ax.set_xlabel('x', fontsize=16)
    ax.set_ylabel('y', fontsize=16)
    
    ax.streamplot(x, y, np.transpose(u_plot), np.transpose(v_plot), color='k', density=2.0)
    if obs:
        for i, j, k in zip(obs_i, obs_j, L):
            ax.add_patch(
                        patches.Rectangle(
                            xy=(i/Nx, j/Nx),  # point of origin.
                            width=k/Nx, height=k/Nx, linewidth=1,
                            color='black', fill=True, zorder=3.0))
    
    ax.set_aspect('equal')
    
    # redimensiona os eixos
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    
    if not dont_save:
        plt.savefig('images/' + output_path + '_stream.pdf', format='pdf')
    
    if not dont_show:
        plt.show()

def plot_psi_contour(Nx, x, y, Lx, Ly, psi, output_path, obs_i, obs_j, L, obs=False, dont_show=False, dont_save=False):
    
    aux_min = psi[psi[:, :] < 0.0]
    aux_max = psi[psi[:, :] > 0.0]
    amax = np.linspace(np.amin(aux_max), np.amax(aux_max), 100)
    amin = np.linspace(np.amin(aux_min), np.amax(aux_min), 200)
    levels = np.concatenate((amin[:-1], amax[1:]), axis=0)
    levels = [amax[k] for k in range(1, 21, 5)]
    levels = np.concatenate(([amin[k] for k in range(0, 200, 9)], levels[:-2], [amax[k] for k in range(21, 41, 20)], [amax[60]]), axis=0)
    levels[:] = -1*levels[:]
    aux = np.zeros(levels.shape)
    for j, i in enumerate(levels):
        aux[-j-1] = i
    levels = aux
    psi = -1*psi
    # print('psi_max:', np.amax(psi))

    # x = np.linspace(0, Lx, Nx+1)
    # y = np.linspace(0, Ly, Ny+1)
    
    fig, ax = plt.subplots(1, 1, figsize=(4, 4),
    constrained_layout=True,
    sharex=True, sharey=True)
    
    # output_path dos eixos
    ax.set_xlabel('x', fontsize=16)
    ax.set_ylabel('y', fontsize=16)
    
    ax.contour(x, y, np.transpose(psi), levels, colors='k', linewidths=1.0, zorder=4.0)
    if obs:
        for i, j, k in zip(obs_i, obs_j, L):
            ax.add_patch(
                        patches.Rectangle(
                            xy=(i/Nx, j/Nx),  # point of origin.
                            width=k/Nx, height=k/Nx, linewidth=1,
                            color='black', fill=True, zorder=3.0))

    ax.set_aspect('equal')
    ax.grid(True, zorder=0.0)
    
    # redimensiona os eixos
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    
    if not dont_save:
        plt.savefig('images/' + output_path + '_contour.pdf', format='pdf')
    
    if not dont_show:
        plt.show()

def plot_pressure_contour():
    pass

def plot_u_velocity(dir, Re, Lx, Ly, Nx, Ny):
    u = np.load('data/' + str(dir) + '/u.npy')
    u_vel = np.load(f'data/u-velocity/u_velocity_Re{Re}.npy')
    idx = np.load('data/u-velocity/129grid_points.npy')

    y_vel = []
    x = np.linspace(0, Lx, 129)
    for i in idx:
        y_vel.append(x[i-1])

    u_plot = np.zeros((Nx+1, Ny+1), float)

    # Calcula as matrizes para os gráficos
    for i in range(0, Nx+1):
        for j in range(0, Ny+1):
            u_plot[i, j] = round((u[i, j] + u[i, j-1])/2, 3)

    x = np.linspace(0, Lx, Nx+1)
    y = np.linspace(0, Ly, Ny+1)
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 4),
    constrained_layout=True,
    sharex=True, sharey=True)
    
    ax.set_xlabel(r'Velocity $u$', fontsize=16)
    ax.set_ylabel('y', fontsize=16)
    
    ax.plot(u_plot[50, :], y, label='This work')
    ax.grid(True)

    ax.plot(u_vel, y_vel, 'x', label='Ghia')
    
    ax.legend()
    
    plt.savefig(f'images/Re{Re}_u_velocity.pdf', format='pdf')
    plt.show()

def plot_vorticity_contour(w, Nx, Ny, Lx, Ly, Re, output_path, obs_i, obs_j, L, obs=False, dont_show=False, dont_save=False):
    
    aux_min = w[w[:, :] < 0.0]
    aux_max = w[w[:, :] > 0.0]
    amax = np.linspace(np.amin(aux_max), np.amax(aux_max), 200)
    amin = np.linspace(np.amin(aux_min), np.amax(aux_min), 200)
    if Re == 1:
        levels = np.concatenate(([amin[k] for k in [180, 184, 188, 192, 196, 198]], [amax[k] for k in range(0, 200, 15)]), axis=0)
    elif Re == 100:
        levels = np.concatenate(([amin[k] for k in range(187, 200, 2)], [amax[k] for k in range(3, 20, 3)]), axis=0)
    elif Re == 400 or Re == 1000:
        levels = np.concatenate(([amin[k] for k in range(180, 200, 3)], [amax[k] for k in [0, 1, 3, 4, 6, 7, 8, 9, 11]]), axis=0)
    else:
        levels = np.concatenate(([amin[k] for k in range(0, 200, 20)], [amax[k] for k in range(0, 200, 20)]), axis=0)
    levels[:] = -1*levels[:]
    aux = np.zeros(levels.shape)
    for j, i in enumerate(levels):
        aux[-j-1] = i
    levels = aux
    w = -1*w

    # levels = 275
    # levels = np.concatenate(([amin[k] for k in range(0, 200, 3)], [amax[k] for k in range(0, 200, 2)]), axis=0)

    x = np.linspace(0, Lx, Nx+1)
    y = np.linspace(0, Ly, Ny+1)
    
    fig, ax = plt.subplots(1, 1, figsize=(4, 4),
    constrained_layout=True,
    sharex=True, sharey=True)
    
    # nome dos eixos
    ax.set_xlabel('x', fontsize=16)
    ax.set_ylabel('y', fontsize=16)
    
    ax.contour(x, y, np.transpose(w), levels, colors='k', linewidths=1.0, zorder=4.0)
    if obs:
        for i, j, k in zip(obs_i, obs_j, L):
            ax.add_patch(
                        patches.Rectangle(
                            xy=(i/Nx, j/Nx),  # point of origin.
                            width=k/Nx, height=k/Nx, linewidth=1,
                            color='black', fill=True, zorder=3.0))
    
    ax.set_aspect('equal')
    ax.grid(True, zorder=0.0)
    
    # redimensiona os eixos
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    
    if not dont_save:
        plt.savefig(f'images/' + output_path + '_vorticity.pdf', format='pdf')
    
    if not dont_show:
        plt.show()

def get_vectors_plot(u, v, Nx, Ny, Lx, Ly):
    
    u_plot = np.zeros((Nx+1, Ny+1), float)
    v_plot = np.zeros((Nx+1, Ny+1), float)

    # Calcula as matrizes para os gráficos
    for i in range(0, Nx+1):
        for j in range(0, Ny+1):
            u_plot[i, j] = (u[i, j] + u[i, j-1])/2
            v_plot[i, j] = (v[i, j] + v[i-1, j])/2

    x = np.linspace(0, Lx, Nx+1)
    y = np.linspace(0, Ly, Ny+1)

    return u_plot, v_plot, x, y

# Re = 400
# Nx, Ny = 100, 100
# Lx, Ly = 1, 1
# obs_i, obs_j, L = [20], [60], [20]
# obs = True

# output_path = f'Re{str(Re)}_a'
# u = np.load('data/' + output_path + '/u.npy')
# v = np.load('data/' + output_path + '/v.npy')
# psi = np.load('data/' + output_path + '/stream.npy')
# # pressure = np.load('data/' + output_path + '/pressure.npy')
# w = np.load('data/' + output_path + '/vorticity.npy')

# u_plot, v_plot, x, y = get_vectors_plot(u, v, Nx, Ny, Lx, Ly)

# # Plot stream function contour
# # plot_psi_contour(Nx, x, y, Lx, Ly, psi, output_path, obs_i, obs_j, L, obs)

# # Plot streamlines
# plot_psi_stream(u_plot, v_plot, x, y, Nx, Lx, Ly, output_path, obs_i, obs_j, L, obs)

# # Plot vorticity contour
# # plot_vorticity_contour(w, Nx, Ny, Lx, Ly, Re, output_path, obs_i, obs_j, L, obs)