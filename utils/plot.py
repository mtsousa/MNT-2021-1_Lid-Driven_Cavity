"""
@Author: Matheus Teixeira de Sousa (mtsousa14@gmail.com)

Implement the functions to plot the pressure, velocity field and vorticity
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from typing import Tuple

def plot_psi_stream(u_plot: np.ndarray, v_plot: np.ndarray, x: np.ndarray, y: np.ndarray, Nx: int,
                    Lx: float, Ly: float, output_name: str, obs_i: np.ndarray, obs_j: np.ndarray,
                    L: np.ndarray, obs: bool=False, dont_show: bool=False, dont_save: bool=False):
    """
    Plot the velocity field streamlines

    Params
        - u_plot, v_plot: String with obstacle information
        - x, y: Vectors x and y of the axes
        - Nx: Grid size on axis x
        - Lx, Ly: Cavity size
        - output_name: Output plot name
        - obs_i: Position of the obstacle on axis x
        - obs_j: Position of the obstacle on axis y
        - L: Obstacle size
        - obs: Flag to set the obstacle presence
        - dont_show: Flag to don't show the output plot
        - dont_save: Flag to don't save the output plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(4, 4),
                           constrained_layout=True,
                           sharex=True, sharey=True)

    # Set the name of the axes
    ax.set_xlabel('x', fontsize=16)
    ax.set_ylabel('y', fontsize=16)
    
    # Plot the streamlines
    ax.streamplot(x, y, np.transpose(u_plot), np.transpose(v_plot), color='k', density=2.0)
    if obs:
        # For each obstacle, plot a black square
        for i, j, k in zip(obs_i, obs_j, L):
            ax.add_patch(patches.Rectangle(xy=(i/Nx, j/Nx), # Point of origin
                                           width=k/Nx, height=k/Nx, linewidth=1,
                                           color='black', fill=True, zorder=3.0))
    
    # Adjust plot aspect
    ax.set_aspect('equal')
    
    # Resize the axis
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    
    if not dont_save:
        plt.savefig('images/' + output_name + '_stream.jpg', format='jpg', dpi=1200)
    
    if not dont_show:
        plt.show()

def plot_psi_contour(psi: np.ndarray, Nx: int, x: np.ndarray, y: np.ndarray, Lx: float, Ly: float,
                     output_name: str, obs_i: np.ndarray, obs_j: np.ndarray, L: np.ndarray,
                     obs: bool=False, dont_show: bool=False, dont_save: bool=False):
    """
    Plot the velocity field contour

    Params
        - psi: Velocity field matrix
        - Nx: Grid size on axis x
        - x, y: Vectors x and y of the axes
        - Lx, Ly: Cavity size
        - output_name: Output plot name
        - obs_i: Position of the obstacle on axis x
        - obs_j: Position of the obstacle on axis y
        - L: Obstacle size
        - obs: Flag to set the obstacle presence
        - dont_show: Flag to don't show the output plot
        - dont_save: Flag to don't save the output plot
    """
    # Adjust the contour levels to visualization
    aux_min = psi[psi[:, :] < 0.0]
    aux_max = psi[psi[:, :] > 0.0]
    amax = np.linspace(np.amin(aux_max), np.amax(aux_max), 100)
    amin = np.linspace(np.amin(aux_min), np.amax(aux_min), 200)
    
    # Experimentally select some levels to plot
    levels = np.concatenate((amin[:-1], amax[1:]), axis=0)
    levels = [amax[k] for k in range(1, 21, 5)]
    levels = np.concatenate(([amin[k] for k in range(0, 200, 9)], levels[:-2], [amax[k] for k in range(21, 41, 20)], [amax[60]]), axis=0)
    levels[:] = -1*levels[:]
    
    # Invert the signal
    aux = np.zeros(levels.shape)
    for j, i in enumerate(levels):
        aux[-j-1] = i
    levels = aux
    psi = -1*psi
    
    fig, ax = plt.subplots(1, 1, figsize=(4, 4),
                           constrained_layout=True,
                           sharex=True, sharey=True)
    
    # Set the name of the axes
    ax.set_xlabel('x', fontsize=16)
    ax.set_ylabel('y', fontsize=16)
    
    # Plot the contour lines
    ax.contour(x, y, np.transpose(psi), levels, colors='k', linewidths=1.0, zorder=4.0)
    if obs:
        # For each obstacle, plot a black square
        for i, j, k in zip(obs_i, obs_j, L):
            ax.add_patch(patches.Rectangle(xy=(i/Nx, j/Nx), # Point of origin
                                           width=k/Nx, height=k/Nx, linewidth=1,
                                           color='black', fill=True, zorder=3.0))

    # Adjust plot aspect and set the grid zorder
    ax.set_aspect('equal')
    ax.grid(True, zorder=0.0)
    
    # Resize the axis
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    
    if not dont_save:
        plt.savefig('images/' + output_name + '_contour.jpg', format='jpg', dpi=1200)
    
    if not dont_show:
        plt.show()

def plot_u_velocity(Re: int, Lx: float, y: np.ndarray, u_plot: np.ndarray):
    """
    Plot the velocity u profile

    Params
        - Re: Reynolds number
        - Lx: Cavity size
        - y: Vector y of the axis
        - u_plot: Matrix u to plot
    """
    # Load velocity u data from Ghia
    u_vel = np.load(f'data/u-velocity/u_velocity_Re{Re}.npy')
    idx = np.load('data/u-velocity/129grid_points.npy')

    # Get the values of the axis y to plot Ghia data
    y_vel = []
    x_aux = np.linspace(0, Lx, 129)
    for i in idx:
        y_vel.append(x_aux[i-1])
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 4),
                           constrained_layout=True,
                           sharex=True, sharey=True)
    
    # Set the name of the axes
    ax.set_xlabel(r'Velocity $u$', fontsize=16)
    ax.set_ylabel('y', fontsize=16)
    
    # Plot simulation values and Ghia values 
    ax.plot(u_plot[50, :], y, label='This work')
    ax.plot(u_vel, y_vel, 'x', label='Ghia')

    # Add the grid and the legend
    ax.grid(True)
    ax.legend()
    
    plt.savefig(f'images/Re{Re}_u_velocity.jpg', format='jpg', dpi=1200)
    plt.show()

def plot_vorticity_contour(w: np.ndarray, Nx: int, x: np.ndarray, y: np.ndarray, Lx: float, Ly: float,
                           Re: int, output_name: str, obs_i: np.ndarray, obs_j: np.ndarray, L: np.ndarray,
                           obs: bool=False, dont_show: bool=False, dont_save: bool=False):
    """
    Plot the vorticity contour

    Params
        - w: Vorticity matrix
        - Nx: Grid size on axis x
        - x, y: Vectors x and y of the axes
        - Lx, Ly: Cavity size
        - output_name: Output plot name
        - obs_i: Position of the obstacle on axis x
        - obs_j: Position of the obstacle on axis y
        - L: Obstacle size
        - obs: Flag to set the obstacle presence
        - dont_show: Flag to don't show the output plot
        - dont_save: Flag to don't save the output plot
    """
    # Adjust the contour levels to visualization
    aux_min = w[w[:, :] < 0.0]
    aux_max = w[w[:, :] > 0.0]
    amax = np.linspace(np.amin(aux_max), np.amax(aux_max), 200)
    amin = np.linspace(np.amin(aux_min), np.amax(aux_min), 200)

    # Experimentally select some levels to plot
    if Re == 1:
        levels = np.concatenate(([amin[k] for k in [180, 184, 188, 192, 196, 198]], [amax[k] for k in range(0, 200, 15)]), axis=0)
    elif Re == 100:
        levels = np.concatenate(([amin[k] for k in range(187, 200, 2)], [amax[k] for k in range(3, 20, 3)]), axis=0)
    elif Re == 400 or Re == 1000:
        levels = np.concatenate(([amin[k] for k in range(180, 200, 3)], [amax[k] for k in [0, 1, 3, 4, 6, 7, 8, 9, 11]]), axis=0)
    else:
        levels = np.concatenate(([amin[k] for k in range(0, 200, 20)], [amax[k] for k in range(0, 200, 20)]), axis=0)
    
    # Invert the signal
    levels[:] = -1*levels[:]
    aux = np.zeros(levels.shape)
    for j, i in enumerate(levels):
        aux[-j-1] = i
    levels = aux
    w = -1*w
    
    fig, ax = plt.subplots(1, 1, figsize=(4, 4),
                           constrained_layout=True,
                           sharex=True, sharey=True)
    
    # Set the name of the axes
    ax.set_xlabel('x', fontsize=16)
    ax.set_ylabel('y', fontsize=16)
    
    # Plot the contour lines
    ax.contour(x, y, np.transpose(w), levels, colors='k', linewidths=1.0, zorder=4.0)
    if obs:
        # For each obstacle, plot a black square
        for i, j, k in zip(obs_i, obs_j, L):
            ax.add_patch(patches.Rectangle(xy=(i/Nx, j/Nx), # Point of origin
                                           width=k/Nx, height=k/Nx, linewidth=1,
                                           color='black', fill=True, zorder=3.0))
    
    # Adjust plot aspect and set the grid zorder
    ax.set_aspect('equal')
    ax.grid(True, zorder=0.0)
    
    # Resize the axis
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    
    if not dont_save:
        plt.savefig('images/' + output_name + '_vorticity.jpg', format='jpg', dpi=1200)
    
    if not dont_show:
        plt.show()

def get_matrices_plot(u: np.ndarray, v: np.ndarray, Nx: int, Ny: int, Lx: float,
                      Ly: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the matrices u, v, x, and y to plot

    Params
        - u, v: Matrices u and v
        - Nx, Ny: Grid size on axes x and y
        - Lx, Ly: Cavity size

    Returns
        - u_plot, v_plot: Matrix u and v to plot
        - x, y: Vectors x and y of the axes
    """
    u_plot = np.zeros((Nx+1, Ny+1), float)
    v_plot = np.zeros((Nx+1, Ny+1), float)

    # Calculate the matrices to plot
    for i in range(0, Nx+1):
        for j in range(0, Ny+1):
            u_plot[i, j] = (u[i, j] + u[i, j-1])/2
            v_plot[i, j] = (v[i, j] + v[i-1, j])/2

    # Calculate the vector of the axis
    x = np.linspace(0, Lx, Nx+1)
    y = np.linspace(0, Ly, Ny+1)

    return u_plot, v_plot, x, y
