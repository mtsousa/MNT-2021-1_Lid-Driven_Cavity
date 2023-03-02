# Autor: Matheus Teixeira de Sousa
# Disciplina: Métodos Numéricos em Termofluidos, Trabalho Final
#
# 
# Arquivo com as funções desenvolvidas para gerar os gráficos do escoamento

import matplotlib.pyplot as plt
import numpy as np

def plot_psi_stream(u, v, Nx, Ny, Lx, Ly, output_path, mask=np.zeros((1, 1)), obs=False):
    
    u_plot = np.zeros((Nx+1, Ny+1), float)
    v_plot = np.zeros((Nx+1, Ny+1), float)

    u_plot_unt = np.copy(u_plot)
    v_plot_unt = np.copy(v_plot)

    # Calcula as matrizes para os gráficos
    for i in range(0, Nx+1):
        for j in range(0, Ny+1):
            u_plot[i, j] = (u[i, j] + u[i, j-1])/2
            v_plot[i, j] = (v[i, j] + v[i-1, j])/2
            
            norm = (u_plot[i, j]**2 + v_plot[i, j]**2)**0.5
            
            if norm != 0:
                u_plot_unt[i, j] = u_plot[i, j]/norm
                v_plot_unt[i, j] = v_plot[i, j]/norm

    x = np.linspace(0, Lx, Nx+1)
    y = np.linspace(0, Ly, Ny+1)
    
    fig, ax = plt.subplots(1, 1, figsize=(4, 4),
                            constrained_layout=True,
                            sharex=True, sharey=True)

    aux_u = np.copy(u_plot)
    if obs:
        aux_u = np.ma.array(aux_u, mask=mask)
    
    # output_path dos eixos
    ax.set_xlabel('x', fontsize=16)
    ax.set_ylabel('y', fontsize=16)
    
    ax.streamplot(x, y, np.transpose(aux_u), np.transpose(v_plot), color='k', density=2.0)
    if obs:
        ax.imshow(~mask, cmap='gray', alpha=1, extent=(0, 1, 0, 1))
    ax.set_aspect('equal')
    
    # redimensiona os eixos
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    
    plt.savefig('images/' + output_path + '_stream.pdf', format='pdf')
    plt.show()

def plot_psi_contour(Nx, Ny, Lx, Ly, psi, output_path):
    
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

    x = np.linspace(0, Lx, Nx+1)
    y = np.linspace(0, Ly, Ny+1)
    
    fig, ax = plt.subplots(1, 1, figsize=(4, 4),
    constrained_layout=True,
    sharex=True, sharey=True)
    
    # output_path dos eixos
    ax.set_xlabel('x', fontsize=16)
    ax.set_ylabel('y', fontsize=16)
    
    ax.contour(x, y, np.transpose(psi), levels, colors='k', linewidths=1.0)
    ax.set_aspect('equal')
    ax.grid(True)
    
    # redimensiona os eixos
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    
    plt.savefig('images/' + output_path + '_contour.pdf', format='pdf')
    plt.show()
