"""
@Author: Matheus Teixeira de Sousa (mtsousa14@gmail.com)

Implement the functions to calculate velocity and pressure
"""

import numpy as np
from numba import njit

@njit
def calculate_u_star_exp(u, v, Nx, Ny, dx, dy, dt, Re, u_star, U):
    for i in range(1, Nx):
        for j in range(0, Ny):
            C = (v[i, j+1] + v[i-1, j+1] + v[i, j] + v[i-1, j])/4
            aux_convectivo = -dt*(u[i, j]*(u[i+1, j] - u[i-1, j])/(2*dx) +
                                C*(u[i, j+1] - u[i, j-1])/(2*dy))
            aux_difusivo = (dt/Re)*((u[i+1, j] - 2*u[i, j] + u[i-1, j])/dx**2 +
                                    (u[i, j+1] - 2*u[i, j] + u[i, j-1])/dy**2)
            u_star[i, j] = u[i, j] + aux_convectivo + aux_difusivo

    # Atualiza os ghost points de u_star
    u_star[0, 0:Ny] = 0
    u_star[Nx, 0:Ny] = 0
    u_star[0:Nx+1, -1] = -u_star[0:Nx+1, 0]
    u_star[0:Nx+1, Ny] = 2*U[0:Nx+1] - u_star[0:Nx+1, Ny-1]

    return u_star

@njit
def calculate_u_star_imp(u, v, Nx, Ny, dx, dy, dt, Re, tol, u_star, U):
    iteracao = 0
    error = 100
    while error > tol:
        R_max = 0
        for i in range(1, Nx):
            for j in range(0, Ny):
                C = (v[i, j+1] + v[i-1, j+1] + v[i, j] + v[i-1, j])/4
                
                lamb = (1 + 2*dt/(Re*dx**2) + 3*dt/(Re*dy**2))**(-1)
                if j == 0:
                    R = lamb*((u[i, j] - dt*u[i, j]*(u[i+1, j] - u[i-1, j])/(2*dx) - dt*C*(u[i, j+1] - u[i, j-1])/(2*dy)) 
                              - (u_star[i, j] - (dt/Re)*((u_star[i+1, j] - 2*u_star[i, j] + u_star[i-1, j])/(dx**2) + (u_star[i, j+1] - 3*u_star[i, j])/(dy**2))))
                elif j == Ny-1:
                    R = lamb*((u[i, j] - dt*u[i, j]*(u[i+1, j] - u[i-1, j])/(2*dx) - dt*C*(u[i, j+1] - u[i, j-1])/(2*dy)) 
                              - (u_star[i, j] - (dt/Re)*((u_star[i+1, j] - 2*u_star[i, j] + u_star[i-1, j])/(dx**2) + (2*U[i] - 3*u_star[i, j] + u_star[i, j-1])/(dy**2))))
                else:
                    R = lamb*((u[i, j] - dt*u[i, j]*(u[i+1, j] - u[i-1, j])/(2*dx) - dt*C*(u[i, j+1] - u[i, j-1])/(2*dy)) 
                              - (u_star[i, j] - (dt/Re)*((u_star[i+1, j] - 2*u_star[i, j] + u_star[i-1, j])/(dx**2) + (u_star[i, j+1] - 2*u_star[i, j] + u_star[i, j-1])/(dy**2))))

                u_star[i, j] = u_star[i, j] + R

                if abs(R) > R_max:
                    R_max = abs(R)

        error = R_max

        iteracao += 1
        if iteracao > 10**6:
            print('[ERRO] Atingiu o limite de iterações')
            break

    # Update boundary conditions
    u_star[0, 0:Ny] = 0
    u_star[Nx, 0:Ny] = 0
    u_star[0:Nx+1, -1] = -u_star[0:Nx+1, 0]
    u_star[0:Nx+1, Ny] = 2*U[0:Nx+1] - u_star[0:Nx+1, Ny-1]

    return u_star

@njit
def calculate_v_star_exp(u, v, Nx, Ny, dx, dy, dt, Re, v_star):
    for i in range(0, Nx):
        for j in range(1, Ny):
            # Percorrer todas as linhas, inclusive as bordas
            C = (u[i+1, j] + u[i+1, j-1] + u[i, j] + u[i, j-1])/4
            aux_convectivo = -dt*(C*(v[i+1, j] - v[i-1, j])/(2*dx) + 
                                v[i, j]*(v[i, j+1] - v[i, j-1])/(2*dy))
            aux_difusivo = (dt/Re)*((v[i+1, j] - 2*v[i, j] + v[i-1, j])/dx**2 +
                                    (v[i, j+1] - 2*v[i, j] + v[i, j-1])/dy**2)
            v_star[i, j] = v[i, j] + aux_convectivo + aux_difusivo

    # Atualiza os ghost points de v_star
    v_star[-1, 0:Ny+1] = -v_star[0, 0:Ny+1]
    v_star[Nx, 0:Ny+1] = -v_star[Nx-1, 0:Ny+1]
    v_star[0:Nx, 0] = 0
    v_star[0:Nx, Ny] = 0

    return v_star

@njit
def calculate_v_star_imp(u, v, Nx, Ny, dx, dy, dt, Re, tol, v_star):
    iteracao = 0
    error = 100
    while error > tol:
        R_max = 0
        for i in range(0, Nx):
            for j in range(1, Ny):
                C = (u[i+1, j] + u[i, j] + u[i+1, j-1] + u[i, j-1])/4
                
                lamb = (1 + 3*dt/(Re*dx**2) + 2*dt/(Re*dy**2))**(-1)
                if i == 0:
                    R = lamb*((v[i, j] - dt*C*(v[i+1, j] - v[i-1, j])/(2*dx) - dt*v[i, j]*(v[i, j+1] - v[i, j-1])/(2*dy)) 
                              - (v_star[i, j] - dt*((v_star[i+1, j] - 3*v_star[i, j])/(dx**2) + (v_star[i, j+1] - 2*v_star[i, j] + v_star[i, j-1])/(dy**2))/Re))
                elif i == Nx-1:
                    R = lamb*((v[i, j] - dt*C*(v[i+1, j] - v[i-1, j])/(2*dx) - dt*v[i, j]*(v[i, j+1] - v[i, j-1])/(2*dy)) 
                              - (v_star[i, j] - dt*((-3*v_star[i, j] + v_star[i-1, j])/(dx**2) + (v_star[i, j+1] - 2*v_star[i, j] + v_star[i, j-1])/(dy**2))/Re))
                else:
                    R = lamb*((v[i, j] - dt*C*(v[i+1, j] - v[i-1, j])/(2*dx) - dt*v[i, j]*(v[i, j+1] - v[i, j-1])/(2*dy)) 
                              - (v_star[i, j] - dt*((v_star[i+1, j] - 2*v_star[i, j] + v_star[i-1, j])/(dx**2) + (v_star[i, j+1] - 2*v_star[i, j] + v_star[i, j-1])/(dy**2))/Re))
                
                v_star[i, j] = v_star[i, j] + R

                if abs(R) > R_max:
                    R_max = abs(R)

        error = R_max

        iteracao += 1
        if iteracao > 10**6:
            print('[ERRO] Atingiu o limite de iterações')
            break

    # Update boundary conditions
    v_star[-1, 0:Ny+1] = -v_star[0, 0:Ny+1]
    v_star[Nx, 0:Ny+1] = -v_star[Nx-1, 0:Ny+1]
    v_star[0:Nx, 0] = 0
    v_star[0:Nx, Ny] = 0

    return v_star

@njit
def calculate_pressure(u_star, v_star, Nx, Ny, dx, dy, dt, tol, pressure):
    iteracao = 0
    error = 100
    while error > tol:
        R_max = 0
        for i in range(0, Nx):
            for j in range(0, Ny):
                # Percorrer todas as linhas e colunas, inclusive as bordas
                # Topo parede esquerda da cavidade ou da direita do objeto
                if (i == 0 and j == 0):
                    valor_lambda = -(1/dx**2 + 1/dy**2)
                    R = ((u_star[i+1, j] - u_star[i, j])/(dt*dx) +
                        (v_star[i, j+1] - v_star[i, j])/(dt*dy) - 
                        (pressure[i+1, j] - pressure[i, j])/dx**2 - 
                        (pressure[i, j+1] - pressure[i, j])/dy**2)
                # Base da parede esquerda da cavidade ou da direita do objeto
                elif (i == 0 and j == Ny-1):
                    valor_lambda = -(1/dx**2 + 1/dy**2)
                    R = ((u_star[i+1, j] - u_star[i, j])/(dt*dx) +
                        (v_star[i, j+1] - v_star[i, j])/(dt*dy) - 
                        (pressure[i+1, j] - pressure[i, j])/dx**2 - 
                        (-pressure[i, j] + pressure[i, j-1])/dy**2)
                # Topo da parede direita da cavidade ou da parede esquerda do objeto
                elif (i == Nx-1 and j == 0):
                    valor_lambda = -(1/dx**2 + 1/dy**2)
                    R = ((u_star[i+1, j] - u_star[i, j])/(dt*dx) +
                        (v_star[i, j+1] - v_star[i, j])/(dt*dy) - 
                        (-pressure[i, j] + pressure[i-1, j])/dx**2 - 
                        (pressure[i, j+1] - pressure[i, j])/dy**2)
                # Base da parede direita da cavidade ou da parede esquerda do objeto
                elif (i == Nx-1 and j == Ny-1):
                    valor_lambda = -(1/dx**2 + 1/dy**2)
                    R = ((u_star[i+1, j] - u_star[i, j])/(dt*dx) +
                        (v_star[i, j+1] - v_star[i, j])/(dt*dy) - 
                        (-pressure[i, j] + pressure[i-1, j])/dx**2 - 
                        (-pressure[i, j] + pressure[i, j-1])/dy**2)
                # Pontos internos da tampa da cavidade ou da base do objeto
                elif (i == 0 and j != 0 and j != Ny-1):
                    valor_lambda = -(1/dx**2 + 2/dy**2)
                    R = ((u_star[i+1, j] - u_star[i, j])/(dt*dx) +
                        (v_star[i, j+1] - v_star[i, j])/(dt*dy) - 
                        (pressure[i+1, j] - pressure[i, j])/dx**2 - 
                        (pressure[i, j+1] - 2*pressure[i, j] + pressure[i, j-1])/dy**2)
                # Pontos internos da base da cavidade ou da tampa do objeto
                elif (i == Nx-1 and j != 0 and j != Ny-1):
                    valor_lambda = -(1/dx**2 + 2/dy**2)
                    R = ((u_star[i+1, j] - u_star[i, j])/(dt*dx) +
                        (v_star[i, j+1] - v_star[i, j])/(dt*dy) - 
                        (-pressure[i, j] + pressure[i-1, j])/dx**2 - 
                        (pressure[i, j+1] - 2*pressure[i, j] + pressure[i, j-1])/dy**2)
                # Pontos internos da parede esquerda da cavidade ou da parede direita do objeto
                elif (i != 0 and i != Nx-1 and j == 0):
                    valor_lambda = -(2/dx**2 + 1/dy**2)
                    R = ((u_star[i+1, j] - u_star[i, j])/(dt*dx) +
                        (v_star[i, j+1] - v_star[i, j])/(dt*dy) - 
                        (pressure[i+1, j] - 2*pressure[i, j] + pressure[i-1, j])/dx**2 - 
                        (pressure[i, j+1] - pressure[i, j])/dy**2)
                # Pontos internos da parede direita da cavidade ou da parede esquerda do objeto
                elif (i != 0 and i != Nx-1 and j == Ny-1):
                    valor_lambda = -(2/dx**2 + 1/dy**2)
                    R = ((u_star[i+1, j] - u_star[i, j])/(dt*dx) +
                        (v_star[i, j+1] - v_star[i, j])/(dt*dy) - 
                        (pressure[i+1, j] - 2*pressure[i, j] + pressure[i-1, j])/dx**2 - 
                        (-pressure[i, j] + pressure[i, j-1])/dy**2)
                # Se não for borda da cavidade ou do objeto, vem para cá
                else:
                    valor_lambda = -(2/dx**2 + 2/dy**2)
                    R = ((u_star[i+1, j] - u_star[i, j])/(dt*dx) +
                        (v_star[i, j+1] - v_star[i, j])/(dt*dy) - 
                        (pressure[i+1, j] - 2*pressure[i, j] + pressure[i-1, j])/dx**2 - 
                        (pressure[i, j+1] - 2*pressure[i, j] + pressure[i, j-1])/dy**2)
                
                R = R/valor_lambda
                pressure[i, j] = pressure[i, j] + R

                if np.abs(R) > R_max:
                    R_max = np.abs(R)

        error = R_max
        iteracao += 1
        if iteracao > 10**6:
            print('[ERRO] Atingiu o limite de iterações')
            break
        
    # Atualiza os ghost points da pressão
    pressure[0:Nx, -1] = pressure[0:Nx, 0]
    pressure[0:Nx, Ny] = pressure[0:Nx, Ny-1]
    pressure[-1, 0:Ny] = pressure[0, 0:Ny]
    pressure[Nx, 0:Ny] = pressure[Nx-1, 0:Ny]

    # Atualiza dos ghost points das quinas
    pressure[-1, -1] = pressure[0, 0]
    pressure[-1, Ny] = pressure[0, Ny-1]
    pressure[Nx, -1] = pressure[Nx-1, 0]
    pressure[Nx, Ny] = pressure[Nx-1, Ny-1]

    return pressure

# Verificar as condições daqui para baixo
@njit
def calculate_new_u(u_star, pressure, Nx, Ny, dx, dt, u):
    for i in range(1, Nx):
        for j in range(-1, Ny+1):
            u[i, j] = u_star[i, j] - dt*(pressure[i, j] - pressure[i-1, j])/dx

    return u

@njit
def calculate_new_v(v_star, pressure, Nx, Ny, dy, dt, v):
    for i in range(-1, Nx+1):
        for j in range(1, Ny):
            v[i, j] = v_star[i, j] - dt*(pressure[i, j] - pressure[i, j-1])/dy
    
    return v

@njit
def calculate_psi(u, v, Nx, Ny, dx, dy, dt, tol, psi):
    valor_lambda = -(2/dx**2 + 2/dy**2)
    error = 100
    iteracao = 0
    while error > tol:
        R_max = 0
        for i in range(1, Nx):
            for j in range(1, Ny):
                # if not ((i >= obs_min and i <= obs_max) and (j >= obs_min and j <= obs_max)):
                R = (-(v[i, j] - v[i-1, j])/dx + (u[i, j] - u[i, j-1])/dy
                    -(psi[i+1, j] - 2*psi[i, j] + psi[i-1, j])/dx**2
                    -(psi[i, j+1] - 2*psi[i, j] + psi[i, j-1])/dy**2)
            
                R = R/valor_lambda
                psi[i, j] = psi[i, j] + R
                
                if np.abs(R) > R_max:
                    R_max = np.abs(R)

        error = R_max
        iteracao += 1
        if iteracao > 10**6:
            print('[ERRO] Atingiu o limite de iterações')
            break
    
    return psi

def utils_compiler():
    """
    Call all other functions one time to compile with Numba
    """
    Re = 1
    tol = 1.e-2

    Nx, Ny = 10, 10
    Lx, Ly = 1.0, 1.0

    dx = Lx/Nx
    dy = Ly/Ny

    dt = Re*dx**2/5

    u = np.zeros((Nx+1, Ny+2), float)
    v = np.zeros((Nx+2, Ny+1), float)

    u_star = np.copy(u)
    v_star = np.copy(v)

    pressure = np.zeros((Nx+2, Ny+2), float)

    psi = np.zeros((Nx+1, Ny+1), float)

    U = np.zeros(Nx+1, float)
    U[:] = 1
    u[:, Ny] = 2*U[:]

    calculate_u_star_imp(u, v, Nx, Ny, dx, dy, dt, Re, tol, u_star, U)
    calculate_v_star_imp(u, v, Nx, Ny, dx, dy, dt, Re, tol, v_star)
    calculate_u_star_exp(u, v, Nx, Ny, dx, dy, dt, Re, u_star, U)
    calculate_v_star_exp(u, v, Nx, Ny, dx, dy, dt, Re, v_star)
    calculate_pressure(u_star, v_star, Nx, Ny, dx, dy, dt, tol, pressure)
    calculate_new_u(u_star, pressure, Nx, Ny, dx, dt, u)
    calculate_new_v(v_star, pressure, Nx, Ny, dy, dt, v)
    calculate_psi(u, v, Nx, Ny, dx, dy, dt, tol, psi)

def max_psi(dir):
    psi = np.load('data/' + str(dir) + '/stream.npy')
    print('psi_max:', -1*np.amin(psi))
