"""
@Author: Matheus Teixeira de Sousa (mtsousa14@gmail.com)

Implement the functions to calculate the velocity and pressure
"""

import numpy as np
from numba import njit
from typing import Tuple

@njit
def calculate_u_star_exp(u: np.ndarray, v: np.ndarray, Nx: int, Ny: int, dx: float, dy: float, dt: float,
                         Re: int, u_star: np.ndarray, U: np.ndarray, obs_i: np.ndarray, obs_j: np.ndarray,
                         L: np.ndarray, inside_obs: np.ndarray, obs: bool=False) -> np.ndarray:
    """
    Calculate the matrix u* with the explicit method

    Params
        - u, v: Matrices u and v
        - Nx, Ny: Grid size on axes x and y
        - dx, dy: Spatial increment
        - dt: Temporal increment
        - Re: Reynolds number
        - u_star: Matrix u*
        - U: Velocity profile U
        - obs_i: Position of the obstacle on axis x
        - obs_j: Position of the obstacle on axis y
        - L: Obstacle size
        - inside_obs: Vector to check if the loop is inside any obstacle
        - obs: Flag to set the obstacle presence
    
    Returns
        - u_star: Matrix u*
    """
    for i in range(1, Nx):
        for j in range(0, Ny):
            # Check if the point is inside an obstacle
            for k in range(len(obs_i)):
                inside_obs[k] = ((i > obs_i[k] and i < obs_i[k]+L[k]) and (j > obs_j[k] and j < obs_j[k]+L[k]))

            if not inside_obs.any() or not obs:
                C = (v[i, j+1] + v[i-1, j+1] + v[i, j] + v[i-1, j])/4
                aux_convectivo = -dt*(u[i, j]*(u[i+1, j] - u[i-1, j])/(2*dx) +
                                    C*(u[i, j+1] - u[i, j-1])/(2*dy))
                aux_difusivo = (dt/Re)*((u[i+1, j] - 2*u[i, j] + u[i-1, j])/dx**2 +
                                        (u[i, j+1] - 2*u[i, j] + u[i, j-1])/dy**2)
                u_star[i, j] = u[i, j] + aux_convectivo + aux_difusivo

    # Update the ghost points of u_star
    u_star[0, 0:Ny] = 0.0
    u_star[Nx, 0:Ny] = 0.0
    u_star[0:Nx+1, -1] = -u_star[0:Nx+1, 0]
    u_star[0:Nx+1, Ny] = 2*U[0:Nx+1] - u_star[0:Nx+1, Ny-1]

    if obs:
        # For each obstacle, update the ghost points
        for i, j, k in zip(obs_i, obs_j, L):
            u_star[i, j:j+k] = 0.0
            u_star[i+k, j:j+k] = 0.0
            u_star[i+1:i+k, j+1] = -u_star[i+1:i+k, j]
            u_star[i+1:i+k, j+k-1] = -u_star[i+1:i+k, j+k]

    return u_star

@njit
def calculate_u_star_imp(u: np.ndarray, v: np.ndarray, Nx: int, Ny: int, dx: float, dy: float, dt: float,
                         Re: int, tol: float, u_star: np.ndarray, U: np.ndarray, obs_i: np.ndarray,
                         obs_j: np.ndarray, L: np.ndarray, inside_obs: np.ndarray, obs: bool=False) -> np.ndarray:
    """
    Calculate the matrix u* with the implicit method

    Params
        - u, v: Matrices u and v
        - Nx, Ny: Grid size on axes x and y
        - dx, dy: Spatial increment
        - dt: Temporal increment
        - Re: Reynolds number
        - tol: Tolarece of the iteration error
        - u_star: Matrix u*
        - U: Velocity profile U
        - obs_i: Position of the obstacle on axis x
        - obs_j: Position of the obstacle on axis y
        - L: Obstacle size
        - inside_obs: Vector to check if the loop is inside any obstacle
        - obs: Flag to set the obstacle presence
    
    Returns
        - u_star: Matrix u*
    """
    iteracao = 0
    error = 100
    while error > tol:
        R_max = 0
        for i in range(1, Nx):
            for j in range(0, Ny):
                # Check if the point is inside an obstacle
                for k in range(len(obs_i)):
                    inside_obs[k] = ((i > obs_i[k] and i < obs_i[k]+L[k]) and (j > obs_j[k] and j < obs_j[k]+L[k]))

                if not inside_obs.any() or not obs:
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
            print('[ERROR] - Maximum number of iterations reached.')
            break

    # Update the ghost points of u_star
    u_star[0, 0:Ny] = 0.0
    u_star[Nx, 0:Ny] = 0.0
    u_star[0:Nx+1, -1] = -u_star[0:Nx+1, 0]
    u_star[0:Nx+1, Ny] = 2*U[0:Nx+1] - u_star[0:Nx+1, Ny-1]

    if obs:
        # For each obstacle, update the ghost points
        for i, j, k in zip(obs_i, obs_j, L):
            u_star[i, j:j+k] = 0.0
            u_star[i+k, j:j+k] = 0.0
            u_star[i+1:i+k, j+1] = -u_star[i+1:i+k, j]
            u_star[i+1:i+k, j+k-1] = -u_star[i+1:i+k, j+k]
    
    return u_star

@njit
def calculate_v_star_exp(u: np.ndarray, v: np.ndarray, Nx: int, Ny: int, dx: float, dy: float, dt: float,
                         Re: int, v_star: np.ndarray, obs_i: np.ndarray, obs_j: np.ndarray,
                         L: np.ndarray, inside_obs: np.ndarray, obs: bool=False) -> np.ndarray:
    """
    Calculate the matrix v* with the explicit method

    Params
        - u, v: Matrices u and v
        - Nx, Ny: Grid size on axes x and y
        - dx, dy: Spatial increment
        - dt: Temporal increment
        - Re: Reynolds number
        - v_star: Matrix v*
        - obs_i: Position of the obstacle on axis x
        - obs_j: Position of the obstacle on axis y
        - L: Obstacle size
        - inside_obs: Vector to check if the loop is inside any obstacle
        - obs: Flag to set the obstacle presence
    
    Returns
        - v_star: Matrix v*
    """
    for i in range(0, Nx):
        for j in range(1, Ny):
            # Check if the point is inside an obstacle
            for k in range(len(obs_i)):
                inside_obs[k] = ((i > obs_i[k] and i < obs_i[k]+L[k]) and (j > obs_j[k] and j < obs_j[k]+L[k]))

            if not inside_obs.any() or not obs:
                C = (u[i+1, j] + u[i+1, j-1] + u[i, j] + u[i, j-1])/4
                aux_convectivo = -dt*(C*(v[i+1, j] - v[i-1, j])/(2*dx) + 
                                    v[i, j]*(v[i, j+1] - v[i, j-1])/(2*dy))
                aux_difusivo = (dt/Re)*((v[i+1, j] - 2*v[i, j] + v[i-1, j])/dx**2 +
                                        (v[i, j+1] - 2*v[i, j] + v[i, j-1])/dy**2)
                v_star[i, j] = v[i, j] + aux_convectivo + aux_difusivo

    # Update the ghost points of v_star
    v_star[-1, 0:Ny+1] = -v_star[0, 0:Ny+1]
    v_star[Nx, 0:Ny+1] = -v_star[Nx-1, 0:Ny+1]
    v_star[0:Nx, 0] = 0
    v_star[0:Nx, Ny] = 0

    if obs:
        # For each obstacle, update the ghost points
        for i, j, k in zip(obs_i, obs_j, L):
            v_star[i+k-1, j+1:j+k] = -v_star[i+k, j+1:j+k]
            v_star[i+1, j+1:j+k] = -v_star[i, j+1:j+k]
            v_star[i:i+k, j] = 0
            v_star[i:i+k, j+k] = 0 
    
    return v_star

@njit
def calculate_v_star_imp(u: np.ndarray, v: np.ndarray, Nx: int, Ny: int, dx: float, dy: float, dt: float,
                         Re: int, tol: float, v_star: np.ndarray, obs_i: np.ndarray, obs_j: np.ndarray,
                         L: np.ndarray, inside_obs: np.ndarray, obs: bool=False) -> np.ndarray:
    """
    Calculate the matrix v* with the implicit method

    Params
        - u, v: Matrices u and v
        - Nx, Ny: Grid size on axes x and y
        - dx, dy: Spatial increment
        - dt: Temporal increment
        - Re: Reynolds number
        - tol: Tolarece of the iteration error
        - v_star: Matrix v*
        - obs_i: Position of the obstacle on axis x
        - obs_j: Position of the obstacle on axis y
        - L: Obstacle size
        - inside_obs: Vector to check if the loop is inside any obstacle
        - obs: Flag to set the obstacle presence
    
    Returns
        - v_star: Matrix v*
    """
    iteracao = 0
    error = 100
    while error > tol:
        R_max = 0
        for i in range(0, Nx):
            for j in range(1, Ny):
                # Check if the point is inside an obstacle
                for k in range(len(obs_i)):
                    inside_obs[k] = ((i > obs_i[k] and i < obs_i[k]+L[k]) and (j > obs_j[k] and j < obs_j[k]+L[k]))

                if not inside_obs.any() or not obs:
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
            print('[ERROR] - Maximum number of iterations reached.')
            break

    # Update the ghost points of v_star
    v_star[-1, 0:Ny+1] = -v_star[0, 0:Ny+1]
    v_star[Nx, 0:Ny+1] = -v_star[Nx-1, 0:Ny+1]
    v_star[0:Nx, 0] = 0
    v_star[0:Nx, Ny] = 0

    if obs:
        # For each obstacle, update the ghost points
        for i, j, k in zip(obs_i, obs_j, L):
            v_star[i+k-1, j+1:j+k] = -v_star[i+k, j+1:j+k]
            v_star[i+1, j+1:j+k] = -v_star[i, j+1:j+k]
            v_star[i:i+k, j] = 0
            v_star[i:i+k, j+k] = 0

    return v_star

@njit
def calculate_pressure(u_star: np.ndarray, v_star: np.ndarray, Nx: int, Ny: int, dx: float, dy: float,
                       dt: float, tol: float, pressure: np.ndarray, obs_i: np.ndarray, obs_j: np.ndarray,
                       L: np.ndarray, inside_obs: np.ndarray, obs: bool=False) -> np.ndarray:
    """
    Calculate the matrix pressure

    Params
        - u_star, v_star: Matrices u* and v*
        - Nx, Ny: Grid size on axes x and y
        - dx, dy: Spatial increment
        - dt: Temporal increment
        - tol: Tolarece of the iteration error
        - pressure: Matrix pressure
        - obs_i: Position of the obstacle on axis x
        - obs_j: Position of the obstacle on axis y
        - L: Obstacle size
        - inside_obs: Vector to check if the loop is inside any obstacle
        - obs: Flag to set the obstacle presence
    
    Returns
        - pressure: Matrix pressure
    """
    iteracao = 0
    error = 100
    while error > tol:
        R_max = 0
        for i in range(0, Nx):
            for j in range(0, Ny):
                # Check if the point is inside an obstacle
                for k in range(len(obs_i)):
                    inside_obs[k] = ((i > obs_i[k] and i < obs_i[k]+L[k]) and (j > obs_j[k] and j < obs_j[k]+L[k]))

                if not inside_obs.any() or not obs:
                    if (i == 0 and j == 0):
                        valor_lambda = -(1/dx**2 + 1/dy**2)
                        R = ((u_star[i+1, j] - u_star[i, j])/(dt*dx) +
                            (v_star[i, j+1] - v_star[i, j])/(dt*dy) - 
                            (pressure[i+1, j] - pressure[i, j])/dx**2 - 
                            (pressure[i, j+1] - pressure[i, j])/dy**2)
                    
                    elif (i == 0 and j == Ny-1):
                        valor_lambda = -(1/dx**2 + 1/dy**2)
                        R = ((u_star[i+1, j] - u_star[i, j])/(dt*dx) +
                            (v_star[i, j+1] - v_star[i, j])/(dt*dy) - 
                            (pressure[i+1, j] - pressure[i, j])/dx**2 - 
                            (-pressure[i, j] + pressure[i, j-1])/dy**2)
                    
                    elif (i == Nx-1 and j == 0):
                        valor_lambda = -(1/dx**2 + 1/dy**2)
                        R = ((u_star[i+1, j] - u_star[i, j])/(dt*dx) +
                            (v_star[i, j+1] - v_star[i, j])/(dt*dy) - 
                            (-pressure[i, j] + pressure[i-1, j])/dx**2 - 
                            (pressure[i, j+1] - pressure[i, j])/dy**2)
                    
                    elif (i == Nx-1 and j == Ny-1):
                        valor_lambda = -(1/dx**2 + 1/dy**2)
                        R = ((u_star[i+1, j] - u_star[i, j])/(dt*dx) +
                            (v_star[i, j+1] - v_star[i, j])/(dt*dy) - 
                            (-pressure[i, j] + pressure[i-1, j])/dx**2 - 
                            (-pressure[i, j] + pressure[i, j-1])/dy**2)
                    
                    elif (i == 0 and j != 0 and j != Ny-1):
                        valor_lambda = -(1/dx**2 + 2/dy**2)
                        R = ((u_star[i+1, j] - u_star[i, j])/(dt*dx) +
                            (v_star[i, j+1] - v_star[i, j])/(dt*dy) - 
                            (pressure[i+1, j] - pressure[i, j])/dx**2 - 
                            (pressure[i, j+1] - 2*pressure[i, j] + pressure[i, j-1])/dy**2)
                    
                    elif (i == Nx-1 and j != 0 and j != Ny-1):
                        valor_lambda = -(1/dx**2 + 2/dy**2)
                        R = ((u_star[i+1, j] - u_star[i, j])/(dt*dx) +
                            (v_star[i, j+1] - v_star[i, j])/(dt*dy) - 
                            (-pressure[i, j] + pressure[i-1, j])/dx**2 - 
                            (pressure[i, j+1] - 2*pressure[i, j] + pressure[i, j-1])/dy**2)
                    
                    elif (i != 0 and i != Nx-1 and j == 0):
                        valor_lambda = -(2/dx**2 + 1/dy**2)
                        R = ((u_star[i+1, j] - u_star[i, j])/(dt*dx) +
                            (v_star[i, j+1] - v_star[i, j])/(dt*dy) - 
                            (pressure[i+1, j] - 2*pressure[i, j] + pressure[i-1, j])/dx**2 - 
                            (pressure[i, j+1] - pressure[i, j])/dy**2)
                    
                    elif (i != 0 and i != Nx-1 and j == Ny-1):
                        valor_lambda = -(2/dx**2 + 1/dy**2)
                        R = ((u_star[i+1, j] - u_star[i, j])/(dt*dx) +
                            (v_star[i, j+1] - v_star[i, j])/(dt*dy) - 
                            (pressure[i+1, j] - 2*pressure[i, j] + pressure[i-1, j])/dx**2 - 
                            (-pressure[i, j] + pressure[i, j-1])/dy**2)
                    
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
            print('[ERROR] - Maximum number of iterations reached.')
            break
    
    # Update the ghost points of pressure
    pressure[0:Nx, -1] = pressure[0:Nx, 0]
    pressure[0:Nx, Ny] = pressure[0:Nx, Ny-1]
    pressure[-1, 0:Ny] = pressure[0, 0:Ny]
    pressure[Nx, 0:Ny] = pressure[Nx-1, 0:Ny]

    pressure[-1, -1] = pressure[0, 0]
    pressure[-1, Ny] = pressure[0, Ny-1]
    pressure[Nx, -1] = pressure[Nx-1, 0]
    pressure[Nx, Ny] = pressure[Nx-1, Ny-1]

    if obs:
        # For each obstacle, update the ghost points
        for i, j, k in zip(obs_i, obs_j, L):
            pressure[i+1:i+k, i+1] = pressure[i+1:i+k, j]
            pressure[i+1:i+k, j+k-1] = pressure[i+1:i+k, j+k]
            pressure[i+k-1, j+1:j+k] = pressure[i+k, j+1:j+k]
            pressure[i+1, j+1:j+k] = pressure[i, j+1:j+k]

    return pressure

@njit
def calculate_new_u(u_star: np.ndarray, pressure: np.ndarray, Nx: int, Ny: int, dx: float,
                    dt: float, u: np.ndarray, obs_i: np.ndarray, obs_j: np.ndarray, L: np.ndarray,
                    inside_obs: np.ndarray, obs: bool=False) -> np.ndarray:
    """
    Update the matrix u

    Params
        - u_star: Matrix u*
        - pressure: Matrix pressure
        - Nx, Ny: Grid size on axes x and y
        - dx: Spatial increment
        - dt: Temporal increment
        - u: Matrix u
        - obs_i: Position of the obstacle on axis x
        - obs_j: Position of the obstacle on axis y
        - L: Obstacle size
        - inside_obs: Vector to check if the loop is inside any obstacle
        - obs: Flag to set the obstacle presence
    
    Returns
        - u: Matrix u
    """
    for i in range(1, Nx):
        for j in range(-1, Ny+1):
            # Check if the point is inside an obstacle
            for k in range(len(obs_i)):
                inside_obs[k] = ((i > obs_i[k] and i < obs_i[k]+L[k]) and (j > obs_j[k] and j < obs_j[k]+L[k]))

            if not inside_obs.any() or not obs:
                u[i, j] = u_star[i, j] - dt*(pressure[i, j] - pressure[i-1, j])/dx

    return u

@njit
def calculate_new_v(v_star: np.ndarray, pressure: np.ndarray, Nx: int, Ny: int, dy: float,
                    dt: float, v: np.ndarray, obs_i: np.ndarray, obs_j: np.ndarray, L: np.ndarray,
                    inside_obs: np.ndarray, obs: bool=False) -> np.ndarray:
    """
    Update the matrix v

    Params
        - v_star: Matrix v*
        - pressure: Matrix pressure
        - Nx, Ny: Grid size on axes x and y
        - dy: Spatial increment
        - dt: Temporal increment
        - v: Matrix v
        - obs_i: Position of the obstacle on axis x
        - obs_j: Position of the obstacle on axis y
        - L: Obstacle size
        - inside_obs: Vector to check if the loop is inside any obstacle
        - obs: Flag to set the obstacle presence
    
    Returns
        - v: Matrix v
    """
    for i in range(-1, Nx+1):
        for j in range(1, Ny):
            # Check if the point is inside an obstacle
            for k in range(len(obs_i)):
                inside_obs[k] = ((i > obs_i[k] and i < obs_i[k]+L[k]) and (j > obs_j[k] and j < obs_j[k]+L[k]))

            if not inside_obs.any() or not obs:
                v[i, j] = v_star[i, j] - dt*(pressure[i, j] - pressure[i, j-1])/dy
    
    return v

@njit
def calculate_psi(u: np.ndarray, v: np.ndarray, Nx: int, Ny: int, dx: float, dy: float,
                  tol: float, psi: np.ndarray, obs_i: np.ndarray, obs_j: np.ndarray,
                  L: np.ndarray, inside_obs: np.ndarray, obs: bool=False) -> np.ndarray:
    """
    Calculate the velocity field (psi)

    Params
        - u, v: Matrices u and v
        - Nx, Ny: Grid size on axes x and y
        - dx, dy: Spatial increment
        - tol: Tolarece of the iteration error
        - psi: Matrix psi
        - obs_i: Position of the obstacle on axis x
        - obs_j: Position of the obstacle on axis y
        - L: Obstacle size
        - inside_obs: Vector to check if the loop is inside any obstacle
        - obs: Flag to set the obstacle presence
    
    Returns
        - psi: Matrix psi
    """
    valor_lambda = -(2/dx**2 + 2/dy**2)
    error = 100
    iteracao = 0
    while error > tol:
        R_max = 0
        for i in range(1, Nx):
            for j in range(1, Ny):
                # Check if the point is inside an obstacle
                for k in range(len(obs_i)):
                    inside_obs[k] = ((i >= obs_i[k] and i <= obs_i[k]+L[k]) and (j >= obs_j[k] and j <= obs_j[k]+L[k]))

                if not inside_obs.any() or not obs:
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
            print('[ERROR] - Maximum number of iterations reached.')
            break
    
    return psi

def utils_compiler():
    """
    Call all the other functions once to 'compile' with Numba
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
    w = np.zeros((Nx+1, Ny+1), float)

    U = np.zeros(Nx+1, float)
    U[:] = 1
    u[:, Ny] = 2*U[:]

    obs_i = np.asarray([4])
    obs_j = np.asarray([4])
    L = np.asarray([2])
    inside_obs = np.zeros(obs_i.shape, bool)
    obs = False

    calculate_u_star_imp(u, v, Nx, Ny, dx, dy, dt, Re, tol, u_star, U, obs_i, obs_j, L, inside_obs, obs)
    calculate_v_star_imp(u, v, Nx, Ny, dx, dy, dt, Re, tol, v_star, obs_i, obs_j, L, inside_obs, obs)
    calculate_u_star_exp(u, v, Nx, Ny, dx, dy, dt, Re, u_star, U, obs_i, obs_j, L, inside_obs, obs)
    calculate_v_star_exp(u, v, Nx, Ny, dx, dy, dt, Re, v_star, obs_i, obs_j, L, inside_obs, obs)
    calculate_pressure(u_star, v_star, Nx, Ny, dx, dy, dt, tol, pressure, obs_i, obs_j, L, inside_obs, obs)
    calculate_new_u(u_star, pressure, Nx, Ny, dx, dt, u, obs_i, obs_j, L, inside_obs, obs)
    calculate_new_v(v_star, pressure, Nx, Ny, dy, dt, v, obs_i, obs_j, L, inside_obs, obs)
    calculate_psi(u, v, Nx, Ny, dx, dy, tol, psi, obs_i, obs_j, L, inside_obs, obs)
    calculate_vorticity(u, v, dx, dy, Nx, Ny, w, obs_i, obs_j, L, inside_obs, obs)

@njit
def calculate_vorticity(u: np.ndarray, v: np.ndarray, dx: float, dy: float, Nx: int, Ny: int,
                        w: np.ndarray, obs_i: np.ndarray, obs_j: np.ndarray, L: np.ndarray,
                        inside_obs: np.ndarray, obs: bool=False) -> np.ndarray:
    """
    Calculate the vorticity (w)

    Params
        - u, v: Matrices u and v
        - dx, dy: Spatial increment
        - Nx, Ny: Grid size on axes x and y
        - w: Matrix w
        - obs_i: Position of the obstacle on axis x
        - obs_j: Position of the obstacle on axis y
        - L: Obstacle size
        - inside_obs: Vector to check if the loop is inside any obstacle
        - obs: Flag to set the obstacle presence
    
    Returns
        - w: Matrix w
    """
    for i in range(1, Nx):
        for j in range(1, Ny):
            # Check if the point is inside an obstacle
            for k in range(len(obs_i)):
                inside_obs[k] = ((i >= obs_i[k] and i <= obs_i[k]+L[k]) and (j >= obs_j[k] and j <= obs_j[k]+L[k]))

            if not inside_obs.any() or not obs:
                w[i, j] = (v[i+1, j] - v[i-1, j])/(2*dx) - (u[i, j+1] - u[i, j-1])/(2*dy)

    return w

def get_obstacle_size(info: str, num: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decode the obstacle information

    Params
        - info: String with obstacle information
        - num: Number of obstacles

    Returns
        - i, j, L: Vectors with position and size of each obstacle
    """
    aux = info.split(",")
    i, j, L = [], [], []
    m = 0
    
    for k in range(num):
        i.append(int(aux[m]))
        j.append(int(aux[m+1]))
        L.append(int(aux[m+2]))
        m += 3
    
    return np.asarray(i), np.asarray(j), np.asarray(L)

def check_diff(u_new, v_new, u_old, v_old, tol=1.e-6) -> bool:
    """
    Check the difference between the two iterations

    Params
        - u_new, v_new: Matrices u and v of the current iteration
        - u_old, v_old: Matrices u and v of the last iteration
        - tol: Tolerance to break the loop
    
    Returns
        - boolean to represent if the difference if lower than the tolerance
    """
    error_u = np.max(abs(u_new - u_old))
    error_v = np.max(abs(v_new - v_old))
    
    if error_u < tol and error_v < tol:
        return True
    
    return False