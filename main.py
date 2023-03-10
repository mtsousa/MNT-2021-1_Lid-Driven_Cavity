"""
@Author: Matheus Teixeira de Sousa (mtsousa14@gmail.com)

Simulate a newtonian fluid flow over a lid-driven cavity
"""

from utils.utils import *
from utils.plot import *
import numpy as np
from tqdm import tqdm
from os.path import exists
from os import makedirs
import argparse

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
             description='Simulte newtonian flow...')
    
    parser.add_argument('--re', required=True,
                        help="Reynolds number")
    parser.add_argument('--tf', required=True,
                        help="Final time to simulation")
    parser.add_argument('--implicit', default=False, action='store_true',
                        help="Set to use implicit method")
    parser.add_argument('--n', default=100,
                        help="Grid discretization (default 100)")
    parser.add_argument('--dt', default=0.0001,
                        help="Time increment (default 0.0001)")
    parser.add_argument('--tol', default=1.e-8,
                        help="Tolerance (default 1.e-8)")
    parser.add_argument('--validation', default=False, action="store_true",
                        help="Set True to validation problem")
    # parser.add_argument('--num_obs', default=1.0,
    #                     help="Number of obstacles")
    parser.add_argument('--obstacle', default="0 0 0",
                        help="Obstacle location (i, j) and size as: 'i'x'j'x'L'.")
    parser.add_argument('--output',
                        help="Set the output name.")

    args = parser.parse_args()

    # Call functions to compile
    utils_compiler()

    # Define Reynolds, tolerance, N, L, dx, dy and dt
    Re = int(args.re)
    tol = float(args.tol)

    Nx, Ny = int(args.n), int(args.n)
    Lx, Ly = 1.0, 1.0

    dx = Lx/Nx
    dy = Ly/Ny
    dt = float(args.dt)

    # Check numerical restrictions
    error = False
    if dt >= dx:
        print('[ERROR] - dt must be lower than dx to avoid numerical instabilities.', flush=True)
        error = True
    if dx >= Re**(-0.5):
        print('[ERROR] - dx must be lower than Re^(-0.5) to avoid numerical instabilities.', flush=True)
        error = True
    if dt >= Re*(dx**2)/4  and not args.implicit:
        print('[ERROR] - dx must be lower than Re^(-0.5) to avoid numerical instabilities.', flush=True)
        error = True

    if args.validation:
        obs_i, obs_j = 0, 0
        L = 0
        obs = False
    else:
        obs_i, obs_j, L = get_obstacle_size(args.obstacle)#, args.num_obs)
        obs = True
        error = not obs

    if not error:

        # Define matrices
        u = np.zeros((Nx+1, Ny+2), float)
        v = np.zeros((Nx+2, Ny+1), float)
        pressure = np.zeros((Nx+2, Ny+2), float)
        psi = np.zeros((Nx+1, Ny+1), float)
        w = np.zeros((Nx+1, Ny+1), float)

        # Set final time
        tf = float(args.tf) # 60.0

        # Set u component initial condition
        U = np.zeros(Nx+1, float)
        U[:] = 1
        u[:, Ny] = 2*U[:]

        u_star = np.copy(u)
        v_star = np.copy(v)

        aux_u = np.copy(u)
        aux_v = np.copy(v)

        # Call explicit functions
        if not args.implicit:
            output_path = f'Re_{str(Re)}_exp_obs' if obs else f'Re_{str(Re)}_exp'
            for k in tqdm(range(int(tf/dt)), desc ="Iterations", position=0, leave=True):
                u_star = calculate_u_star_exp(u, v, Nx, Ny, dx, dy, dt, Re, u_star, U, obs_i, obs_j, L, obs)
                v_star = calculate_v_star_exp(u, v, Nx, Ny, dx, dy, dt, Re, v_star, obs_i, obs_j, L, obs)
                pressure = calculate_pressure(u_star, v_star, Nx, Ny, dx, dy, dt, tol, pressure, obs_i, obs_j, L, obs)
                u = calculate_new_u(u_star, pressure, Nx, Ny, dx, dt, u, obs_i, obs_j, L, obs)
                v = calculate_new_v(v_star, pressure, Nx, Ny, dy, dt, v, obs_i, obs_j, L, obs) 
        
                if check_diff(u, v, aux_u, aux_v, tol):
                    break

                aux_u = np.copy(u)
                aux_v = np.copy(v)
        
        # Call implicit functions
        else:
            output_path = f'Re_{str(Re)}_imp_obs' if obs else f'Re_{str(Re)}_imp'
            for k in tqdm(range(int(tf/dt)), desc ="Iterations", position=0, leave=True):
                u_star = calculate_u_star_imp(u, v, Nx, Ny, dx, dy, dt, Re, tol, u_star, U, obs_i, obs_j, L, obs)
                v_star = calculate_v_star_imp(u, v, Nx, Ny, dx, dy, dt, Re, tol, v_star, obs_i, obs_j, L, obs)
                pressure = calculate_pressure(u_star, v_star, Nx, Ny, dx, dy, dt, tol, pressure, obs_i, obs_j, L, obs)
                u = calculate_new_u(u_star, pressure, Nx, Ny, dx, dt, u, obs_i, obs_j, L, obs)
                v = calculate_new_v(v_star, pressure, Nx, Ny, dy, dt, v, obs_i, obs_j, L, obs)

                if check_diff(u, v, aux_u, aux_v, tol):
                    break

                aux_u = np.copy(u)
                aux_v = np.copy(v)

        if tol > 1.e-8:
            tol = 1.e-8
        psi = calculate_psi(u, v, Nx, Ny, dx, dy, dt, tol, psi, obs_i, obs_j, L, obs)
        print('psi_max:', -1*np.amin(psi))
        w = calculate_voticity(u, v, dx, dy, Nx, Ny, w, obs_i, obs_j, L, obs)

        if args.output != None:
            output_path = args.output
        
        # Save final matrices
        if not exists('data/' + output_path):
            makedirs('data/' + output_path)

        np.save('data/' + output_path + '/u.npy', u)
        np.save('data/' + output_path + '/v.npy', v)
        np.save('data/' + output_path + '/stream.npy', psi)
        np.save('data/' + output_path + '/pressure.npy', pressure)
        np.save('data/' + output_path + '/vorticity.npy', w)

        u_plot, v_plot, x, y = get_vectors_plot(u, v, Nx, Ny, Lx, Ly)
        
        # Plot stream function contour
        if not obs:
            plot_psi_contour(Nx, x, y, Lx, Ly, psi, output_path, obs_i, obs_j, L, obs)

        # Plot streamlines
        plot_psi_stream(u_plot, v_plot, x, y, Nx, Lx, Ly, output_path, obs_i, obs_j, L, obs)

        # Plot vorticity contour
        plot_vorticity_contour(w, Nx, Ny, Lx, Ly, Re, output_path, obs_i, obs_j, L, obs)