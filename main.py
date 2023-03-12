"""
@Author: Matheus Teixeira de Sousa (mtsousa14@gmail.com)

Simulate ...
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
    parser.add_argument('--num_obs', default=1,
                        help="Number of obstacles")
    parser.add_argument('--obstacle', default="0,0,0",
                        help="Obstacle location (i, j) and size as 'i','j','L' for all obstacles.")
    parser.add_argument('--output',
                        help="Set the output name.")
    parser.add_argument('--early_stopping', default=False, action="store_true",
                        help="Set early stop to True to simulate until permanent situation or tf.")
    parser.add_argument('--dont_save', default=False, action="store_true",
                        help="Don't save output plots at the end of simulation.")
    parser.add_argument('--dont_show', default=False, action="store_true",
                        help="Don't show output plots at the end of simulation.")

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
        obs_i, obs_j = np.zeros((1, 1)), np.zeros((1, 1))
        inside_obs = np.zeros(obs_i.shape, bool)
        L = np.zeros((1, 1))
        obs = False
    else:
        try:
            obs_i, obs_j, L = get_obstacle_size(args.obstacle, int(args.num_obs))
            inside_obs = np.zeros(obs_i.shape, bool)
            obs = True
        except:
            error = True
            print('[ERROR] - Error during obstacles information extraction. Please, check the format indicated.', flush=True)

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
                u_star = calculate_u_star_exp(u, v, Nx, Ny, dx, dy, dt, Re, u_star, U, obs_i, obs_j, L, inside_obs, obs)
                v_star = calculate_v_star_exp(u, v, Nx, Ny, dx, dy, dt, Re, v_star, obs_i, obs_j, L, inside_obs, obs)
                pressure = calculate_pressure(u_star, v_star, Nx, Ny, dx, dy, dt, tol, pressure, obs_i, obs_j, L, inside_obs, obs)
                u = calculate_new_u(u_star, pressure, Nx, Ny, dx, dt, u, obs_i, obs_j, L, inside_obs, obs)
                v = calculate_new_v(v_star, pressure, Nx, Ny, dy, dt, v, obs_i, obs_j, L, inside_obs, obs) 
        
                if check_diff(u, v, aux_u, aux_v, tol) and args.early_stopping:
                    break

                aux_u = np.copy(u)
                aux_v = np.copy(v)
        
        # Call implicit functions
        else:
            output_path = f'Re_{str(Re)}_imp_obs' if obs else f'Re_{str(Re)}_imp'
            for k in tqdm(range(int(tf/dt)), desc ="Iterations", position=0, leave=True):
                u_star = calculate_u_star_imp(u, v, Nx, Ny, dx, dy, dt, Re, tol, u_star, U, obs_i, obs_j, L, inside_obs, obs)
                v_star = calculate_v_star_imp(u, v, Nx, Ny, dx, dy, dt, Re, tol, v_star, obs_i, obs_j, L, inside_obs, obs)
                pressure = calculate_pressure(u_star, v_star, Nx, Ny, dx, dy, dt, tol, pressure, obs_i, obs_j, L, inside_obs, obs)
                u = calculate_new_u(u_star, pressure, Nx, Ny, dx, dt, u, obs_i, obs_j, L, inside_obs, obs)
                v = calculate_new_v(v_star, pressure, Nx, Ny, dy, dt, v, obs_i, obs_j, L, inside_obs, obs)

                if check_diff(u, v, aux_u, aux_v, tol) and args.early_stopping:
                    break

                aux_u = np.copy(u)
                aux_v = np.copy(v)

        if tol > 1.e-8:
            tol = 1.e-8
        
        psi = calculate_psi(u, v, Nx, Ny, dx, dy, dt, tol, psi, obs_i, obs_j, L, inside_obs, obs)
        print('psi_max:', -1*np.amin(psi))
        w = calculate_vorticity(u, v, dx, dy, Nx, Ny, w, obs_i, obs_j, L, inside_obs, obs)

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
            plot_psi_contour(Nx, x, y, Lx, Ly, psi, output_path, obs_i, obs_j, L, obs, args.dont_show, args.dont_save)

        # Plot streamlines
        plot_psi_stream(u_plot, v_plot, x, y, Nx, Lx, Ly, output_path, obs_i, obs_j, L, obs, args.dont_show, args.dont_save)

        # Plot vorticity contour
        plot_vorticity_contour(w, Nx, Ny, Lx, Ly, Re, output_path, obs_i, obs_j, L, obs, args.dont_show, args.dont_save)