"""
@Author: Matheus Teixeira de Sousa (mtsousa14@gmail.com)

Simulate the flow of a newtonian fluid in a lid-drive cavity with internals obstacles.
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
    parser = argparse.ArgumentParser(description='Simulate the flow of a newtonian fluid in a lid-drive cavity with internals obstacles')
    
    parser.add_argument('-re', '--num_re', required=True, type=int,
                        help="Reynolds number.")
    parser.add_argument('--final_time', required=True, type=float,
                        help="Final time to the simulation.")
    parser.add_argument('-i', '--implicit', default=False, action='store_true',
                        help="Set to use implicit method. (Default: False)")
    parser.add_argument('--grid_size', default=100, type=int,
                        help="Grid discretization. (Default: 100)")
    parser.add_argument('--dt', default=0.0001, type=float,
                        help="Time increment. (Default: 0.0001)")
    parser.add_argument('--tol', default=1.e-8, type=float,
                        help="Tolerance of the iteration error. (Default: 1.e-8)")
    parser.add_argument('-v', '--validation', default=False, action="store_true",
                        help="Set True for the validation problem. (Default: False)")
    parser.add_argument('--num_obs', default=1, type=int,
                        help="Number of obstacles. (Default: 1)")
    parser.add_argument('-obs', '--obstacle', default='40,40,20', type=str,
                        help="Obstacle location (i, j) and size as 'i','j','L' to all obstacles. (Default: 40,40,20)")
    parser.add_argument('-o', '--output', default=None,
                        help="Set the output name. (Default: None)")
    parser.add_argument('--early_stopping', default=False, action="store_true",
                        help="Set early stop to True to simulate until permanent situation or final time. (Default: False)")
    parser.add_argument('--dont_save', default=False, action="store_true",
                        help="Don't save output plots at the end of simulation. (Default: False)")
    parser.add_argument('--dont_show', default=False, action="store_true",
                        help="Don't show output plots at the end of simulation. (Default: False)")

    args = parser.parse_args()

    # Call functions to compile
    utils_compiler()

    # Define Reynolds, tolerance, N, L, dx, dy and dt
    Re = args.num_re
    tol = args.tol

    Nx, Ny = args.grid_size, args.grid_size
    Lx, Ly = 1.0, 1.0

    dx = Lx/Nx
    dy = Ly/Ny
    dt = args.dt

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
        obs_i, obs_j = np.asarray([0]), np.asarray([0])
        inside_obs = np.zeros(obs_i.shape, bool)
        L = np.asarray([0])
        obs = False
    else:
        try:
            obs_i, obs_j, L = get_obstacle_size(args.obstacle, args.num_obs)
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
        final_time = args.final_time # 60.0

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
            for k in tqdm(range(int(final_time/dt)), desc ="Iterations", position=0, leave=True):
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
            for k in tqdm(range(int(final_time/dt)), desc ="Iterations", position=0, leave=True):
                u_star = calculate_u_star_imp(u, v, Nx, Ny, dx, dy, dt, Re, tol, u_star, U, obs_i, obs_j, L, inside_obs, obs)
                v_star = calculate_v_star_imp(u, v, Nx, Ny, dx, dy, dt, Re, tol, v_star, obs_i, obs_j, L, inside_obs, obs)
                pressure = calculate_pressure(u_star, v_star, Nx, Ny, dx, dy, dt, tol, pressure, obs_i, obs_j, L, inside_obs, obs)
                u = calculate_new_u(u_star, pressure, Nx, Ny, dx, dt, u, obs_i, obs_j, L, inside_obs, obs)
                v = calculate_new_v(v_star, pressure, Nx, Ny, dy, dt, v, obs_i, obs_j, L, inside_obs, obs)

                if check_diff(u, v, aux_u, aux_v, tol) and args.early_stopping:
                    break

                aux_u = np.copy(u)
                aux_v = np.copy(v)

        # Adjust the tolerance to calculate the velocity field
        if tol > 1.e-8:
            tol = 1.e-8
        
        # Calculate the velocity field and vorticity
        psi = calculate_psi(u, v, Nx, Ny, dx, dy, tol, psi, obs_i, obs_j, L, inside_obs, obs)
        print('psi_max:', -1*np.amin(psi))
        w = calculate_vorticity(u, v, dx, dy, Nx, Ny, w, obs_i, obs_j, L, inside_obs, obs)

        if args.output != None:
            output_path = args.output
        
        # Save the final matrices
        if not exists('data/' + output_path):
            makedirs('data/' + output_path)

        np.save('data/' + output_path + '/u.npy', u)
        np.save('data/' + output_path + '/v.npy', v)
        np.save('data/' + output_path + '/stream.npy', psi)
        np.save('data/' + output_path + '/pressure.npy', pressure)
        np.save('data/' + output_path + '/vorticity.npy', w)

        u_plot, v_plot, x, y = get_matrices_plot(u, v, Nx, Ny, Lx, Ly)
        
        # Plot stream function contour
        if not obs:
            plot_psi_contour(psi, Nx, x, y, Lx, Ly, output_path, obs_i, obs_j, L, obs, args.dont_show, args.dont_save)

        # Plot streamlines
        plot_psi_stream(u_plot, v_plot, x, y, Nx, Lx, Ly, output_path, obs_i, obs_j, L, obs, args.dont_show, args.dont_save)

        # Plot vorticity contour
        plot_vorticity_contour(w, Nx, x, y, Lx, Ly, Re, output_path, obs_i, obs_j, L, obs, args.dont_show, args.dont_save)
