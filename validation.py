"""
@Author: Matheus Teixeira de Sousa (mtsousa14@gmail.com)

Simulate a newtonian fluid flow over a lid-driven cavity
"""

from utils.utils import calculate_u_star_exp, calculate_u_star_imp, calculate_v_star_exp, calculate_v_star_imp
from utils.utils import calculate_pressure, calculate_new_u, calculate_new_v, calculate_psi, utils_compiler
from utils.plot import plot_psi_contour, plot_psi_stream
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

    if not error:

        # Define matrices
        u = np.zeros((Nx+1, Ny+2), float)
        v = np.zeros((Nx+2, Ny+1), float)

        pressure = np.zeros((Nx+2, Ny+2), float)

        psi = np.zeros((Nx+1, Ny+1), float)

        # Set final time
        tf = float(args.tf) # 60.0

        # Set u component initial condition
        U = np.zeros(Nx+1, float)
        U[:] = 1
        u[:, Ny] = 2*U[:]

        u_star = np.copy(u)
        v_star = np.copy(v)

        # Call explicit functions
        if not args.implicit:
            output_path = f'Re_{str(Re)}_exp'
            for k in tqdm(range(int(tf/dt)), desc ="Iterations", position=0, leave=True):
                u_star = calculate_u_star_exp(u, v, Nx, Ny, dx, dy, dt, Re, u_star, U)
                v_star = calculate_v_star_exp(u, v, Nx, Ny, dx, dy, dt, Re, v_star)
                pressure = calculate_pressure(u_star, v_star, Nx, Ny, dx, dy, dt, tol, pressure)
                u = calculate_new_u(u_star, pressure, Nx, Ny, dx, dt, u)
                v = calculate_new_v(v_star, pressure, Nx, Ny, dy, dt, v)  
        # Call implicit functions
        else:
            output_path = f'Re_{str(Re)}_imp'
            for k in tqdm(range(int(tf/dt)), desc ="Iterations", position=0, leave=True):
                u_star = calculate_u_star_imp(u, v, Nx, Ny, dx, dy, dt, Re, tol, u_star, U)
                v_star = calculate_v_star_imp(u, v, Nx, Ny, dx, dy, dt, Re, tol, v_star)
                pressure = calculate_pressure(u_star, v_star, Nx, Ny, dx, dy, dt, tol, pressure)
                u = calculate_new_u(u_star, pressure, Nx, Ny, dx, dt, u)
                v = calculate_new_v(v_star, pressure, Nx, Ny, dy, dt, v)

        tol = 1.e-8
        psi = calculate_psi(u, v, Nx, Ny, dx, dy, dt, tol, psi)
        print('psi_max:', -1*np.amin(psi))

        # Save final matrices
        if not exists('data/' + output_path):
            makedirs('data/' + output_path)

        np.save('data/' + output_path + '/u.npy', u)
        np.save('data/' + output_path + '/v.npy', v)
        np.save('data/' + output_path + '/stream.npy', psi)
        np.save('data/' + output_path + '/pressure.npy', pressure)

        # Plot stream function contour
        plot_psi_contour(Nx, Ny, Lx, Ly, psi, output_path)

        # Plot stream function stream plot
        plot_psi_stream(u, v, Nx, Ny, Lx, Ly, output_path)