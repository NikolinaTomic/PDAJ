import argparse
import numpy as np
import csv
from scipy.integrate import odeint
from multiprocessing import Pool

DEFAULT_RESOLUTION = 6
DEFAULT_TMAX = 30
DEFAULT_DT = 0.01

DEFAULT_L1 = 1
DEFAULT_L2 = 1
DEFAULT_M1 = 1
DEFAULT_M2 = 1

# The gravitational acceleration (m.s-2).
g = 9.81


def deriv(y, t, L1, L2, m1, m2):
    """Return the first derivatives of y = theta1, z1, theta2, z2."""
    theta1, z1, theta2, z2 = y

    c, s = np.cos(theta1-theta2), np.sin(theta1-theta2)

    theta1dot = z1
    z1dot = (m2*g*np.sin(theta2)*c - m2*s*(L1*z1**2*c + L2*z2**2) -
             (m1+m2)*g*np.sin(theta1)) / L1 / (m1 + m2*s**2)
    theta2dot = z2
    z2dot = ((m1+m2)*(L1*z1**2*s - g*np.sin(theta2) + g*np.sin(theta1)*c) +
             m2*L2*z2**2*s*c) / L2 / (m1 + m2*s**2)
    return theta1dot, z1dot, theta2dot, z2dot

def solve(parametri):
    (L1, L2, m1, m2, tmax, dt, y0, theta1_init, theta2_init)=parametri
    t = np.arange(0, tmax+dt, dt)

    # Do the numerical integration of the equations of motion
    y = odeint(deriv, y0, t, args=(L1, L2, m1, m2))
    theta1, theta2 = y[:,0], y[:,2]

    # Convert to Cartesian coordinates of the two bob positions.
    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)
    return (theta1, theta2, x1, y1, x2, y2, theta1_init, theta2_init)

def gen_simulation_model_params(theta_resolution, L1, L2, m1, m2, tmax, dt):
    search_space = np.linspace(0, 2*np.pi, theta_resolution)
    for theta1_init in search_space:
        for theta2_init in search_space:
            yield (L1,L2,m1,m2,tmax,dt,np.array([theta1_init, 0, theta2_init, 0]),theta1_init,theta2_init)

def simulate_pendulum(theta_resolution, dt=DEFAULT_DT, tmax=DEFAULT_TMAX, L1=DEFAULT_L1, L2=DEFAULT_L2, m1=DEFAULT_M1, m2=DEFAULT_M2):
    with Pool() as pool:
        results = pool.imap(solve, gen_simulation_model_params(theta_resolution,L1,L2,m1,m2,tmax,dt),1000)
        for theta1, theta2, x1, y1, x2, y2, theta1_init, theta2_init in results:
            yield theta1[-1], theta2[-1], theta1_init, theta2_init

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'results_file',
        help="Filename where the results will be stored, in CSV format"
    )
    parser.add_argument(
        '-r',
        '--resolution',
        metavar='NUM',
        type=int,
        default=DEFAULT_RESOLUTION,
        help="Resolution, %d by default" % DEFAULT_RESOLUTION
    )
    parser.add_argument(
        '--tmax',
        metavar='NUM',
        type=float,
        default=DEFAULT_TMAX,
        help="Simulation time, %f by default" % DEFAULT_TMAX
    )
    parser.add_argument(
        '--dt',
        metavar='NUM',
        type=float,
        default=DEFAULT_DT,
        help="Simulation time step, %f by default" % DEFAULT_DT
    )
    args = parser.parse_args()
    results = simulate_pendulum(
        theta_resolution=args.resolution,
        dt=args.dt,
        tmax=args.tmax,
    )
    file = args.results_file

    with open(file, 'w', newline='') as csvfile:
        fieldnames = ['theta1_init', 'theta2_init', 'theta1', 'theta2']
        spamwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        spamwriter.writeheader()
        for result in results:
            spamwriter.writerow({'theta1_init': result[2], 'theta2_init': result[3],'theta1': result[0], 'theta2': result[1]})

if __name__ == '__main__':
    main()
