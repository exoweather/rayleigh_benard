"""
Dedalus script for 2D Rayleigh-Benard convection.

This script uses a Fourier basis in the x direction with periodic boundary
conditions.  The equations are scaled in units of the buoyancy time (Fr = 1).

Usage:
    rayleigh_benard_onset.py [options] 

Options:
    --Rayleigh=<Rayleigh>      Rayleigh number [default: 1e6]
    --Prandtl=<Prandtl>        Prandtl number = nu/kappa [default: 1]
    --nz=<nz>                  Vertical resolution (8 yields single precision, 16 yields near double precision) [default: 8]
    --nx=<nx>                  Horizontal resolution [default: 128]
    --multiplier=<multiplier>  How much higher resolution is the verification run [default: 1.5]
    --aspect=<aspect>          Aspect ratio of problem [default: 64]
    --restart=<restart_file>   Restart from checkpoint
    --label=<label>            Optional additional case name label

    --no_slip                  Use no-slip boundary conditions
    --stress_free              Use stress-free boundary conditions
    --no_lid                   Use no-slip/stress-free boundary conditions
"""
import logging
logger = logging.getLogger(__name__)

import numpy as np
import time

from dedalus import public as de

def Rayleigh_Benard_onset(Prandtl=1, nz=32, nx=128, aspect=64, data_dir='./',
                          multiplier=1.5,
                          no_slip=False, stress_free=False, no_lid=False):
    if not no_slip and not stress_free and not no_lid:
        no_slip = True
        
    # input parameters
    logger.info(" Pr = {}".format(Prandtl))
            
    # Parameters
    Lz = 1.
    Lx = aspect*Lz

    logger.info("resolution: [{}x{}]".format(nx, nz))
    # Create bases and domain
    x_basis = de.Fourier('x',   nx, interval=(0, Lx), dealias=3/2)
    z_basis_set = []
    nz_set = [nz, int(nz*multiplier)]
    for nz_solve in nz_set:
        z_basis_set.append(de.Chebyshev('z', nz_solve, interval=(0, Lz), dealias=3/2))

    domain_set = []
    for z_basis in z_basis_set:
        domain_set.append(de.Domain([x_basis, z_basis], grid_dtype=np.float64))

    solver_set = []
    # 2D Boussinesq hydrodynamics
    for domain in domain_set:
        problem = de.EVP(domain, variables=['p','b','u','w','bz','uz','wz'], eigenvalue='Ra')
        problem.meta['p','b','u','w']['z']['dirichlet'] = True

        problem.parameters['F'] = F = 1
        problem.parameters['Pr'] = np.sqrt(Prandtl)
    
        problem.parameters['Lx'] = Lx
        problem.parameters['Lz'] = Lz
        problem.substitutions['plane_avg(A)'] = 'integ(A, "x")/Lx'
        problem.substitutions['vol_avg(A)']   = 'integ(A)/Lx/Lz'

        # for eq_type_1 = False; not working
        problem.substitutions['dt(f)'] = '(0*f)'
        problem.substitutions['P'] = '(1/sqrt(Ra)*1/sqrt(Pr))'
        problem.substitutions['R'] = '(1/sqrt(Ra)*sqrt(Pr))'            
        #problem.substitutions['scale'] = 'sqrt(Ra)' 

        eq_type_1 = True
        problem.add_equation("dx(u) + wz = 0")
        if eq_type_1:    
            problem.add_equation(" - (dx(dx(b)) + dz(bz)) - F*w          = 0")
            problem.add_equation(" - (dx(dx(u)) + dz(uz)) + dx(p)        = 0")
            problem.add_equation(" - (dx(dx(w)) + dz(wz)) + dz(p) - Ra*b = 0")
        else:
            # not working
            problem.add_equation("(dt(b) - P*(dx(dx(b)) + dz(bz)) - F*w       ) = -(u*dx(b) + w*bz)")
            problem.add_equation("(dt(u) - R*(dx(dx(u)) + dz(uz)) + dx(p)     ) = -(u*dx(u) + w*uz)")
            problem.add_equation("(dt(w) - R*(dx(dx(w)) + dz(wz)) + dz(p) - b ) = -(u*dx(w) + w*wz)")
            
        problem.add_equation("bz - dz(b) = 0")
        problem.add_equation("uz - dz(u) = 0")
        problem.add_equation("wz - dz(w) = 0")
        problem.add_bc("left(b) = 0")
        if no_slip:
            problem.add_bc("left(u) = 0")
            problem.add_bc("right(u) = 0")
            logger.info("using no_slip BCs")
        elif stress_free:
            problem.add_bc("left(uz) = 0")
            problem.add_bc("right(uz) = 0")
            logger.info("using stress_free BCs")
        elif no_lid:
            problem.add_bc("left(u) = 0")
            problem.add_bc("right(uz) = 0")
            logger.info("using stress_free BCs")
            
        problem.add_bc("left(w) = 0")
        problem.add_bc("right(b) = 0")
        problem.add_bc("right(w) = 0", condition="(nx != 0)")
        problem.add_bc("integ(p, 'z') = 0", condition="(nx == 0)")
    
        solver_set.append(problem.build_solver())
        
    logger.info('Solver built')
    
    # Main loop
    try:
        logger.info('Starting loop')
        start_time = time.time()
        crit_Ra_set = []
        min_wavenumber = 1
        max_wavenumber = int(nx/2)
        first_output = True
        for wave in np.arange(min_wavenumber, max_wavenumber):
            low_e_val_set = []
            for solver in solver_set:
                solver.solve(solver.pencils[wave])
                eigenvalue_indices = np.argsort(np.abs(solver.eigenvalues))
                eigenvalues = np.copy(solver.eigenvalues[eigenvalue_indices])
                low_e_val_set.append(eigenvalues[0])
            x_grid = solver.domain.grid(0)
            
            if np.isfinite(low_e_val_set[0]):
                if first_output:
                    print("k_h                      Ra_1           Ra_2            relative error")
                    print("                       (nz={:4d})      (nz={:4d})    |Ra_1 - Ra_2|/|Ra_1|".format(nz_set[0], nz_set[1]))
                    first_output = False
                    
                print("{:4d}:{:12.4g}   {:>12.4g}   {:>12.4g}   {:8.3g}".format(wave, wave/Lx, low_e_val_set[0], low_e_val_set[1],
                                                                                np.abs(np.abs(low_e_val_set[0]-low_e_val_set[1])/low_e_val_set[0])))
                crit_Ra_set.append(low_e_val_set[1])
            else:
                print(wave, "no finite values")
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise
    finally:
        i_min_Ra = np.argmin(crit_Ra_set)
        logger.info("Minimum Ra = {:g} at wavenumber={:g} (# {:d})".format(crit_Ra_set[i_min_Ra], (i_min_Ra + min_wavenumber)/Lx, i_min_Ra + min_wavenumber))
        end_time = time.time()
        logger.info('Run time: %.2f sec' %(end_time-start_time))
        #logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))
        
if __name__ == "__main__":
    from docopt import docopt
    args = docopt(__doc__)

    import sys
    # save data in directory named after script
    data_dir = sys.argv[0].split('.py')[0]
    data_dir += "_Ra{}_Pr{}_a{}".format(args['--Rayleigh'], args['--Prandtl'], args['--aspect'])
    if args['--label'] is not None:
        data_dir += "_{}".format(args['--label'])
    data_dir += '/'
    #logger.info("saving run in: {}".format(data_dir))

    if args['--nx'] is not None:
        nx = int(args['--nx'])
    else:
        nx = None
        
    Rayleigh_Benard_onset(
                    Prandtl=float(args['--Prandtl']),
                    aspect=int(args['--aspect']),
                    nz=int(args['--nz']),
                    nx=nx,
                    data_dir=data_dir,
                    multiplier=float(args['--multiplier']),
                    no_slip=args['--no_slip'],
                    stress_free=args['--stress_free'],
                    no_lid=args['--no_lid'])
    

