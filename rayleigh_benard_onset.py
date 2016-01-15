"""
Dedalus script for 2D Rayleigh-Benard convection.

This script uses a Fourier basis in the x direction with periodic boundary
conditions.  The equations are scaled in units of the buoyancy time (Fr = 1).

Usage:
    rayleigh_benard_onset.py [options] 

Options:
    --Rayleigh=<Rayleigh>      Rayleigh number [default: 1e6]
    --Prandtl=<Prandtl>        Prandtl number = nu/kappa [default: 1]
    --nz=<nz>                  Vertical resolution [default: 128]
    --nx=<nx>                  Horizontal resolution; if not set, nx=aspect*nz_cz
    --aspect=<aspect>          Aspect ratio of problem [default: 4]
    --restart=<restart_file>   Restart from checkpoint
    --label=<label>            Optional additional case name label

    --no_slip                  Use no-slip boundary conditions
    --stress_free              Use stress-free boundary conditions
    --no_lid                   Use no-slip/stress-free boundary conditions
"""
import logging
logger = logging.getLogger(__name__)

import numpy as np
from mpi4py import MPI
import time

from dedalus import public as de
from dedalus.extras import flow_tools
from dedalus.tools  import post
try:
    from dedalus.extras.checkpointing import Checkpoint
    checkpointing = True
except:
    logger.info("No checkpointing available; disabling capability")
    checkpointing = False
    
def global_noise(domain, seed=42, scale=None, **kwargs):            
    # Random perturbations, initialized globally for same results in parallel
    gshape = domain.dist.grid_layout.global_shape(scales=domain.dealias)
    slices = domain.dist.grid_layout.slices(scales=domain.dealias)
    rand = np.random.RandomState(seed=seed)
    noise = rand.standard_normal(gshape)[slices]

    # filter in k-space
    noise_field = domain.new_field()
    noise_field.set_scales(domain.dealias, keep_data=False)
    noise_field['g'] = noise
    filter_field(noise_field, **kwargs)
    if scale is not None:
        noise_field.set_scales(scale, keep_data=True)
        
    return noise_field['g']

def filter_field(field,frac=0.5):
    logger.info("filtering field with frac={}".format(frac))
    dom = field.domain
    local_slice = dom.dist.coeff_layout.slices(scales=dom.dealias)
    coeff = []
    for i in range(dom.dim)[::-1]:
        coeff.append(np.linspace(0,1,dom.global_coeff_shape[i],endpoint=False))
    cc = np.meshgrid(*coeff)

    field_filter = np.zeros(dom.local_coeff_shape,dtype='bool')

    for i in range(dom.dim):
        field_filter = field_filter | (cc[i][local_slice] > frac)
    field['c'][field_filter] = 0j

def Rayleigh_Benard(Rayleigh=1e6, Prandtl=1, nz=64, nx=None, aspect=4, restart=None, data_dir='./',
                    no_slip=False, stress_free=False, no_lid=False):
    if not no_slip and not stress_free and not no_lid:
        no_slip = True
        
    # input parameters
    logger.info("Ra = {}, Pr = {}".format(Rayleigh, Prandtl))
            
    # Parameters
    Lz = 1.
    Lx = aspect*Lz

    if nx is None:
        nx = int(nz*aspect)

    logger.info("resolution: [{}x{}]".format(nx, nz))
    # Create bases and domain
    x_basis = de.Fourier('x',   nx, interval=(0, Lx), dealias=3/2)
    z_basis_set = []
    nz_set = [nz, int(nz*3/2)]
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
    
        problem.add_equation("dx(u) + wz = 0")
        problem.add_equation(" - (dx(dx(b)) + dz(bz)) - F*w          = 0")
        problem.add_equation(" - (dx(dx(u)) + dz(uz)) + dx(p)        = 0")
        problem.add_equation(" - (dx(dx(w)) + dz(wz)) + dz(p) - Ra*b = 0")
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
        max_wavenumber = 32
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
                    print("k_h      Ra_1             Ra_2            relative error")
                    print("       (nz={:4d})        (nz={:4d})    |Ra_1 - Ra_2|/|Ra_1|".format(nz_set[0], nz_set[1]))
                    first_output = False
                    
                print("{:12.4g}   {:>12.4g}   {:>12.4g}   {:8.3g}".format(wave/Lx, low_e_val_set[0], low_e_val_set[1],
                                                    np.abs(np.abs(low_e_val_set[0]-low_e_val_set[1])/low_e_val_set[0])))
                crit_Ra_set.append(low_e_val_set[1])
            else:
                print(wave, "no finite values")
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise
    finally:
        i_min_Ra = np.argmin(crit_Ra_set)
        logger.info("Minimum Ra = {:g} at wavenumber={:d}".format(crit_Ra_set[i_min_Ra], i_min_Ra + min_wavenumber))
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
    logger.info("saving run in: {}".format(data_dir))

    if args['--nx'] is not None:
        nx = int(args['--nx'])
    else:
        nx = None
        
    Rayleigh_Benard(Rayleigh=float(args['--Rayleigh']),
                    Prandtl=float(args['--Prandtl']),
                    restart=(args['--restart']),
                    aspect=int(args['--aspect']),
                    nz=int(args['--nz']),
                    nx=nx,
                    data_dir=data_dir,
                    no_slip=args['--no_slip'],
                    stress_free=args['--stress_free'],
                    no_lid=args['--no_lid'])
    

