"""
Dedalus script for 2D Rayleigh-Benard convection.

This script uses a Fourier basis in the x direction with periodic boundary
conditions.  The equations are scaled in units of the buoyancy time (Fr = 1).

Usage:
    rayleigh_benard.py [options] 

Options:
    --Rayleigh=<Rayleigh>      Rayleigh number [default: 1e6]
    --Prandtl=<Prandtl>        Prandtl number = nu/kappa [default: 1]
    --nz=<nz>                  Vertical resolution [default: 128]
    --nx=<nx>                  Horizontal resolution; if not set, nx=aspect*nz_cz
    --aspect=<aspect>          Aspect ratio of problem [default: 4]
    --restart=<restart_file>   Restart from checkpoint
    --label=<label>            Optional additional case name label
    
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

def Rayleigh_Benard(Rayleigh=1e6, Prandtl=1, nz=64, nx=None, aspect=4, restart=None, data_dir='./'):
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
    z_basis = de.Chebyshev('z', nz, interval=(0, Lz), dealias=3/2)
    domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)

    if domain.distributor.rank == 0:
        import os
        if not os.path.exists('{:s}/'.format(data_dir)):
            os.mkdir('{:s}/'.format(data_dir))

    # 2D Boussinesq hydrodynamics
    problem = de.IVP(domain, variables=['p','b','u','w','bz','uz','wz'])
    problem.meta['p','b','u','w']['z']['dirichlet'] = True

    problem.parameters['P'] = (Rayleigh * Prandtl)**(-1/2)
    problem.parameters['R'] = (Rayleigh / Prandtl)**(-1/2)
    problem.parameters['F'] = F = 1
    
    problem.parameters['Lx'] = Lx
    problem.parameters['Lz'] = Lz
    problem.substitutions['plane_avg(A)'] = 'integ(A, "x")/Lx'
    problem.substitutions['vol_avg(A)']   = 'integ(A)/Lx/Lz'
    
    problem.substitutions['enstrophy'] = '(dx(w) - uz)**2'
    problem.substitutions['vorticity'] = '(dx(w) - uz)' 

    problem.add_equation("dx(u) + wz = 0")
    problem.add_equation("dt(b) - P*(dx(dx(b)) + dz(bz))             = -(u*dx(b) + w*bz)")
    problem.add_equation("dt(u) - R*(dx(dx(u)) + dz(uz)) + dx(p)     = -(u*dx(u) + w*uz)")
    problem.add_equation("dt(w) - R*(dx(dx(w)) + dz(wz)) + dz(p) - b = -(u*dx(w) + w*wz)")
    problem.add_equation("bz - dz(b) = 0")
    problem.add_equation("uz - dz(u) = 0")
    problem.add_equation("wz - dz(w) = 0")
    problem.add_bc("left(b) = left(-F*z)")
    problem.add_bc("left(uz) = 0")
    problem.add_bc("left(w) = 0")
    problem.add_bc("right(b) = right(-F*z)")
    problem.add_bc("right(uz) = 0")
    problem.add_bc("right(w) = 0", condition="(nx != 0)")
    problem.add_bc("integ(p, 'z') = 0", condition="(nx == 0)")

    # Build solver
    #ts = de.timesteppers.RK222
    #cfl_safety = 1
    #cfl_safety = 1.5 # works; rings
    #cfl_safety = 1.6 # verge of breaking
    #cfl_safety = 1.75 # breaks

    ts = de.timesteppers.RK443
    cfl_safety = 1 #1.5
    #cfl_safety = 2 # works; rings
    
    solver = problem.build_solver(ts)
    logger.info('Solver built')

    # Checkpointing
    if checkpointing:
        checkpoint = Checkpoint(data_dir)
        checkpoint.set_checkpoint(solver, wall_dt=1800)
    
    # Initial conditions
    x = domain.grid(0)
    z = domain.grid(1)
    b = solver.state['b']
    bz = solver.state['bz']

    # Random perturbations, initialized globally for same results in parallel
    noise = global_noise(domain, scale=1, frac=0.25)

    if restart is None:
        # Linear background + perturbations damped at walls
        zb, zt = z_basis.interval
        pert =  1e-3 * noise * (zt - z) * (z - zb)
        b['g'] = -F*(z - pert)
        b.differentiate('z', out=bz)
    else:
        logger.info("restarting from {}".format(restart))
        checkpoint.restart(restart, solver)
        
    # Integration parameters
    solver.stop_sim_time = np.inf
    solver.stop_wall_time = 144. * 3600.
    solver.stop_iteration = np.inf

    # Analysis
    analysis_tasks = []
    snapshots = solver.evaluator.add_file_handler(data_dir+'slices', sim_dt=0.1, max_writes=10)
    snapshots.add_task("b")
    snapshots.add_task("b - plane_avg(b)", name="b'")
    snapshots.add_task("enstrophy")
    snapshots.add_task("vorticity")
    analysis_tasks.append(snapshots)

    coeffs = solver.evaluator.add_file_handler(data_dir+'coeffs', sim_dt=0.1, max_writes=10)
    coeffs.add_task("b", layout='c')
    coeffs.add_task("b - plane_avg(b)", name="b'", layout='c')
    coeffs.add_task("w", layout='c')
    coeffs.add_task("u", layout='c')
    coeffs.add_task("enstrophy", layout='c')
    coeffs.add_task("vorticity", layout='c')
    analysis_tasks.append(coeffs)

    profiles = solver.evaluator.add_file_handler(data_dir+'profiles', sim_dt=0.1, max_writes=10)
    profiles.add_task("plane_avg(b)", name="b")
    profiles.add_task("plane_avg(u)", name="u")
    profiles.add_task("plane_avg(enstrophy)", name="enstrophy")
    profiles.add_task("plane_avg(w*b) / P - plane_avg(bz)", name="Nu")
    analysis_tasks.append(profiles)

    scalar = solver.evaluator.add_file_handler(data_dir+'scalar', sim_dt=0.1, max_writes=10)
    scalar.add_task("vol_avg(b)", name="IE")
    scalar.add_task("0.5*vol_avg(u*u+w*w)", name="KE")
    scalar.add_task("0.5*vol_avg(u*u)", name="KE_x")
    scalar.add_task("0.5*vol_avg(w*w)", name="KE_z")
    scalar.add_task("0.5*vol_avg(u*u+w*w)-vol_avg(b)", name="TE")
    scalar.add_task("vol_avg(plane_avg(u)**2)", name="u_avg")
    scalar.add_task("vol_avg((u - plane_avg(u))**2)", name="u1")
    scalar.add_task("(vol_avg(w*b) / P)+1.", name="Nu")
    analysis_tasks.append(scalar)

    # workaround for issue #29
    problem.namespace['enstrophy'].store_last = True

    # CFL
    CFL = flow_tools.CFL(solver, initial_dt=0.1, cadence=1, safety=cfl_safety,
                         max_change=1.5, min_change=0.5, max_dt=0.1, threshold=0.1)
    CFL.add_velocities(('u', 'w'))

    # Flow properties
    flow = flow_tools.GlobalFlowProperty(solver, cadence=1)
    flow.add_property("sqrt(u*u + w*w) / R", name='Re')

    # Main loop
    try:
        logger.info('Starting loop')
        start_time = time.time()
        while solver.ok:
            dt = CFL.compute_dt()
            solver.step(dt) #, trim=True)
            log_string =  'Iteration: {:5d}, '.format(solver.iteration)
            log_string += 'Time: {:8.3e}, dt: {:8.3e}, '.format(solver.sim_time, dt)
            log_string += 'Re: {:8.3e}/{:8.3e}'.format(flow.grid_average('Re'), flow.max('Re'))
            logger.info(log_string)
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise
    finally:
        end_time = time.time()
        logger.info('Iterations: %i' %solver.iteration)
        logger.info('Sim end time: %f' %solver.sim_time)
        logger.info('Run time: %.2f sec' %(end_time-start_time))
        logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))
        
        logger.info('beginning join operation')
        if checkpointing:
            logger.info(data_dir+'/checkpoint/')
            post.merge_analysis(data_dir+'/checkpoint/')

        for task in analysis_tasks:
            logger.info(task.base_path)
            post.merge_analysis(task.base_path)
        
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
                    data_dir=data_dir)
    

