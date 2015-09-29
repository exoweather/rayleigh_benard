# README #

These are basic Rayleigh-Benard convection problems, executed
with the [Dedalus](http://dedalus-project.org) pseudospectral
framework.  To run these problems, first install
[Dedalus](http://dedalus-project.org/) (and on
[bitbucket](https://bitbucket.org/dedalus-project/dedalus)).
These scripts are extensions of the scripts included in the default dedalus installation.


Once [Dedalus](http://dedalus-project.org/) is installed and activated, do the following:
```
#!bash
mpirun -np 1024 python3 rayleigh_benard.py --nz=2048 --Rayleigh=1e10 --aspect=2
```
After roughly 24 hours on a machine comparable to NASA/Pleiades, you will have achieved a state similar to the temperature image below:

![snapshot_000227_reduced_temperature.png](https://bitbucket.org/repo/jjgd8z/images/4120742230-snapshot_000227_reduced_temperature.png)

The accompanying enstrophy snapshot, at same time, suggests that we are well resolved at this resolution and Rayleigh number:

![snapshot_000227_reduced_enstrophy.png](https://bitbucket.org/repo/jjgd8z/images/1823301896-snapshot_000227_reduced_enstrophy.png)

These images were produced using the included `plot_results_parallel.py` script, via:
```
#!bash
mpirun -np 30 python3 plot_results_parallel.py rayleigh_benard_Ra1e10_Pr1_a2
```
When used on different cases, the case name must be replaced and the number of cores must be manually adjusted to agree with the number of outputs.  Before the plotting script can be executed, the distributed analysis files must be joined.  If the `rayleigh_benard.py` script completes before the wall time is exceeded, the script will attempt to join the data.  If this fails, the data can be joined in post-processing using `merge.py` or similar tools.

At lower Rayleigh number (Ra=1e9), the resolution can be lowered:
```
#!bash
mpirun -np 512 python3 rayleigh_benard.py --nz=1024 --Rayleigh=1e9 --aspect=2
```

We are using docopt in these general driver cases, and
```
#!bash
python3 FC_multi.py --help
```
will describe various command-line options.

Contact the [exoweather](http://exoweather.org/) team for more details.