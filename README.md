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
After roughly 24 hours on a machine comparable to NASA/Pleiades, you will have achieved a state similar to the image below:



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

Contact the exoweather team for more details.