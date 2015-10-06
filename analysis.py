import numpy as np
import h5py

import logging
logger = logging.getLogger(__name__.split('.')[-1])

from collections import OrderedDict

class DedalusData():
    def __init__(self,  files, *args,
                 keys=None, verbose=False, **kwargs):
        
        self.verbose = verbose
        self.files = sorted(files, key=lambda x: int(x.split('.h5')[0].split('_s')[1]))
        logger.debug("opening: {}".format(self.files))
        
        if keys is None:
            self.get_keys(self.files[0], keys=keys)
        else:
            self.keys = keys
            
        self.data = OrderedDict()
        for key in self.keys:
            self.data[key] = np.array([])

        if self.verbose:
            self.report_contents(self.files[0])
            
    def get_keys(self, file, keys=None):
        f = h5py.File(file, flag='r')
        self.keys = np.copy(f['tasks'])
        f.close()
        logger.debug("tasks to study = {}".format(self.keys))

    def report_contents(self, file):
        f = h5py.File(file, flag='r')
        logger.info("Contents of {}".format(file))
        logger.info(10*'-'+' tasks '+10*'-')
        for task in f['tasks']:
            logger.info(task)
        
        logger.info(10*'-'+' scales '+10*'-')
        for key in f['scales']:
            logger.info(key)
        f.close()
        
class Scalar(DedalusData):
    def __init__(self, files, *args, keys=None, **kwargs):
        super(Scalar, self).__init__(files, *args,
                                     keys=keys, **kwargs)
        self.read_data()
            
    def read_data(self):
        self.times = np.array([])
        
        N = 1
        for filename in self.files:
            logger.debug("opening {}".format(filename))
            f = h5py.File(filename, flag='r')
            # clumsy
            for key in self.keys:
                if N == 1:
                    self.data[key] = f['tasks'][key][:]
                    logger.debug("{} shape {}".format(key, self.data[key].shape))
                else:
                    self.data[key] = np.append(self.data[key], f['tasks'][key][:], axis=0)

            N += 1
            self.times = np.append(self.times, f['scales']['sim_time'][:])
            f.close()
            
        for key in self.keys:
            self.data[key] = self.data[key][:,0,0]
            logger.debug("{} shape {}".format(key, self.data[key].shape))
            
class Profile(DedalusData):
    def __init__(self, files, *args, keys=None, **kwargs):
        super(Profile, self).__init__(files, *args,
                                      keys=keys, **kwargs)
        self.read_data()
        self.average_data()
        
    def read_data(self):

        self.times = np.array([])

        N = 1
        for filename in self.files:
            f = h5py.File(filename, flag='r')
            # clumsy
            for key in self.keys:
                if N == 1:
                    self.data[key] = f['tasks'][key][:]
                    logger.debug("{} shape {}".format(key, self.data[key].shape))
                else:
                    self.data[key] = np.append(self.data[key], f['tasks'][key][:], axis=0)

            N += 1
            # same z for all files
            self.z = f['scales']['z']['1.0'][:]
            self.times = np.append(self.times, f['scales']['sim_time'][:])
            f.close()

        for key in self.keys:
            logger.debug("{} shape {}".format(key, self.data[key].shape))

    def average_data(self):
        self.average = OrderedDict()
        self.std_dev = OrderedDict()
        for key in self.keys:
            self.average[key] = np.mean(self.data[key], axis=0)[0]
            self.std_dev[key] = np.std( self.data[key], axis=0)[0]

        for key in self.keys:
            logger.debug("{} shape {} and {}".format(key, self.average[key].shape, self.std_dev[key].shape))

class Slice(DedalusData):
    def __init__(self, files, *args, keys=None, **kwargs):
        super(Slice, self).__init__(files, *args,
                                     keys=keys, **kwargs)
        self.read_data()
            
    def read_data(self):
        self.times = np.array([])
        self.writes = np.array([], dtype=np.int)
        
        N = 1
        for filename in self.files:
            logger.debug("opening {}".format(filename))
            f = h5py.File(filename, flag='r')
            # clumsy
            for key in self.keys:
                if N == 1:
                    self.data[key] = f['tasks'][key][:]
                    logger.debug("{} shape {}".format(key, self.data[key].shape))
                else:
                    self.data[key] = np.append(self.data[key], f['tasks'][key][:], axis=0)

            N += 1
            self.x = f['scales']['x']['1.0'][:]
            self.z = f['scales']['z']['1.0'][:]

            self.times = np.append(self.times, f['scales']['sim_time'][:])
            self.writes = np.append(self.writes, f['scales']['write_number'][:])
            f.close()
            
        for key in self.keys:
            logger.debug("{} shape {}".format(key, self.data[key].shape))

class Coeff(DedalusData):
    def __init__(self, files, *args, keys=None, **kwargs):
        super(Coeff, self).__init__(files, *args,
                                     keys=keys, **kwargs)
        self.read_data()
        self.compute_power_spectrum()
        
    def read_data(self):
        self.times = np.array([])
        self.writes = np.array([], dtype=np.int)
        
        N = 1
        for filename in self.files:
            logger.debug("opening {}".format(filename))
            f = h5py.File(filename, flag='r')
            # clumsy
            for key in self.keys:
                if N == 1:
                    self.data[key] = f['tasks'][key][:]
                    logger.debug("{} shape {}".format(key, self.data[key].shape))
                else:
                    self.data[key] = np.append(self.data[key], f['tasks'][key][:], axis=0)

            N += 1
            self.kx = f['scales']['kx'][:]
            try:
                # single basis
                self.kz = f['scales']['Tz'][:]
            except:
                # single compound basis
                self.kz = f['scales']['(T,T)z'][:]

            self.times = np.append(self.times, f['scales']['sim_time'][:])
            self.writes = np.append(self.writes, f['scales']['write_number'][:])
            f.close()
            
        for key in self.keys:
            logger.debug("{} shape {}".format(key, self.data[key].shape))
            
    def compute_power_spectrum(self):
        self.power_spectrum = OrderedDict()
        for key in self.keys:
            self.power_spectrum[key] = np.real(self.data[key]*np.conj(self.data[key]))

class APJSingleColumnFigure():
    def __init__(self, aspect_ratio=None, lineplot=True, fontsize=8):
        import scipy.constants as scpconst
        import matplotlib.pyplot as plt

        self.plt = plt
        
        if aspect_ratio is None:
            self.aspect_ratio = scpconst.golden
        else:
            self.aspect_ratio = aspect_ratio

        if lineplot:
            self.dpi = 600
        else:
            self.dpi = 300
        
        self.fontsize=fontsize

        self.figure()
        self.add_subplot()
        self.set_fontsize(fontsize=fontsize)

    def figure(self):
            
        x_size = 3.5 # width of single column in inches
        y_size = x_size/self.aspect_ratio

        self.fig = self.plt.figure(figsize=(x_size, y_size))

    def add_subplot(self):
        self.ax = self.fig.add_subplot(1,1,1)

    def savefig(self, filename, dpi=None, **kwargs):
        if dpi is None:
            dpi = self.dpi

        self.plt.tight_layout(pad=0.25)
        self.fig.savefig(filename, dpi=dpi, **kwargs)

    def set_fontsize(self, fontsize=None):
        if fontsize is None:
            fontsize = self.fontsize

        for item in ([self.ax.title, self.ax.xaxis.label, self.ax.yaxis.label] +
             self.ax.get_xticklabels() + self.ax.get_yticklabels()):
            item.set_fontsize(fontsize)

    def legend(self, title=None, fontsize=None, **kwargs):
        if fontsize is None:
            self.legend_fontsize = apjfig.fontsize
        else:
            self.legend_fontsize = fontsize

        self.legend_object = self.ax.legend(prop={'size':self.legend_fontsize}, **kwargs)
        if title is not None:
            self.legend_object.set_title(title=title, prop={'size':self.legend_fontsize})

        return self.legend_object

def semilogy_posneg(ax, x, y, color=None,  color_pos=None, color_neg=None, **kwargs):
    pos_mask = np.logical_not(y>0)
    neg_mask = np.logical_not(y<0)
    pos_line = np.ma.MaskedArray(y, pos_mask)
    neg_line = np.ma.MaskedArray(y, neg_mask)

    if color is None:
        color = next(ax._get_lines.color_cycle)

    if color_pos is None:
        color_pos = color

    if color_neg is None:
        color_neg = color
        
    ax.semilogy(x, pos_line, color=color_pos, **kwargs)
    ax.semilogy(x, np.abs(neg_line), color=color_neg, linestyle='dashed')

def cheby_newton_root(z, f, z0=None, degree=512):
    import numpy.polynomial.chebyshev as npcheb
    import scipy.optimize as scpop
    
    Lz = np.max(z)-np.min(z) 
    if z0 is None:
        z0 = Lz/2

    def to_x(z, Lz):
        # convert back to [-1,1]
        return (2/Lz)*z-1

    def to_z(x, Lz):
        # convert back from [-1,1]
        return (x+1)*Lz/2 
    
    logger.info("searching for roots starting from z={}".format(z0))
    x  = to_x(z, Lz) 
    x0 = to_x(z0, Lz)
    cheb_coeffs = npcheb.chebfit(x, f, degree)
    cheb_interp = npcheb.Chebyshev(cheb_coeffs)
    cheb_der    = npcheb.chebder(cheb_coeffs)
    
    
    def newton_func(x_newton):
        return npcheb.chebval(x_newton, cheb_coeffs)

    def newton_derivative_func(x_newton):
         return npcheb.chebval(x_newton, cheb_der)
     
    try:
        x_root = scpop.newton(newton_func, x0, fprime=newton_derivative_func, tol=1e-10)
        z_root = to_z(x_root, Lz)
    except:
        logger.info("error in root find")
        x_root = np.nan
        z_root = np.nan

    logger.info("newton: found root z={} (x0:{} -> {})".format(z_root, x0, x_root))

    for x0 in x:
        print(x0, newton_func(x0))
    a = Lz/4
    b = Lz*3/4
    logger.info("bisecting between z=[{},{}] (x=[{},{}])".format(a, b, to_x(a, Lz), to_x(b, Lz)))
    logger.info("f(a) = {}  and f(b) = {}".format(newton_func(to_x(a, Lz)), newton_func(to_x(b, Lz))))
    x_root_2 = scpop.bisect(newton_func, to_x(a, Lz), to_x(b, Lz))
    z_root_2 = to_z(x_root_2, Lz)
    logger.info("bisect: found root z={} (x={})".format(z_root_2, x_root_2))

    return z_root_2

def interp_newton_root(z, f, z0=None, a=None, b=None):
    import scipy.optimize as scpop
    import scipy.interpolate as scpint
    
    Lz = np.max(z)-np.min(z) 
    if z0 is None:
        z0 = Lz/2
    
    logger.info("searching for roots starting from z={}".format(z0))

    int_f = scpint.interp1d(z, f)

    def newton_func(x_newton):
        return int_f(x_newton)

    #try:
    #    z_root = scpop.newton(newton_func, x0, tol=1e-10)
    #except:
    #    logger.info("error in root find")
    #    z_root = np.nan

    #logger.info("newton: found root z={} (z0:{})".format(z_root, z0))

    # root find with bisect; this is working more robustly.
    if a is None:
        a = Lz/4
    if b is None:
        b = Lz*3/4
        
    logger.info("bisecting between z=[{},{}]".format(a, b))
    logger.info("f(a) = {}  and f(b) = {}".format(newton_func(a), newton_func(b)))
    try:
        z_root_2 = scpop.bisect(newton_func, a, b)
    except:
        try:
            logger.info("f(a/2) = {}  and f(b) = {}".format(newton_func(a/2), newton_func(b)))
            z_root_2 = scpop.bisect(newton_func, a/2, b)
        except:
            try:
                logger.info("f(a/10) = {}  and f(b) = {}".format(newton_func(a/10), newton_func(b)))
                z_root_2 = scpop.bisect(newton_func, a/10, b)
            except:
                z_root_2 = np.nan

    logger.info("bisect: found root z={}".format(z_root_2))
    z_root = z_root_2
        
    return z_root
