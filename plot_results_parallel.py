import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as ticker
import shelve
import time
import os
import sys
import brewer2mpl
import h5py
from mpi4py import MPI

comm_world = MPI.COMM_WORLD
rank = comm_world.rank
size = comm_world.size

# helper conversion function for string input via sys.argv
def num(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            print("Problem parsing ",s)
            return s


# directory name
if len(sys.argv) > 1:
    data_dir = sys.argv[1]
else:
    data_dir = '.'

if len(sys.argv) > 2:
    data_prefix = sys.argv[2]
else:
    data_prefix = 'slices'
    

#first iteration
if len(sys.argv) > 3:
    iteration = num(sys.argv[3])
    startup_report_string = "opening files from {:s} starting with {:d}".format(data_dir, iteration)
else:
    #restartfile = 'unified_data'
    iteration = 1
    startup_report_string = "opening files from {:s} starting with {:d}".format(data_dir, iteration)

# cadence of iterations
if len(sys.argv) > 4:
    iter_step = num(sys.argv[4])
    startup_report_string += " at cadence {:d}".format(iter_step)
else:
    iter_step = 1
  
# number of iterations
if len(sys.argv) > 5:
    n_iter = num(sys.argv[5])
    startup_report_string += " repeating {:d} times".format(n_iter)

else:
    n_iter = size # number of mpi processes

if size>n_iter:
    raise NameError("Number of processors must be <= n_iter")

if n_iter % size != 0:
    raise NameError("Number of processors must divide n_iter")

n_iter = int(n_iter/size)
iteration = iteration + int(n_iter*rank*iter_step)

name = 'snapshot'
startup_report_string += " and writing to {:s}".format(name)
startup_report_string += ", starting at iter {:d}".format(iteration) 
 
print()
print(startup_report_string)
print()




#plt.ion()
# Options
fnames = ["b","enstrophy"]
xstr = 'x/H'
ystr = 'z/H'
cmapname = 'Spectral_r'
color_map = [('RdYlBu', 'diverging', 11), ('BuPu', 'sequential', 9)]

reverse_color_map = [True, True, True, True]
float_this_scale = [False, False, False, False]

reverse_color_map = [True, True]
float_this_scale = [False, False]

even_scale = False
units = True
static_scale = True
sliding_average = False
box_size = 30
true_aspect_ratio = True
vertical_stack = True
scale_late = True
add_background_s0 = False

log_list = [] #['enstrophy'] #fnames[1]

single = False

def load(var, iteration=None):

  root_dir = '{:s}/{:s}'.format(data_dir,data_prefix)
  filename = '{:s}_s{:d}.h5'.format(data_prefix,iteration)

  f = h5py.File("{:s}/".format(root_dir)+filename, flag='r')

  x = np.array(f['scales']['x']['1.0'][:])
  y = np.array(f['scales']['z']['1.0'][:])
  t = np.array(f['scales']['sim_time'][:])
  writes = np.array(f['scales']['write_number'][:])
  variable = np.array(f['tasks'][var])
  f.close()
      
  shape = variable.shape
  print(shape)
  #variable = variable.reshape(shape[0]*shape[1],shape[2],shape[3])

  return (variable, x, y, t, writes)

def background_s0(z):
    Lz = 106
    gamma = 5./3.
    epsilon = 1e-4
    z0 = Lz + 1
    return -epsilon/gamma*(np.log(z0)-np.log(z0-z))

def read_fields(iteration=None):
    fields = []

    for fn in fnames:
        var,x,y,t,writes = load(fn, iteration=iteration)

        print(var.shape)
        print(np.min(var))
        print(np.max(var))
        if fn in log_list:
            if np.min(var) == 0:
                var[var.nonzero()] = np.log10(var[var.nonzero()])
                var[np.where(var == 0)]=np.min(var[var.nonzero()])
                print("taking log10 of non-zero elements of ", fn)
                print("and setting zero elements to ", np.min(var[var.nonzero()]))
            elif np.min(var) < 0:
                var = np.log10(np.abs(var))
                print("taking log10 of |", fn,"|")
            else:
                var = np.log10(var)
                print("taking log10 of ", fn)

        fields.append(var)
        
    return fields, x, y, t, writes

if scale_late:
    # scale by mid-run of movie values
    first_read_iteration = iteration+int(iter_step*(n_iter-1))
else:
    first_read_iteration = iteration

print("scaling iteration:", first_read_iteration)
fields, x, y, t, writes = read_fields(iteration=first_read_iteration)

print("times: ",t)

# Storage
images = []
image_axes = []
cbar_axes = []

# Determine grid size
if vertical_stack:
  nrows = len(fields)
  ncols = 1
else:
  nrows = 1
  ncols = len(fields)

# Setup spacing [top, bottom, left, right] and [height, width]
t_mar, b_mar, l_mar, r_mar = (0.2, 0.2, 0.2, 0.2)
t_pad, b_pad, l_pad, r_pad = (0.15, 0.03, 0.03, 0.03)
h_cbar, w_cbar = (0.05, 1.)

domain_width = np.max(x)-np.min(x)
domain_height = np.max(y)-np.min(y)
if true_aspect_ratio:
  h_data, w_data = (1., domain_width/domain_height)
else:
  h_data, w_data = (1., 1.)




h_im = t_pad + h_cbar + h_data + b_pad
w_im = l_pad + w_data + r_pad
h_total = t_mar + nrows * h_im + b_mar
w_total = l_mar + ncols * w_im + r_mar
scale = 3.0

print("figure size is {:g}x{:g}".format(scale * w_total, scale * h_total))

# Create figure and axes
fig = plt.figure(1, figsize=(scale * w_total,
                             scale * h_total))
row = 0
cindex = 0

for j, (fname, field) in enumerate(zip(fnames, fields)):

    left = (l_mar + w_im * cindex + l_pad) / w_total
    bottom = 1 - (t_mar + h_im * (row + 1) - b_pad) / h_total
    width = w_data / w_total
    height = h_data / h_total
    image_axes.append(fig.add_axes([left, bottom, width, height]))
    image_axes[j].lastrow = (row == nrows - 1)
    image_axes[j].firstcol = (cindex == 0)

    left = (l_mar + w_im * cindex + l_pad) / w_total
    bottom = 1 - (t_mar + h_im * row + t_pad + h_cbar) / h_total
    width = w_cbar / w_total
    height = h_cbar / h_total
    cbar_axes.append(fig.add_axes([left, bottom, width, height]))

    cindex+=1
    if cindex%ncols == 0:
        # wrap around and start the next row
        row += 1
        cindex = 0
    
# Title
height = 1 - (0.6 * t_mar) / h_total
timestring = fig.suptitle(r'', y=height, size=16)


def create_limits_mesh(x, y):
    xd = np.diff(x)
    yd = np.diff(y)
    shape = x.shape
    xm = np.zeros((y.size+1, x.size+1))
    ym = np.zeros((y.size+1, x.size+1))
    xm[:, 0] = x[0] - xd[0] / 2.
    xm[:, 1:-1] = x[:-1] + xd / 2.
    xm[:, -1] = x[-1] + xd[-1] / 2.
    ym[0, :] = y[0] - yd[0] / 2.
    ym[1:-1, :] = (y[:-1] + yd / 2.)[:, None]
    ym[-1, :] = y[-1] + yd[-1] / 2.

    return xm, ym


def add_image(fig, imax, cbax, x, y, data, cmap):

    if units:
        xm, ym = create_limits_mesh(x, y)

        im = imax.pcolormesh(xm, ym, data, cmap=cmap, zorder=1)
        plot_extent = [xm.min(), xm.max(), ym.min(), ym.max()]
        imax.axis(plot_extent)
    else:
        im = imax.imshow(data, zorder=1, aspect='auto',
                         interpolation='none', origin='lower',
                         cmap=cmap)
        shape = data.shape
        plot_extent = [-0.5, shape[1] - 0.5, -0.5, shape[0] - 0.5]
        imax.axis(plot_extent)

    cb = fig.colorbar(im, cax=cbax, orientation='horizontal',
                      ticks=ticker.MaxNLocator(nbins=5, prune='both'))

    cb.formatter.set_powerlimits((4, 3))
    sci_not_loc = cb.formatter.get_offset()
    cb.update_ticks()
    return im

def percent_trim(field, percent_cut=0.03):
    if isinstance(percent_cut, list):
        if len(percent_cut) > 1:
            low_percent_cut  = percent_cut[0]
            high_percent_cut = percent_cut[1]
        else:
            low_percent_cut  = percent_cut[0]
            high_percent_cut = percent_cut[0]
    else:
        low_percent_cut  = percent_cut
        high_percent_cut = percent_cut
        
    # trimming method from Ben's ASH analysis package
    sorted_field = np.sort(field, axis=None)
    N_elements = len(sorted_field)
    min_value = sorted_field[low_percent_cut*N_elements]
    max_value = sorted_field[(1-high_percent_cut)*N_elements-1]
    return min_value, max_value

def set_scale(field, fixed_lim=None, even_scale=True, percent_cut=0.03):
    if fixed_lim is None:
        if even_scale:
            image_min, image_max = percent_trim(field, percent_cut=percent_cut)
            if np.abs(image_min) > image_max:
                image_max = np.abs(image_min)
            elif image_min < 0:
                image_min = -np.abs(image_max)
        else:
            image_min, image_max = percent_trim(field, percent_cut=percent_cut)
    else:
        image_min = fixed_lim[0]
        image_max = fixed_lim[1]

    return image_min, image_max
      
def update_image(im, data, float_this_scale=False, fixed_lim=None, even_scale=True):

    if units:
        im.set_array(np.ravel(data))
    else:
        im.set_data(data)

    if not static_scale or float_this_scale:
        image_min, image_max = set_scale(field, fixed_lim=fixed_lim, even_scale=even_scale)
        images[j].set_clim(image_min, image_max)

def add_labels(imax, cbax, fname):

    # Title
    title = imax.set_title('%s' %fname, size=14)
    title.set_y(1.1)

    # Colorbar
    cbax.xaxis.set_ticks_position('top')
    plt.setp(cbax.get_xticklabels(), size=10)

    if imax.lastrow:
        imax.set_xlabel(xstr, size=12)
        plt.setp(imax.get_xticklabels(), size=10)
    else:
        plt.setp(imax.get_xticklabels(), visible=False)

    if imax.firstcol:
        imax.set_ylabel(ystr, size=12)
        plt.setp(imax.get_yticklabels(), size=10)
    else:
        plt.setp(imax.get_yticklabels(), visible=False)



# Set up images and labels
for j, (fname, field) in enumerate(zip(fnames, fields)):
    imax = image_axes[j]
    cbax = cbar_axes[j]

    #cmap = matplotlib.cm.get_cmap(cmapname)
    #cmap.set_bad('0.7')
    cmap = brewer2mpl.get_map(*color_map[j], reverse=reverse_color_map[j]).mpl_colormap

    images.append(add_image(fig, imax, cbax, x, y, field[0].T, cmap))
    if fname in log_list:
        add_labels(imax, cbax, 'log10 '+fname)
    else:
        add_labels(imax, cbax, fname)

    if static_scale:
        if fname in log_list:
            static_min, static_max = set_scale(field, even_scale=False, percent_cut=[0.4, 0.0])
        else:
            # center on zero
            static_min, static_max = set_scale(field, even_scale=even_scale, percent_cut=0.1)

        if scale_late:
            static_min = comm_world.scatter([static_min]*size,root = size-1)
            static_max = comm_world.scatter([static_max]*size,root = size-1)
        else:
            static_min = comm_world.scatter([static_min]*size,root = 0)
            static_max = comm_world.scatter([static_max]*size,root = 0)

        images[j].set_clim(static_min, static_max)
        print(fname, ": +- ", -static_min, static_max)


first_iteration = iteration        
# plot images        
print(x.shape)  
dpi_png = max(150, len(x)/(w_total*scale))


print("dpi:", dpi_png, " -> ", w_total*scale*dpi_png, "x",h_total*scale*dpi_png)
for i_iter in range(n_iter):

    iteration = first_iteration + i_iter*iter_step

    if n_iter > 1:
        if i_iter > 0 or scale_late:
            fields, x, y, t, writes = read_fields(iteration=iteration)

    # note, this assumes that all files have the same number of frames.
    # this breaks if one file (like the last one) is smaller in size.
    #i_fig = t.size*(rank*n_iter+i_iter)

    for i in range(t.size):
        for j, field in enumerate(fields):
            if sliding_average:
                if t.size - i > box_size:
                    i_avg_start = i
                else:
                    # we've run out of elements; fix average
                    i_avg_start = t.size - box_size
        
                i_avg_end = i_avg_start+ box_size
                if i < box_size:
                    # average forward
                    sliding_min, sliding_max = percent_trim(field[i_avg_start:i_avg_end], percent_cut=0.1)
                else:
                    # do moving average from here forwards
                    sliding_min_current, sliding_max_current = percent_trim(field[i], percent_cut=0.05)
                    sliding_min = (box_size-1)/box_size*sliding_min + 1/box_size*sliding_min_current
                    sliding_max = (box_size-1)/box_size*sliding_max + 1/box_size*sliding_max_current
                
                    print(sliding_min, sliding_max)
                    update_image(images[j], field[i].T, fixed_lim=[sliding_min, sliding_max])
            elif fnames[j] == 's_fluc':
                update_image(images[j], field[i].T, even_scale=even_scale)
            else:
                update_image(images[j], field[i].T, float_this_scale=float_this_scale[j])

        # pull the figure label from the writing order
        i_fig = writes[i]
        
        # Update time title
        tstr = 't = {:6.3e}'.format(t[i])
        timestring.set_text(tstr)

        figure_file = "{:s}/{:s}_{:06d}.png".format(data_dir,name,i_fig)
        fig.savefig(figure_file, dpi=dpi_png)
        print("writting {:s}".format(figure_file))

