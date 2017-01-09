"""
Plot slices from joint analysis files.

Usage:
    plot_powspec.py join <base_path>
    plot_powspec.py <files>... [options]

Options:
    --output=<output>         Output directory; if blank a guess based on likely case name will be made
    --fields=<fields>         Comma separated list of fields to plot [default: b,w]
"""
import numpy as np
import analysis

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
        plt.style.use('ggplot')
except:
        print("Upgrade matplotlib; for now we're falling back to old plot styles")

import logging
logger = logging.getLogger(__name__.split('.')[-1])

import plot_slices

def main(files, fields, output_path='./', output_name='powspec',
         static_scale=True):
    
    from mpi4py import MPI

    comm_world = MPI.COMM_WORLD
    rank = comm_world.rank
    size = comm_world.size
    
    data = analysis.Coeff(files, power=True)
    kz = data.kz/np.max(data.kz)
    kx = data.kx/np.max(np.abs(data.kx))
    logger.info("min max kx {} {}; shape: {}".format(min(kx), max(kx), kx.shape))
    logger.info("min max kz {} {}; shape: {}".format(min(kz), max(kz), kz.shape))


    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    
    # select down to the data you wish to plot
    data_list = []
    color_dict = {}
    for field in fields:
        logger.info(data.power_spectrum[field].shape)
        data_list.append(np.log10(data.power_spectrum[field][0,:]))
        color_dict[field] = next(ax1._get_lines.color_cycle)

    logger.info("making line plots")
    for i, time in enumerate(data.times):
        ax1.cla()
        ax2.cla()
        for field in fields:
            field_label = '$\mathrm{'+'{:s}'.format(field)+'}^*\mathrm{'+'{:s}'.format(field)+'}$'
            ax1.loglog(data.kz, np.mean(data.power_spectrum[field][i,:], axis=0),
                       label=field_label, color=color_dict[field])
            ax2.loglog(data.kx, np.mean(data.power_spectrum[field][i,:], axis=1),
                       label=field_label, color=color_dict[field])

        ax1.set_xlabel('Tz')
        ax2.set_xlabel('kx')
        ax1.set_ylabel(r'$\langle \mathrm{power}\rangle_{\mathrm{kx}}$')
        ax2.set_ylabel(r'$\langle \mathrm{power}\rangle_{\mathrm{Tz}}$')
        ax1.legend(loc='upper right')
        ax2.legend(loc='upper right')
        
        i_fig = data.writes[i]
        fig.savefig('{:s}/{:s}_tz_{:06d}'.format(output_path, output_name,i_fig), dpi=600)
        
    plt.close(fig)

    logger.info("making imagestack")
    imagestack = plot_slices.ImageStack(kx, kz, data_list, fields, xstr='kx/max(kx)', ystr='Tz/Nz')

    scale_late = True
    if static_scale:
        for i, image in enumerate(imagestack.images):
            static_min, static_max = image.get_scale(data_list[i], percent_cut=0.1)
            print(static_min, static_max)
            if scale_late:
                static_min = comm_world.scatter([static_min]*size,root = size-1)
                static_max = comm_world.scatter([static_max]*size,root = size-1)
            else:
                static_min = comm_world.scatter([static_min]*size,root = 0)
                static_max = comm_world.scatter([static_max]*size,root = 0)
            print("post comm: {}--{}".format(static_min, static_max))
            image.set_scale(static_min, static_max)

    for i, time in enumerate(data.times):
        current_data = []
        for field in fields:
            current_data.append(np.log10(data.power_spectrum[field][i,:]))
        
        imagestack.update(current_data)
                       
        i_fig = data.writes[i]
        # Update time title
        tstr = 't = {:6.3e}'.format(time)
        imagestack.timestring.set_text(tstr)
        imagestack.write(output_path, output_name, i_fig)
        
    imagestack.close()


if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    args = docopt(__doc__)
    if args['join']:
        post.merge_analysis(args['<base_path>'])
    else:
        if args['--output'] is not None:
            output_path = pathlib.Path(args['--output']).absolute()
        else:
            data_dir = args['<files>'][0].split('/')[0]
            data_dir += '/'
            output_path = pathlib.Path(data_dir).absolute()
        # Create output directory if needed
        with Sync() as sync:
            if sync.comm.rank == 0:
                if not output_path.exists():
                    output_path.mkdir()
        fields = args['--fields'].split(',')
        logger.info("plotting for {}".format(fields))
        logger.info("output to {}".format(output_path))
        
        def accumulate_files(filename,start,count,file_list):
            if start==0:
                file_list.append(filename)
        file_list = []
        post.visit_writes(args['<files>'],  accumulate_files, file_list=file_list)
            
        if len(file_list) > 0:
            main(file_list, fields, output_path=str(output_path)+'/')


