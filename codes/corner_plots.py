# Imports
# numpy for array handling
import numpy as np
import pandas as pd

# os for path handling
import os

from codes.helper_functions import gtc_ax_ids

# Plt settings
import matplotlib.pyplot as plt
from pathlib import Path
module_path = Path(__file__).parent
plt.style.use( str(module_path) + '/config/mystyle.mplstyle')

# For the corner plots
import pygtc
from scipy.stats import norm
import yaml

from matplotlib.patches import Patch
from matplotlib.lines import Line2D


# Functions

colors = yaml.safe_load(open(str(module_path) + '/config/colors.yaml'))

def make_corner(chains, params=None,
                savefig=False, figdir='./figures/', outfile=None, 
                usecolors=False, figsize=None, title=None, ksys_prior=False,
                labelfsize=None, legendfsize=None, return_fig=False):
    """
    Makes a corner plot for an arbitrary number of parameters given a list of chain objects. 
    It automatically decides the number of axes given the number of parameters to be plotted
    and handles indexing under the hood. Font sizes and spacing are automatically scaled to 
    look good based on the number of parameters as well as the number of chains, though many
    things can still be adjusted manually with one of the optional arguments. 

    Parameters
    ----------
    chains : list
        List of initialized chain objects. They must hold their array data (no lazy_load). 
    params : list, optional
        List of strings used to define which POIs will be plotted. If not specified, it will
        plot all possible parameters by default. The names of the parameters given must match 
        what are used in the MathModel.parameter_defaults. You can use chain.show_params()
        to print the parameter options.
    savefig : bool, optional
        If True the figure will be saved. In this case, 'outfile' must be specified. Defaults to False.
    figdir : str, optional
        String specifying the output directory of the figure if it will be saved. Defaults to './figures/'
    outfile : str, optional
        Output filename of the figure if it is saved. Must be specified if 'savefig' is set to True.
    usecolors : bool, optional
        If True, the chain objects must have a color specified, and those colors will be used when plotting.
        If False, colors are assigned according to the order of the chain objects and the order in
        '/codes/config/colors.yaml'.
    figsize : int, optional
        Can be specified to manually set figure size. By default, figure size is set automatically.
    title : str, optional
        The title that will be printed at the top of the figure. If it is too long or too many chains
        are plotted at the same time, then the title may overlap with quoted measurements. For now the only 
        work-around is to pad manually pad the title with whitespace. 
    ksys_prior : bool, optional
        Deprecated. 
    labelfsize : int, optional
        Value used to define the font size of the measurments displayed over the top axes. By default
        the size is scaled automatically.
    legendfsize : int, optional
        Value used to define the font size of the legend. By default the size is scaled automatically.
    return_fig : bool, optional
        If True, the figure object will be returned and further plot elements may be added. Default is False.
    """
    parameter_defaults = pd.DataFrame.from_dict(chains[0].parameter_info, orient='index')
    parameter_defaults_keys = list(parameter_defaults.index)

    if params:
        pass
    else:
        params = parameter_defaults_keys

    par_index = [parameter_defaults_keys.index(x) for x in params]   
    N = len(par_index)
    axIDs = gtc_ax_ids(N)
    decs = list(parameter_defaults.loc[params]['num_decimals'])
    plot_labels = list(parameter_defaults.loc[params]['plot_label'])
    units = list(parameter_defaults.loc[params]['unit'])

    # Decide if I want truths later
    # truths = (0, None, None, None, None)

    if chains[0].chain_array.ndim >1:
        chains_toplot = [c.chain_array[:,par_index] for c in chains]
    else:
        chains_toplot = [c.chain_array.reshape((len(c.chain_array), 1)) for c in chains]
    if usecolors:
        allowed_colors = list(colors.keys())
        chain_colors = [c.color for c in chains]
        taken_colors = [x for x in chain_colors if x is not None]
        leftover_colors = [x for x in allowed_colors if x not in taken_colors]
        
        colors_keys = []
        i = 0
        for c in chains:
            if c.color is not None:
                colors_keys.append(c.color)
            else:
                colors_keys.append(leftover_colors[i])
                i += 1
    else:
        colors_keys = list(colors.keys())[:len(chains)]
    colors_gtc = [colors[key]['gtc_color'] for key in colors_keys]
    colors_text = [colors[key]['text_color'] for key in colors_keys]

    num_params = len(params)
    num_chains = len(chains)

    if figsize == None:
        # figsize = 10 * 0.95**len(params)
        # figsize = 10
        figsize = 10 * (num_params/5.)**0.5
    
    if labelfsize == None:
        fsize_plot = (15) * 0.95**len(params) 
        # fsize_plot = (15) / (num_chains**0.4)
    else: 
        fsize_plot = labelfsize
        
    if legendfsize == None:
        fsize_legend = (15-1*len(chains))#*(5./len(params))
        # fsize_legend = (15) / (num_chains**0.4)

    else:
        fsize_legend = legendfsize

    GTC = pygtc.plotGTC(chains=chains_toplot,paramNames=plot_labels,
                        # truths = truths, 
                        figureSize=figsize,nContourLevels=2,
                        customTickFont={'family':'serif','size':15 - np.abs(5-len(params))},
                        # customTickFont={'family':'serif','size':15*(5./len(params))},
                        customLabelFont={'family':'serif','size':15},
                        # customLabelFont={'family':'serif','size':15 - 0.5*np.abs(5-len(params))},
                        customLegendFont={'family':'serif','size':fsize_legend},
                        colorsOrder=colors_gtc, chainLabels=None,
                        legendMarker='None',
                        nBins=100,smoothingKernel=3)
    
    # Get text scaling values:
    text = GTC.axes[axIDs[0]].text(0.75, 1.05, 'test', transform=GTC.axes[axIDs[0]].transAxes)
    GTC.canvas.draw()

    bb = text.get_window_extent().transformed(GTC.axes[axIDs[0]].transAxes.inverted())
    dy = bb._points[1,1]-bb._points[0,1]
    buffer_size = 0.3*dy
    text.remove()

    for axi in range(len(axIDs)):
        for i in range(len(chains)):
            c = chains[i]
            qnts = c.qnts
            # GTC.axes[axIDs[axi]].text(0.06,1.05+0.29*i*(0.894**len(chains)) ,plot_labels[axi]+r' = ${'+\
            #                       str(np.round(qnts[par_index[axi]][1],decimals=decs[axi]))+\
            #                       '}_{-'+str(np.round(qnts[par_index[axi]][2],decimals=decs[axi]))+\
            #                       '}^{+'+str(np.round(qnts[par_index[axi]][0],decimals=decs[axi]))+'}$'+units[axi],
            #                       transform=GTC.axes[axIDs[axi]].transAxes,color=colors_text[i],
            #                       fontweight='bold',fontsize=fsize_plot)
            GTC.axes[axIDs[axi]].text(0.06, 1 + (buffer_size+dy)*i + buffer_size, plot_labels[axi]+r' = ${'+\
                                  str(np.round(qnts[par_index[axi]][1],decimals=decs[axi]))+\
                                  '}_{-'+str(np.round(qnts[par_index[axi]][2],decimals=decs[axi]))+\
                                  '}^{+'+str(np.round(qnts[par_index[axi]][0],decimals=decs[axi]))+'}$'+units[axi],
                                  transform=GTC.axes[axIDs[axi]].transAxes,color=colors_text[i],
                                  fontweight='bold',fontsize=fsize_plot)
    if ksys_prior:
        ...
        # Need to adjust the axis indexing for when the number of parameters plotted changes. 
        # x_axis = np.linspace(GTC.axes[axIDs[2]].get_xlim()[0],GTC.axes[axIDs[2]].get_xlim()[1], 50)
        # GTC.axes[axIDs[2]].plot(x_axis, norm.pdf(x_axis, 0, 10),lw=1,ls='--',color='k')
        
        # x_axis = np.linspace(GTC.axes[axIDs[3]].get_xlim()[0],GTC.axes[axIDs[3]].get_xlim()[1], 50)
        # GTC.axes[axIDs[3]].plot(x_axis, norm.pdf(x_axis, 0, 10),lw=1,ls='--',color='k')
        
        # x_axis = np.linspace(GTC.axes[axIDs[4]].get_xlim()[0],GTC.axes[axIDs[4]].get_xlim()[1], 50)
        # GTC.axes[axIDs[4]].plot(x_axis, norm.pdf(x_axis, 0, 10),lw=1,ls='--',color='k')
    
    legend_elements = [Patch(facecolor=colors_text[chains.index(c)], label=c.label) for c in chains][::-1]
    
    GTC.axes[axIDs[0]].legend(handles = legend_elements, 
                          loc='upper right',
                          bbox_to_anchor=(0.85, 0.85),
                          bbox_transform=GTC.transFigure,
                          fontsize=fsize_legend)
    
    if title is not None:
        plt.suptitle(title, x=0.52)

    if savefig:
        if outfile==None:
            raise Exception('Figure will be saved but outfile was not specified!')
        if not os.path.exists(figdir):
            os.mkdir(figdir)
        plt.savefig(figdir+outfile, dpi=300)
    plt.show()

    if return_fig:
        return GTC