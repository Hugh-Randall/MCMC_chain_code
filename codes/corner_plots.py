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
                usecolors=False, figsize=None, title=None, ksys_prior=True,
                labelfsize=None, legendfsize=None):
    
    parameter_defaults = pd.DataFrame.from_dict(chains[0].parameter_defaults, orient='index')
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

    chains_toplot = [c.chain_array[:,par_index] for c in chains]
    if usecolors:
        ...
        # Check that all chains have an assigned color, otherwise assign one to those without
        ...

        colors_keys = [c.color for c in chains]
    else:
        colors_keys = list(colors.keys())[:len(chains)]
    colors_gtc = [colors[key]['gtc_color'] for key in colors_keys]
    colors_text = [colors[key]['text_color'] for key in colors_keys]

    if figsize == None:
        figsize = 10 * 0.95**len(params)
        figsize = 10
    
    if labelfsize == None:
        fsize_plot = (15) * 0.95**len(params) 
    else: 
        fsize_plot = labelfsize
    if legendfsize == None:
        fsize_legend = (15-1*len(chains))#*(5./len(params))
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
            GTC.axes[axIDs[axi]].text(0.06,1.05+0.29*i*(0.894**len(chains)) ,plot_labels[axi]+r' = ${'+\
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