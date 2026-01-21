# Imports
# numpy for array handling
import numpy as np

# Plt settings
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('lines', lw = 3)
plt.rc('xtick', top = True, direction = 'in')
plt.rc('xtick.minor', visible = True)
plt.rc('font', size = 16)
plt.rc('axes', labelsize = 16)
plt.rc('axes', titlesize = 22)
plt.rc('xtick', labelsize = 12)
plt.rc('ytick', labelsize = 12)
plt.rc('ytick', right = True, direction = 'in')
plt.rc('ytick.minor', visible = True)
plt.rc('savefig', bbox = 'tight')

# For the corner plots
import pygtc
from scipy.stats import norm
import yaml

from matplotlib.patches import Patch
from matplotlib.lines import Line2D


# Functions

colors = yaml.safe_load(open('./codes/colors.yaml'))

def make_corner(chains, savefig=False, figdir='./figures/', outfile=None, usecolors=False, figsize=10, title=None, ksys_prior=True,
                labelfsize=None, legendfsize=None, tired=False):

    labels_base = [r'$f_{\mathrm{NL}}$',r'$b_{1g}$',r'$K_{\mathrm{DEC}}$', r'$K_{\mathrm{SGC}}$', r'$K_{\mathrm{MZLS}}$']
    labels_tick = [r'$f_{\mathrm{NL}}$',r'$b_{1g}$',r'$K_{\mathrm{DEC}}$', r'$K_{\mathrm{SGC}}$', r'$K_{\mathrm{MZLS}}$']
    
    truths = (0, None, None, None, None)

    chains_toplot = [c.chain_toplot for c in chains]
    if usecolors:
        colors_keys = [c.color for c in chains]
    else:
        colors_keys = list(colors.keys())[:len(chains)]
    colors_gtc = [colors[key]['gtc_color'] for key in colors_keys]
    colors_text = [colors[key]['text_color'] for key in colors_keys]
    
    axIDs = [10,11,12,13,14]
    par_index = [0,1,6,7,8]
    decs = [0,2,2,2,2,2,2,2]
    perc = ['', '', r'[$\%$]', r'[$\%$]',r'[$\%$]']

    if labelfsize == None:
        fsize_plot = 15-1*len(chains)
    else: 
        fsize_plot = labelfsize
    if legendfsize == None:
        fsize_legend = 15-1*len(chains)
    else:
        fsize_legend = legendfsize

    GTC = pygtc.plotGTC(chains=chains_toplot,paramNames=labels_tick, truths = truths, figureSize=figsize,nContourLevels=2,
                    customTickFont={'family':'serif','size':15},customLabelFont={'family':'serif','size':labelfsize},
                    customLegendFont={'family':'serif','size':legendfsize},
                    colorsOrder=colors_gtc, chainLabels=None,
                    legendMarker='None',
                    nBins=100,smoothingKernel=3)
    
    for axi in range(len(axIDs)):
        for i in range(len(chains)):
            c = chains[i]
            qnts = c.qnts
            if tired:
                GTC.axes[axIDs[axi]].text(0.06,1.05+0.29*i*(0.894**len(chains)),labels_base[axi]+r' = ${'+\
                                      str(np.round(qnts[par_index[axi]][1],decimals=decs[axi]))+\
                                      '}_{-'+str(np.round(qnts[par_index[axi]][2],decimals=decs[axi]))+\
                                      '}^{+'+str(np.round(qnts[par_index[axi]][0],decimals=decs[axi]))+'}$'+perc[axi],
                                      transform=GTC.axes[axIDs[axi]].transAxes,color=colors_text[i],
                                      fontweight='bold',fontsize=fsize_plot)
            else:
                GTC.axes[axIDs[axi]].text(0.06,1.1+0.25*i*(0.894**len(chains)),labels_base[axi]+r' = ${'+\
                                          str(np.round(qnts[par_index[axi]][1],decimals=decs[axi]))+\
                                          '}_{-'+str(np.round(qnts[par_index[axi]][2],decimals=decs[axi]))+\
                                          '}^{+'+str(np.round(qnts[par_index[axi]][0],decimals=decs[axi]))+'}$'+perc[axi],
                                          transform=GTC.axes[axIDs[axi]].transAxes,color=colors_text[i],
                                          fontweight='bold',fontsize=fsize_plot)
    if ksys_prior:
        x_axis = np.linspace(GTC.axes[axIDs[2]].get_xlim()[0],GTC.axes[axIDs[2]].get_xlim()[1], 50)
        GTC.axes[axIDs[2]].plot(x_axis, norm.pdf(x_axis, 0, 10),lw=1,ls='--',color='k')
        
        x_axis = np.linspace(GTC.axes[axIDs[3]].get_xlim()[0],GTC.axes[axIDs[3]].get_xlim()[1], 50)
        GTC.axes[axIDs[3]].plot(x_axis, norm.pdf(x_axis, 0, 10),lw=1,ls='--',color='k')
        
        x_axis = np.linspace(GTC.axes[axIDs[4]].get_xlim()[0],GTC.axes[axIDs[4]].get_xlim()[1], 50)
        GTC.axes[axIDs[4]].plot(x_axis, norm.pdf(x_axis, 0, 10),lw=1,ls='--',color='k')
    
    legend_elements = [Patch(facecolor=colors_text[chains.index(c)], label=c.label) for c in chains][::-1]
    
    GTC.axes[10].legend(handles = legend_elements, 
                        bbox_to_anchor =[4.5, 0.2], 
                        fontsize=fsize_legend)
    if title is not None:
        plt.suptitle(title, x=0.52)
    if savefig:
        if outfile==None:
            raise Exception('Figure will be saved but outfile was not specified!')
        plt.savefig(figdir+outfile, dpi=300)
    plt.show()