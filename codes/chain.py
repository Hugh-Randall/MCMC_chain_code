import numpy as np
from codes.helper_functions import *
import yaml

class chain:
    def __init__(self, filepath, label,
                 color=None,
                 lazy_load=False):
        """
        Initializes a chain object which stores the MCMC output array as well as
        corresponding metadata. These are the objects that are used to make coner plots
        with the corner.py file.
    
        Parameters
        ----------
        filepath : str
            Full path/filename to the txt file that was saved in running PNGmodel.run_sampling. 
            Matches the string passed to fname_chain in PNGmodel.run_sampling.
        label : str
            Label that will be used to identify this chain in a corner plot legend
        color : str, optional
            The color to be used for this chain (and only this chain) in all corner plots.
            If specified, it must be one of the highest-level colors in /codes/config/colors.yaml.
            It is useful to specify if you are at the end of the analysis and want to have consistent 
            color schemes throughout the paper/presentation (differentiating LRGs and QSOs, for example)
        lazy_load : bool, optional
            Boolean to decide if the full chain array will be loaded or only the metadata. Defaults 
            to False. If True, only metadata will be loaded. This is helpful if you are only 
            interested in values of the measurements and not the distributions themselves.
            For example, if you are computing a pull test and loading many chains but only care about 
            mu and sigma, this speeds up the process and reduces the amount of data loaded to RAM. 
        """
        self.filepath = filepath            
        self.color = color
        self.label = label
        
        fname_meta = chain_meta_fname(filepath)
        with open(fname_meta, 'r') as f:
            meta = yaml.safe_load(f)
        for key in meta.keys():
            setattr(self, key, meta[key])

        if not lazy_load:
            self.load_array()

        if not hasattr(self, 'qnts'):
            self.load_array()
            self.qnts = get_ints(self.chain_array)
        
        print(f'finished loading "{self.label}"')

    def load_array(self):
        self.chain_array = np.loadtxt(self.filepath)

    def show_params(self):
        print(list(self.parameter_info.keys()))