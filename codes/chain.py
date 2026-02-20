import numpy as np
from codes.helper_functions import *
import yaml

class chain:
    def __init__(self, filepath, label,
                 color=None,
                 load_array=False):
        
        self.filepath = filepath            
        self.color = color
        self.label = label
        
        fname_meta = chain_meta_fname(filepath)
        with open(fname_meta, 'r') as f:
            meta = yaml.safe_load(f)
        for key in meta.keys():
            setattr(self, key, meta[key])

        if load_array:
            self.load_array()

        if not hasattr(self, 'qnts'):
            self.load_array()
            self.qnts = get_ints(self.chain_array)
        
        print(f'finished loading "{self.label}"')

    def load_array(self):
        self.chain_array = np.loadtxt(filepath)