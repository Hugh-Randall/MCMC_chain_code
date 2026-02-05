import numpy as np
from codes.helper_functions import *
import yaml

class chain:
    def __init__(self, filepath, label,
                 color=None,
                 idx_choice=[0,1,6,7,8]):
        self.chain_array = np.loadtxt(filepath)
        self.color = color
        self.label = label
        self.qnts = get_ints(self.chain_array)
        
        fname_meta = chain_meta_fname(filepath)
        with open(fname_meta, 'r') as f:
            meta = yaml.safe_load(f)
        for key in meta.keys():
            setattr(self, key, meta[key])
        print(f'finished loading "{self.label}"')