import numpy as np
from codes.helper_functions import get_ints

class chain:
    def __init__(self, filepath, label,
                 color=None,
                 idx_choice=[0,1,6,7,8]):
        self.chain = np.loadtxt(filepath)
        self.color = color
        self.label = label
        self.qnts = get_ints(self.chain)
        self.chain_toplot = self.chain[:,idx_choice]
        print(f'finished loading "{self.label}"')