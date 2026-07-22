import numpy as np
import pandas as pd
from astropy.io import fits
# from astropy.table import Table

# def get_2pcf_idx_slice(file, s_min, s_max, s_cutwindow, terms) :
#     # Returns the idx locations for diag npcf
#     # Output is a boolean array, True for included idxs
#     s_vec = file['s']
#     s_slice = np.zeros(len(s_vec),dtype='bool')
#     if s_min==None:
#         s_min = min(s_vec)
#     if s_max==None:
#         s_max = max(s_vec)
#     if s_cutwindow is None:
#         for i in range(len(s_slice)):
#             if (s_vec[i]>=s_min)&(s_vec[i]<=s_max):
#                 s_slice[i] = True
#     else:
#         for i in range(len(s_slice)):
#             if (s_vec[i]>=s_min)&(s_vec[i]<=s_cutwindow[0])|(s_vec[i]>=s_cutwindow[1])&(s_vec[i]<=s_max):
#                 s_slice[i] = True
#     return s_slice

def get_2pcf_idx_slice(file, s_min, s_max, s_cutwindow, terms) :
    # Returns the idx locations for diag npcf
    # Output is a boolean array, True for included idxs
    # Need to update this for case where any of smax smin or scutwindow can be dicts and the others can be floats
    s_vec = file['s']
    s_slice = np.zeros(len(s_vec),dtype='bool')
    if s_min==None:
        s_min = min(s_vec)
    if s_max==None:
        s_max = max(s_vec)
        
    if (type(s_max)==dict):
        masks = []
        for term in terms:
            s_min_term = s_min
            s_max_term = s_max[term]
            s_cutwindow_term = s_cutwindow
            s_temp = s_vec[file['term']==term].copy()
            mask_term = np.zeros(len(s_temp),dtype='bool')
            if s_cutwindow_term is None:
                for i in range(len(mask_term)):
                    if (s_temp[i]>=s_min_term)&(s_temp[i]<=s_max_term):
                        mask_term[i] = True
            else:
                for i in range(len(mask_term)):
                    if (s_temp[i]>=s_min_term)&(s_temp[i]<=s_cutwindow_term[0])|(s_temp[i]>=s_cutwindow_term[1])&(s_temp[i]<=s_max_term):
                        mask_term[i] = True
            masks.append(mask_term)
        s_slice = np.concatenate(masks)
    else:
        if s_cutwindow is None:
            for i in range(len(s_slice)):
                if (s_vec[i]>=s_min)&(s_vec[i]<=s_max):
                    s_slice[i] = True
        else:
            for i in range(len(s_slice)):
                if (s_vec[i]>=s_min)&(s_vec[i]<=s_cutwindow[0])|(s_vec[i]>=s_cutwindow[1])&(s_vec[i]<=s_max):
                    s_slice[i] = True
                
    return s_slice

# def obs_unwrapper(pkg_loc):
#     # Unwraps an individual corr function to use as the observable
#     with fits.open(pkg_loc, memmap=False) as hdul:
#             pkg = hdul[1].data.copy()
#     # pkg = Table.read(pkg_loc)
#     col_names = pkg.columns.names
    
#     terms = col_names.copy()
#     terms.remove('s')

#     to_concat = []
#     for term in terms:
#         to_concat.append(pkg[term])
#     xi_obs = np.concatenate(to_concat)
#     return xi_obs, terms

def obs_unwrapper(pkg_loc):
    # Unwraps an individual corr function to use as the observable
    with fits.open(pkg_loc, memmap=False) as hdul:
            pkg = hdul[1].data.copy()   

    terms = list(dict.fromkeys(pkg['term']))
    
    xi_obs = np.asarray(pkg['obs'])
    return xi_obs, terms

def Omega_m_z(z: float, Om_m0: float):
    # Case with no curvature!
    Om_L0 = 1-Om_m0
    return Om_m0*(z+1)**3/(Om_m0*(z+1)**3+Om_L0)
    
def gz(z: float, Om_m0: float):
    Om_mz = Omega_m_z(z,Om_m0)
    Om_Lz = 1-Om_mz
    return (5*Om_mz/2)*(Om_mz**0.55-Om_Lz+(1+Om_mz/2)*(1+Om_Lz/70))**(-1)

def Dz_norm(z: float, Om_m0: float):
    return gz(z,Om_m0 = Om_m0)/(gz(0,Om_m0 = Om_m0)*(1+z))
    
def get_ints(chain):
    ints = []
    for i in range(len(chain.T)):
        qnts = np.quantile(chain.T[i],(0.16,0.5,0.84))
        ints.append( (qnts[2]-qnts[1],qnts[1],qnts[1]-qnts[0]) )
    return ints

# def quad_2pars(x,a,b):
#     return a*x**2+b*x

def concatenate_quadfits(files):
    c1 = np.array([])
    c2 = np.array([])
    for f in files:
        df = pd.read_csv(f)
        c1_temp = df['c1']
        c2_temp = df['c2']
        c1 = np.concatenate([c1, c1_temp])
        c2 = np.concatenate([c2, c2_temp])
    return c1, c2

def concatenate_fits(files):
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
    df_out = pd.concat(dfs)
    return df_out.to_numpy()

def reorder_fits(df, terms):
    df['term'] = pd.Categorical(df['term'], categories=terms, ordered=True)
    return df.sort_values(['term', 's']).reset_index(drop=True)

def chain_meta_fname(fname):
    return fname.split('.txt')[0]+'.meta.yaml'

def gtc_ax_ids(N):
    num_axs = int((N*(N+1))/2)
    axids = list(range(num_axs))
    axids = axids[-N:]
    return axids

def hartlap_factor(num_data, num_mocks):
    # to be applied to inverse covariance matrix
    return (num_mocks-num_data-2)/(num_mocks-1)
    
def percival_factor(num_data, num_mocks, num_params):
    # The inverse of this must be applied to the inverse covariance matrix
    A = 2/((num_mocks-num_data-1)*(num_mocks-num_data-4))
    B = (num_mocks-num_data-2)/((num_mocks-num_data-1)*(num_mocks-num_data-4))
    return (1+B*(num_data-num_params))/(1 + A + B*(num_params+1))