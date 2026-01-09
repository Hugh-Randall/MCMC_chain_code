# =========================================================================== #

# INPUTS

# numpy for array handling
import numpy as np
import pandas as pd
import sys

# astropy fits for file storage
from astropy.io import fits

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

# MCMC Sampler
import emcee

# Curvefit for photo vary fits
from scipy.optimize import curve_fit

# =========================================================================== #
# =========================================================================== #

# HELPER ROUTINES


def get_2pcf_idx_slice(file, s_min, s_max, s_cutwindow) :
    # Returns the idx locations for diag npcf
    # Output is a boolean array, True for included idxs
    s_vec = file[1].data['s']
    s_slice = np.zeros(len(s_vec),dtype='bool')
    if s_min==None:
        s_min = min(s_vec)
    if s_max==None:
        s_max = max(s_vec)
    if s_cutwindow is None:
        for i in range(len(s_slice)):
            if (s_vec[i]>=s_min)&(s_vec[i]<=s_max):
                s_slice[i] = True
    else:
        for i in range(len(s_slice)):
            if (s_vec[i]>=s_min)&(s_vec[i]<=s_cutwindow[0])|(s_vec[i]>=s_cutwindow[1])&(s_vec[i]<=s_max):
                s_slice[i] = True
    return s_slice

def obs_unwrapper(pkg_loc,mod):
    # Unwraps an individual corr function to use as the observable
    pkg = fits.open(pkg_loc)
    xi0 = pkg[1].data['xi0']
    xi2 = pkg[1].data['xi2']
    xi4 = pkg[1].data['xi4']
    xi_obs = np.concatenate([xi0,xi2,xi4])[mod.mask]
    return xi_obs

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

def quad_2pars(x,a,b):
    return a*x**2+b*x

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
# =========================================================================== #
# =========================================================================== #

# STATISTICAL ROUTINES

def xi_modded_base_pars(params, base_case, z_eff, Om_m0_g, z_fid, z_halo, Om_m0_h, c1_coef, c2_coef,
                        pvar_A1, pvar_B1, pvar_A2, pvar_B2, pvar_A3, pvar_B3, xi0_cond, xi2_cond, xi4_cond):
    fNL, b1g, b1h, b1g_fid, ph, pg, Psys1, Psys2, Psys3 = params
    f_g = Omega_m_z(z_eff,Om_m0_g)**0.55
    f_fid = Omega_m_z(z_fid,Om_m0_g)**0.55
    f_h = Omega_m_z(z_halo,Om_m0_h)**0.55
    Dz_g = Dz_norm(z_eff,Om_m0=Om_m0_g)
    Dz_h = Dz_norm(z_halo,Om_m0=Om_m0_h)
    
    ### Define rescale factors ######
    r_fac_fid = np.ones(len(base_case))
    r_fac_c1 = np.ones(len(base_case))
    r_fac_c2 = np.ones(len(base_case))
    
    r_fac_fid[xi0_cond] = (b1g**2 + (2/3)*b1g*f_g + (f_g**2)/5)/(b1g_fid**2 + (2/3)*b1g_fid*f_fid + (f_fid**2)/5)
    r_fac_fid[xi2_cond] = ( (4/3)*b1g*f_g + (4/7)*(f_g**2) )/( (4/3)*b1g_fid*f_fid + (4/7)*(f_fid**2) )
    r_fac_fid[xi4_cond] = (f_g/f_fid)**2

    r_fac_c1[xi0_cond] = ((b1g + f_g/3)*(b1g-pg)*(Om_m0_g/Dz_g))/\
                            ((b1h + f_h/3)*(b1h-ph)*(Om_m0_h/Dz_h))
    r_fac_c2[xi0_cond] = (((b1g-pg)**2)*(Om_m0_g/Dz_g))/(((b1h-ph)**2)*(Om_m0_h/Dz_h))
    r_fac_c1[xi2_cond] = (f_g*(b1g-pg)*(Om_m0_g/Dz_g))/(f_h*(b1h-ph)*(Om_m0_h/Dz_h))
    #################################
    c1_coef[xi4_cond] = np.zeros(len(c1_coef[xi4_cond]))
    c2_coef[~xi0_cond] = np.zeros(len(c2_coef[~xi0_cond]))
    
    fid_term = r_fac_fid*(base_case)
    PNG_term = r_fac_c1*c1_coef*fNL + r_fac_c2*c2_coef*(fNL**2)
    sys_term = r_fac_fid*((pvar_A1*Psys1**2+pvar_B1*Psys1)+(pvar_A2*Psys2**2+pvar_B2*Psys2)+(pvar_A3*Psys3**2+pvar_B3*Psys3))
    return fid_term + PNG_term + sys_term

def util_chi2_base_pars(params, obs, base_case, mod_cov, z_eff, Om_m0_g, z_fid, z_halo, Om_m0_h, c1_coef, c2_coef,
                    pvar_A1, pvar_B1, pvar_A2, pvar_B2, pvar_A3, pvar_B3, xi0_cond, xi2_cond, xi4_cond):
    # Defines chi2 given data and params
    fNL, b1g, b1h, b1g_fid, ph, pg, Psys1, Psys2, Psys3 = params
    exp = xi_modded_base_pars(params, base_case, z_eff, Om_m0_g, z_fid, z_halo, Om_m0_h, c1_coef, c2_coef,
                            pvar_A1, pvar_B1, pvar_A2, pvar_B2, pvar_A3, pvar_B3, xi0_cond, xi2_cond, xi4_cond)
    cov_inv = np.linalg.inv(mod_cov)
    return -0.5*np.matmul(np.matmul(cov_inv,(obs-exp)),(obs-exp))

def log_prior_base_pars(params, poi_hard_lims, gauss_priors, Psys1_gauss_prior, Psys2_gauss_prior, Psys3_gauss_prior):
    fNL, b1g, b1h, b1g_fid, ph, pg, Psys1, Psys2, Psys3 = params
    if poi_hard_lims[0][0] < fNL < poi_hard_lims[0][1] and \
        poi_hard_lims[1][0] < b1g < poi_hard_lims[1][1]:
        return -(Psys1-Psys1_gauss_prior[0])**2/(Psys1_gauss_prior[1])**2-\
        (Psys2-Psys2_gauss_prior[0])**2/(Psys2_gauss_prior[1])**2-\
        (Psys3-Psys3_gauss_prior[0])**2/(Psys3_gauss_prior[1])**2-\
        (b1h-gauss_priors[0][0])**2/(gauss_priors[0][1])**2-\
        (b1g_fid-gauss_priors[1][0])**2/(gauss_priors[1][1])**2-\
        (ph-gauss_priors[2][0])**2/(gauss_priors[2][1])**2-\
        (pg-gauss_priors[3][0])**2/(gauss_priors[3][1])**2
    return -np.inf

def log_probability_base_pars(params, obs, base_case,
                              mod_cov, poi_hard_lims, gauss_priors, Psys1_gauss_prior, Psys2_gauss_prior, Psys3_gauss_prior, z_eff, 
                              Om_m0_g, z_fid, z_halo, Om_m0_h, c1_coef, c2_coef, pvar_A1, pvar_B1, pvar_A2, pvar_B2, pvar_A3, pvar_B3,
                              xi0_cond, xi2_cond, xi4_cond):
    # Defines the log probability combining the likelihood and priors
    lp = 0.5*log_prior_base_pars(params, poi_hard_lims, gauss_priors, Psys1_gauss_prior, Psys2_gauss_prior, Psys3_gauss_prior)
    if not np.isfinite(lp):
        return -np.inf
    return lp + util_chi2_base_pars(params, obs,
                                    base_case, mod_cov, z_eff, Om_m0_g, z_fid, z_halo, Om_m0_h, c1_coef, c2_coef,
                                    pvar_A1, pvar_B1, pvar_A2, pvar_B2, pvar_A3, pvar_B3, xi0_cond, xi2_cond, xi4_cond)

# =========================================================================== #
# =========================================================================== #

# MODEL OBJECT

class PNGmodel:

    def __init__(self, fid_corr, cov_pkg, exclude=None, s_min=None, s_max=None, s_cutwindow=None):
        # Initializes the model based on a desired s_min/s_max
        print('Initializing...')
        # Set initial params
        self.fid_corr_file = fits.open(fid_corr)
        self.cov_file = cov_pkg
        self.s_min = s_min
        self.s_max = s_max
        
        # Print length of total observable depending on corr type
        self.s_slice = get_2pcf_idx_slice(self.fid_corr_file,self.s_min,self.s_max, s_cutwindow)
        xi0_fid = self.fid_corr_file[1].data['xi0']
        xi2_fid = self.fid_corr_file[1].data['xi2']
        xi4_fid = self.fid_corr_file[1].data['xi4']
        self.xi_fid = np.concatenate([xi0_fid,xi2_fid,xi4_fid])

        len_per_xi = len(self.s_slice)
        total_len = len(self.xi_fid)
        self.xi0_cond = np.array(len_per_xi*[True] + 2*len_per_xi*[False], dtype=bool)
        self.xi2_cond = np.array(len_per_xi*[False] + len_per_xi*[True] + len_per_xi*[False], dtype=bool)
        self.xi4_cond = np.array(2*len_per_xi*[False] + len_per_xi*[True], dtype=bool)

        # Mask out whatever will be excluded
        res = np.concatenate((self.s_slice, self.s_slice,self.s_slice))
        if exclude is not None:
            ex = {'xi0': ~self.xi0_cond,
                  'xi2': ~self.xi2_cond, 
                  'xi4': ~self.xi4_cond}
            for x in exclude:
                res = np.logical_and(res, ex[x])
        
        self.mask = np.array(res, dtype=bool)
        self.xi_fid = self.xi_fid[self.mask]
        self.xi0_cond = self.xi0_cond[self.mask]
        self.xi2_cond = self.xi2_cond[self.mask]
        self.xi4_cond = self.xi4_cond[self.mask]
        total_len = len(self.xi_fid)
        print('Observable will have {} pts'.format(total_len))
        return

    def load_PNG_model(self, png_quadfits_files):
        # Loads c1_n and c2_n coefficients
        print('Loading PNG model...')
        c1, c2 = concatenate_quadfits(png_quadfits_files)
        self.c1 = c1[self.mask]
        self.c2 = c2[self.mask]
        return
    
    def load_covariance(self,cov_rescale_factor=1.):
        # A function to load the model covariance matrix
        # Takes from the fiducial ensemble
        print('Loading covariance matrix...')
        self.cov_mat = cov_rescale_factor*np.load(self.cov_file)[self.mask][:, self.mask]
        return
    
    def load_photo_vary_fits(self, pkg_set1, pkg_set2, pkg_set3):
        print('Loading systematic weight variation...')
        self.pvar_par_B1, self.pvar_par_A1  = concatenate_quadfits(pkg_set1)
        self.pvar_par_B2, self.pvar_par_A2  = concatenate_quadfits(pkg_set2)
        self.pvar_par_B3, self.pvar_par_A3  = concatenate_quadfits(pkg_set3)
        self.pvar_par_B1, self.pvar_par_A1 = self.pvar_par_B1[self.mask], self.pvar_par_A1[self.mask]
        self.pvar_par_B2, self.pvar_par_A2 = self.pvar_par_B2[self.mask], self.pvar_par_A2[self.mask]
        self.pvar_par_B3, self.pvar_par_A3 = self.pvar_par_B3[self.mask], self.pvar_par_A3[self.mask]
        return
    
    def test_model_base_pars(self, min_type, poi_hard_lims, gauss_priors, Psys1_gauss_prior, Psys2_gauss_prior, Psys3_gauss_prior,
                             z_eff,Om_m0_g,z_fid,zhalo,Om_m0_h,
                             nwalkers, nsteps, plt_color,
                             fname = None, poi_toy = None,
                             nuiss_toy = None, Psys1_toy = None, Psys2_toy = None, Psys3_toy = None, khod_toy = None,
                             data_obs = None,
                             plt_out = False):
        print('Exploring paramter space...')
        self.z_eff = z_eff
        self.Om_m0_g = Om_m0_g
        self.z_halo = zhalo
        self.z_fid = z_fid
        self.Om_m0_h = Om_m0_h
        # self.b1_sys = b1_sys
        # self.b2_sys = b2_sys
        # self.b3_sys = b3_sys
        if min_type == 'pseudo':
            print('No!')
            # params_toy = (poi_toy[0],poi_toy[1],nuiss_toy[0],
            #               nuiss_toy[1],nuiss_toy[2],nuiss_toy[3], Psys1_toy, Psys2_toy, Psys3_toy)
            # self.obs = xi_modded_base_pars(params_toy, self.xi_fid,
            #                                self.n_vec, self.A_coeff,
            #                                self.z_eff,self.Om_m0_g,
            #                                self.z_halo,self.Om_m0_h,
            #                                self.pvar_par_A1, self.pvar_par_B1,
            #                                self.b1_sys,
            #                                self.pvar_par_A2, self.pvar_par_B2,
            #                                self.b2_sys,
            #                                self.pvar_par_A3, self.pvar_par_B3,
            #                                self.b3_sys)
        elif min_type == 'data':
            self.obs = data_obs
        # MCMC chain params
        self.poi_hard_lims = poi_hard_lims
        self.gauss_priors = gauss_priors
        self.Psys1_gauss_prior = Psys1_gauss_prior
        self.Psys2_gauss_prior = Psys2_gauss_prior
        self.Psys3_gauss_prior = Psys3_gauss_prior
        self.nwalkers = nwalkers
        self.nsteps = nsteps
        # Define and run the sampler chain
        self.sampler = emcee.EnsembleSampler(self.nwalkers,9,
                                             log_probability_base_pars,
                                             args = (self.obs,
                                                     self.xi_fid,
                                                     self.cov_mat,
                                                     self.poi_hard_lims,
                                                     self.gauss_priors,
                                                     self.Psys1_gauss_prior,
                                                     self.Psys2_gauss_prior,
                                                     self.Psys3_gauss_prior,
                                                     self.z_eff,
                                                     self.Om_m0_g,
                                                     self.z_fid,
                                                     self.z_halo,
                                                     self.Om_m0_h,
                                                     self.c1,
                                                     self.c2,
                                                     self.pvar_par_A1,
                                                     self.pvar_par_B1,
                                                     self.pvar_par_A2,
                                                     self.pvar_par_B2,
                                                     self.pvar_par_A3,
                                                     self.pvar_par_B3,
                                                     self.xi0_cond,
                                                     self.xi2_cond,
                                                     self.xi4_cond))
        # Always start the fNL walkers at 0, bias 2? All hard-coded for now
        # Bias params hard-coded to 2
        # ps hard-coded to 1
        # Rb hard-coded to 1
        # Spread 1e-4
        start_pos = np.asarray([0,1,1,1,1,1,0,0,0])+1e-4*np.random.randn(
            self.nwalkers, 9)
        self.sampler.run_mcmc(start_pos, self.nsteps, progress=True)
        if plt_out == True:
            # Plot walker output
            plt.rc('xtick', labelsize = 12)
            plt.rc('ytick', labelsize = 12)
            plt.rc('lines', lw = 1)
            fig, axes = plt.subplots(9, figsize=(12, 14), sharex=True)
            plt_samples = self.sampler.get_chain()
            plt_labels = [r'$f_{NL}$',r'$b_{1g}$',r'$b_{1h}$',
                          r'$b_{1g}^{fid}$',r'$p_h$',r'$p_g$',r'$K_{\mathrm{SYS1}}$ [\%]', r'$K_{\mathrm{SYS2}}$ [\%]', r'$K_{\mathrm{SYS3}}$ [\%]']
            for i in range(9):
                ax = axes[i]
                ax.plot(plt_samples[:, :, i], plt_color, alpha=0.3)
                ax.set_xlim(0, len(plt_samples))
                ax.set_ylabel(plt_labels[i])
                ax.yaxis.set_label_coords(-0.1, 0.5)
            axes[-1].set_xlabel("step number");
        if fname is not None:
            plt.savefig(fname)
        return
    
    def wrap_chain_base_pars(self,burn_in_steps,thinner,fname_chain):
        burn_samples = self.sampler.get_chain(
            discard=burn_in_steps,thin=thinner,flat=True)
        flat_samples = np.stack((burn_samples.T[0],
                                 burn_samples.T[1],
                                 burn_samples.T[2],
                                 burn_samples.T[3],
                                 burn_samples.T[4],
                                 burn_samples.T[5],
                                 burn_samples.T[6],
                                 burn_samples.T[7],
                                 burn_samples.T[8])).T
        np.savetxt(fname_chain, flat_samples)
        ints_chain = get_ints(flat_samples)
        labels = ['fNL', 'b1g', 'b1h', 'b1gfid', 'pg', 'ph', 'Ksys1', 'Ksys2', 'Ksys3']
        for li in range(len(labels)):
            print(labels[li]+' = '+str(np.round(ints_chain[li][1],decimals=2))+' + '+
                  str(np.round(ints_chain[li][0],decimals=2))+' - '+
                  str(np.round(ints_chain[li][2],decimals=2)))
        return
# =========================================================================== #
mod = PNGmodel(fid_corr = './inputs/abacus_averaged_fiducial_wts_fixed.fits', 
               cov_pkg = './inputs/EZ_mock_covariance_matrix.npy', 
               exclude=['xi2','xi4'],
               s_max=380, s_cutwindow = [90, 130])

# newpath = '/pscratch/sd/h/hrandall/Y1-Multipoles/png_FastPM/HOD_population/quadfits/quadxi0_linxi2/'
# mod.load_PNG_model(png_quadfits_files = [newpath + 'quadfits_LRG_FastPM_frombox_fnl_xi0.csv',
#                                          newpath + 'quadfits_LRG_FastPM_frombox_fnl_xi2.csv',
#                                          newpath + 'quadfits_LRG_FastPM_frombox_fnl_xi4.csv'])
mod.load_PNG_model(png_quadfits_files = ['./inputs/quadfits_LRG_FastPM_Y3_fnl_xi0.csv',
                                         './inputs/quadfits_LRG_FastPM_Y3_fnl_xi2.csv',
                                         './inputs/quadfits_LRG_FastPM_Y3_fnl_xi4.csv'])
# mod.load_PNG_model(png_quadfits_files = ['./analysis/quadfits_LRG_FastPM_frombox_fnl_xi0.csv',
#                                          './analysis/quadfits_LRG_FastPM_frombox_fnl_xi2.csv',
#                                          './analysis/quadfits_LRG_FastPM_frombox_fnl_xi4.csv'])

mod.load_covariance()

mod.load_photo_vary_fits(pkg_set1 = ['./inputs/quadfits_LRG_abacus_Y1_Ksys_SGC_xi0.csv',
                                     './inputs/quadfits_LRG_abacus_Y1_Ksys_SGC_xi2.csv',
                                     './inputs/quadfits_LRG_abacus_Y1_Ksys_SGC_xi4.csv'],
                         pkg_set2 = ['./inputs/quadfits_LRG_abacus_Y1_Ksys_DEC_xi0.csv',
                                     './inputs/quadfits_LRG_abacus_Y1_Ksys_DEC_xi2.csv',
                                     './inputs/quadfits_LRG_abacus_Y1_Ksys_DEC_xi4.csv'],
                         pkg_set3 = ['./inputs/quadfits_LRG_abacus_Y1_Ksys_MZLS_xi0.csv',
                                     './inputs/quadfits_LRG_abacus_Y1_Ksys_MZLS_xi2.csv',
                                     './inputs/quadfits_LRG_abacus_Y1_Ksys_MZLS_xi4.csv'])
mod.test_model_base_pars(min_type = 'data',
                         poi_hard_lims = ((-250,250),(0.5,4)), 
                         gauss_priors = ((1.94,0.04),(1.94,0.04),(1,0.1),(1,0.1)),
                         Psys1_gauss_prior = (0,10),
                         Psys2_gauss_prior = (0,10),
                         Psys3_gauss_prior = (0,10),
                         #khod_hard_lims = (-5,5),
                         z_eff = 0.780, # v1.5 0.780, nbody png 0.787
                         Om_m0_g = 0.315,
                         z_fid = 0.776,
                         zhalo = 0.787, # box 0.819, y3 0.787
                         Om_m0_h = 0.3089, #
                         nwalkers = 75, #50
                         nsteps = 20000, #10000
                         plt_color = 'green',
                         fname = None, 
                         data_obs = obs_unwrapper(
                             pkg_loc=f'./inputs/Y1_LRG_correlations_v1.5_.fits',mod=mod),
                         plt_out = True)
# mod.wrap_chain_base_pars(burn_in_steps = 500, thinner = 1,
#                          fname_chain = f'./analysis/chains/scut380_baomask/Y1_LRG_correlations_v1.5_w_hughs_model_quadxi0_linxi2_FastPMY3.txt')