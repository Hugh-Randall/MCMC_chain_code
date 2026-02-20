# =========================================================================== #
# Imports
import numpy as np
from astropy.io import fits
import emcee
from multiprocessing import Pool
import yaml
from sys import platform

import os
os.environ["OMP_NUM_THREADS"] = "1"

from codes.helper_functions import *

import matplotlib.pyplot as plt
from pathlib import Path
module_path = Path(__file__).parent
plt.style.use( str(module_path) + '/config/mystyle.mplstyle')

class PNGmodel:
         
    def __init__(self, fid_corr, cov_pkg, math_model, exclude=None, s_min=None, s_max=None, s_cutwindow=None):
        # Initializes the model based on a desired s_min/s_max
        print('Initializing...')
        # Set initial params
        self.fid_corr_filename = fid_corr
        with fits.open(self.fid_corr_filename, memmap=False) as hdul:
            self.fid_corr = hdul[1].data.copy()
        self.cov_file = cov_pkg
        self.s_min = s_min
        self.s_max = s_max
        self.s_cutwindow = s_cutwindow
        self.math = math_model
        self.parameter_defaults = math_model.parameter_defaults
        self.num_params = len(self.parameter_defaults)
        self.parameters = list(self.parameter_defaults.index)
        
        # Print length of total observable depending on corr type
        self.s_slice = get_2pcf_idx_slice(self.fid_corr,self.s_min,self.s_max, self.s_cutwindow)
        xi0_fid = self.fid_corr['xi0']
        xi2_fid = self.fid_corr['xi2']
        xi4_fid = self.fid_corr['xi4']
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
        self.cov_inv = np.linalg.inv(self.cov_mat)
        return
    
    def load_photo_vary_fits(self, pkg_set1, pkg_set2, pkg_set3):
        print('Loading systematic weight variation...')     
        self.pvar_par_B1, self.pvar_par_A1  = [x[self.mask] for x in concatenate_quadfits(pkg_set1)]
        self.pvar_par_B2, self.pvar_par_A2  = [x[self.mask] for x in concatenate_quadfits(pkg_set2)]
        self.pvar_par_B3, self.pvar_par_A3  = [x[self.mask] for x in concatenate_quadfits(pkg_set3)]
        return
        
    def load_joint_fits(self, pkg_set):
        print('Loading systematic weight variation...')
        total_fits = concatenate_fits(pkg_set)[self.mask]
        self.c2, self.c1 = total_fits[:,0], total_fits[:,1]
        self.pvar_par_A1, self.pvar_par_B1 = total_fits[:,2], total_fits[:,3]
        self.pvar_par_A2, self.pvar_par_B2 = total_fits[:,4], total_fits[:,5]
        self.pvar_par_A3, self.pvar_par_B3 = total_fits[:,6], total_fits[:,7]
        return

    def xi_modded_base_pars(self, params):
        return self.math.xi_modded_base_pars(self, params)
    
    def util_chi2_base_pars(self, params):
        return self.math.util_chi2_base_pars(self, params)

    def log_prior_base_pars(self, params):
        return self.math.log_prior_base_pars(self, params)

    def log_probability_base_pars(self, params):
        return self.math.log_probability_base_pars(self, params)
    
    def test_model_base_pars(self, min_type, # min_type = 'data' or 'pseudo'
                             data_obs=None, nwalkers=75, nsteps=20000, # model attributes
                             plt_out=True, plt_color='green', savefig=False, fname_out=None, # optional plotting params 
                             multiprocessing=True,
                             **kwargs):
        print('Exploring parameter space...')

        # turn off mutliprocessing for windows
        if platform == 'win32':
            multiprocessing = False
        
        self.nwalkers = nwalkers
        self.nsteps = nsteps

        missing_attributes = self.get_missing_attributes()
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        if min_type == 'pseudo':
            if not hasattr(self, 'params_toy'):
                raise Exception(""" If min_type=='pseudo' you must pass toy parameters in the kwargs
                                    Check which parameters are needed with the show_parameters() method! """ )
            self.obs = self.xi_modded_base_pars(self.params_toy)
        elif min_type == 'data':
            self.obs = obs_unwrapper(data_obs)[self.mask]

        # Pull initial values from parameter defaults
        start_pos = np.asarray(self.parameter_defaults['init'])+1e-4*np.random.randn(
                               self.nwalkers, self.num_params)
        if multiprocessing:
            with Pool(8) as pool:
            # Define and run the sampler chain
                self.sampler = emcee.EnsembleSampler(self.nwalkers,
                                                     self.num_params,
                                                     self.log_probability_base_pars,
                                                     pool=pool)
                self.sampler.run_mcmc(start_pos, self.nsteps, progress=True)
        else:
            self.sampler = emcee.EnsembleSampler(self.nwalkers,
                                                     self.num_params,
                                                     self.log_probability_base_pars)
            self.sampler.run_mcmc(start_pos, self.nsteps, progress=True)

        if savefig:
            if not plt_out:
                print(f'savefig = {savefig} but plt_out = {plt_out}!')
            if fname_out is None:
                print(f'savefig = {savefig} but fname_out = {fname_out}!')
        
        if plt_out == True:
            # Plot walker output
            plt.rc('xtick', labelsize = 12)
            plt.rc('ytick', labelsize = 12)
            plt.rc('lines', lw = 1)
            fig, axes = plt.subplots(self.num_params, figsize=(12, 14), sharex=True)
            plt_samples = self.sampler.get_chain()
            for i in range(self.num_params):
                ax = axes[i]
                ax.plot(plt_samples[:, :, i], plt_color, alpha=0.3)
                ax.set_xlim(0, len(plt_samples))
                ax.set_ylabel(self.parameter_defaults['plot_label'].iloc[i])
                ax.yaxis.set_label_coords(-0.1, 0.5)
            axes[-1].set_xlabel("step number");
            if savefig:
                plt.savefig(fname_out)

        for attr in missing_attributes:
            delattr(self, attr)
        return
    
    def wrap_chain_base_pars(self,burn_in_steps,thinner,fname_chain):
        burn_samples = self.sampler.get_chain(
            discard=burn_in_steps,thin=thinner,flat=True)
        # (burn_samples == flat_samples).all() evaluates to true so I'm not sure what the point of flat_samples is.
        # flat_samples = np.stack((burn_samples.T[0],
        #                          burn_samples.T[1],
        #                          burn_samples.T[2],
        #                          burn_samples.T[3],
        #                          burn_samples.T[4],
        #                          burn_samples.T[5],
        #                          burn_samples.T[6],
        #                          burn_samples.T[7],
        #                          burn_samples.T[8])).T
        # np.savetxt(fname_chain, flat_samples)
        # ints_chain = get_ints(flat_samples)
        np.savetxt(fname_chain, burn_samples)
        qnts = get_ints(burn_samples)
        
        for i in range(self.num_params):
            param = self.parameters[i]
            print(param + ' = '+str(np.round(qnts[i][1],decimals=2))+' + '+
                  str(np.round(qnts[i][0],decimals=2))+' - '+
                  str(np.round(qnts[i][2],decimals=2)))
            
        # save Meta File:
        fname_meta = chain_meta_fname(fname_chain)
        meta = {}
        meta['parameter_defaults'] = self.parameter_defaults.to_dict(orient='index')
        meta['fid_corr_filename'] = self.fid_corr_filename
        meta['cov_filename'] = self.cov_file
        meta['scale'] = {'s_min': self.s_min, 's_max': self.s_max, 's_cutwindow': self.s_cutwindow}
        meta['math_model'] = self.math.__class__.__name__
        meta['qnts'] = [[float(q) for q in tup] for tup in qnts]
        print(meta['qnts'])
        with open(fname_meta, 'w') as f:
            yaml.dump(meta, f, sort_keys=False)
            
        return

    def get_missing_attributes(self):
        return sorted(self.math.extra_parameters - set(vars(self)))
    
    def show_missing_attributes(self):
        missing_attributes = self.get_missing_attributes()
        print("Must pass the following as a dictionary to 'test_model_base_pars':")
        print( missing_attributes )

    def show_parameters(self):
        print('This MathModel needs the following parameters:')
        print(self.parameters)