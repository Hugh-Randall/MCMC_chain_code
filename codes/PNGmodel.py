# =========================================================================== #
# Imports
import numpy as np
import pandas as pd
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
         
    def __init__(self, fid_corr, math_model):
        # Initializes the model based on a desired s_min/s_max
        print('Initializing...')
        # Set initial params
        self.fid_corr_filename = fid_corr
        with fits.open(self.fid_corr_filename, memmap=False) as hdul:
            self.fid_corr = hdul[1].data.copy()
        self.math = math_model
        self.parameter_defaults = math_model.parameter_defaults
        self.num_params = len(self.parameter_defaults)
        self.parameters = list(self.parameter_defaults.index)
        
        self.xi_fid, self.terms = obs_unwrapper(self.fid_corr_filename)
        self.N_obs_vec = len(self.xi_fid)
        return

    def load_PNG_model(self, files):
        # Loads c1_n and c2_n coefficients
        print('Loading PNG model...')
        # c1, c2 = concatenate_quadfits(files)
        # self.c1 = c1[self.mask]
        # self.c2 = c2[self.mask]
        df = reorder_fits(pd.read_csv(files), self.terms)
        self.c1, self.c2 = np.asarray(df['c1']), np.asarray(df['c2'])
        return
    
    def load_covariance(self, cov_pkg, cov_rescale_factor=1.):
        # NOTICE - This is the only part of the model that is assumed to be pre-concatenated.
        # You MUST be sure that the order of the covariance matrix indices matches the order of self.terms!
        ...
        # possibly change this so that it could be computed on the fly like Zack had it originally
        ...
        # A function to load the model covariance matrix
        # Takes from the fiducial ensemble
        print('Loading covariance matrix...')
        self.cov_file = cov_pkg
        # self.cov_mat = cov_rescale_factor*np.load(self.cov_file)[self.mask][:, self.mask]
        self.cov_mat = cov_rescale_factor*np.load(self.cov_file)
        return
    
    def load_photo_vary_fits(self, pkg_set1, pkg_set2, pkg_set3):
        print('Loading systematic weight variation...')
        self.sys_pkg_sets = [pkg_set1, pkg_set2, pkg_set3]
        # self.pvar_par_B1, self.pvar_par_A1  = [x[self.mask] for x in concatenate_quadfits(pkg_set1)]
        # self.pvar_par_B2, self.pvar_par_A2  = [x[self.mask] for x in concatenate_quadfits(pkg_set2)]
        # self.pvar_par_B3, self.pvar_par_A3  = [x[self.mask] for x in concatenate_quadfits(pkg_set3)]
        
        df1 = reorder_fits(pd.read_csv(pkg_set1), self.terms)
        self.pvar_par_B1, self.pvar_par_A1 = np.asarray(df1['c1']), np.asarray(df1['c2'])

        df2 = reorder_fits(pd.read_csv(pkg_set2), self.terms)
        self.pvar_par_B2, self.pvar_par_A2 = np.asarray(df2['c1']), np.asarray(df2['c2'])

        df3 = reorder_fits(pd.read_csv(pkg_set3), self.terms)
        self.pvar_par_B3, self.pvar_par_A3 = np.asarray(df3['c1']), np.asarray(df3['c2'])
        return
        
    def load_joint_fits(self, pkg_set):
        raise Exception( 'This functionality is currently under construction!' )
        # print('Loading systematic weight variation...')
        # # total_fits = concatenate_fits(pkg_set)[self.mask]
        # total_fits = concatenate_fits(pkg_set)[self.mask]
        
        # self.c2, self.c1 = total_fits[:,0], total_fits[:,1]
        # self.pvar_par_A1, self.pvar_par_B1 = total_fits[:,2], total_fits[:,3]
        # self.pvar_par_A2, self.pvar_par_B2 = total_fits[:,4], total_fits[:,5]
        # self.pvar_par_A3, self.pvar_par_B3 = total_fits[:,6], total_fits[:,7]
        return

    def xi_modded_base_pars(self, params):
        return self.math.xi_modded_base_pars(self, params)
    
    def util_chi2_base_pars(self, params):
        return self.math.util_chi2_base_pars(self, params)

    def log_prior_base_pars(self, params):
        return self.math.log_prior_base_pars(self, params)

    def log_probability_base_pars(self, params):
        return self.math.log_probability_base_pars(self, params)
    
    def run_sampling(self, 
                     min_type, # min_type = 'data' or 'pseudo'
                     fname_chain, # filepath for output chain data
                     data_obs=None, # Filepath for input observation, necessary if min_type=='data'
                     s_min=None, s_max=None, s_cutwindow=None, exclude=[], # scale cuts used to decide what to mask in the model
                     nwalkers=75, nsteps=20000, # model attributes
                     plt_out=True, plt_color='green', savefig=False, fname_out=None, # optional plotting params 
                     multiprocessing=False,
                     burn_in_steps=500, thinner=1,
                     **kwargs):
        
        # defining mask
        self.exclude = exclude
        self.s_min = s_min
        self.s_max = s_max
        self.s_cutwindow = s_cutwindow
        self.s_slice = get_2pcf_idx_slice(self.fid_corr,self.s_min,self.s_max, self.s_cutwindow)
        self.s_mask = np.concatenate(len(self.terms)*[self.s_slice])
        
        len_per_xi = len(self.s_slice)
        total_len = len(self.xi_fid)

        self.term_masks = {term: np.zeros(self.N_obs_vec, dtype=bool) for term in self.terms}
        for i,term in enumerate(self.terms):
            self.term_masks[term][i*len_per_xi:(i+1)*len_per_xi] = True
            
        self.mask = self.s_mask.copy()
        for term in exclude:
            self.mask = np.logical_and(self.mask, ~self.term_masks[term])
            
        for term in self.terms:
            self.term_masks[term] = self.term_masks[term][self.mask]

        arrays_to_mask = ['xi_fid', 'c1', 'c2', 'pvar_par_B1', 'pvar_par_A1', 'pvar_par_B2', 'pvar_par_A2', 'pvar_par_B3', 'pvar_par_A3']
        # make it so that arrays_to_mask is created as each part is initialized within the other methods. 
        self.masked = {'cov_inv': np.linalg.inv(self.cov_mat[self.mask][:,self.mask])}
        for tm in arrays_to_mask:
            self.masked[tm] = getattr(self, tm)[self.mask]
            
        self.N_obs_vec_masked = len(self.masked['xi_fid'])
        
        attrs_to_delete = ['mask', 'term_masks', 'masked', 'N_obs_vec_masked', 'exclude']
        print('Observable will have {} pts'.format(len(self.masked['xi_fid'])))
        ######################################################
        
        print()
        
        ######################################################
        print('Exploring parameter space...')
        # turn off mutliprocessing for windows
        if platform == 'win32':
            multiprocessing = False
        
        self.nwalkers = nwalkers
        self.nsteps = nsteps

        missing_attributes = self.get_missing_attributes()
        attrs_to_delete += missing_attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        if min_type == 'pseudo':
            if not hasattr(self, 'params_toy'):
                raise Exception(""" If min_type=='pseudo' you must pass toy parameters in the kwargs
                                    Check which parameters are needed with the show_parameters() method! """ )
            self.obs = self.xi_modded_base_pars(self.params_toy)
        elif min_type == 'data':
            self.obs, _ = obs_unwrapper(data_obs)
            self.obs = self.obs[self.mask]
            
        attrs_to_delete.append(self.obs)

        # Pull initial values from parameter defaults
        start_pos = np.asarray(self.parameter_defaults['init'])+1e-4*np.random.randn(
                               self.nwalkers, self.num_params)

        # The following line is just for fixing the bias 
        # WILL DELETE AT SOME POINT
        # start_pos[:,1] = (self.poi_hard_lims[1][1]+self.poi_hard_lims[1][0])/2.+1e-5*np.random.randn(self.nwalkers)

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

        burn_samples = self.sampler.get_chain(
            discard=burn_in_steps,thin=thinner,flat=True)
        np.savetxt(fname_chain, burn_samples)
        self.qnts = get_ints(burn_samples)
        
        for i in range(self.num_params):
            param = self.parameters[i]
            print(param + ' = '+str(np.round(self.qnts[i][1],decimals=2))+' + '+
                  str(np.round(self.qnts[i][0],decimals=2))+' - '+
                  str(np.round(self.qnts[i][2],decimals=2)))
            
        # save Meta File:
        self.save_meta(fname_chain)

        missing_attributes.append('qnts')
        for attr in missing_attributes:
            delattr(self, attr)
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

    def save_meta(self, fname_chain):
        fname_meta = chain_meta_fname(fname_chain)
        meta = {}
        meta['parameter_defaults'] = self.parameter_defaults.to_dict(orient='index')
        meta['fid_corr_filename'] = self.fid_corr_filename
        meta['cov_filename'] = self.cov_file
        meta['scale'] = {'s_min': self.s_min, 's_max': self.s_max, 's_cutwindow': self.s_cutwindow}
        meta['math_model'] = self.math.__class__.__name__
        meta['qnts'] = [[float(q) for q in tup] for tup in self.qnts]
        meta['exclude'] = self.exclude
        # meta['sys_pkg_sets'] = self.sys_pkg_sets
        with open(fname_meta, 'w') as f:
            yaml.dump(meta, f, sort_keys=False)