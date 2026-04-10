# =========================================================================== #
# Imports
import numpy as np
import pandas as pd
from astropy.io import fits
import emcee
from multiprocessing import Pool
import yaml
from sys import platform
from tqdm import tqdm

import os

from codes.helper_functions import *

import matplotlib.pyplot as plt
from pathlib import Path
module_path = Path(__file__).parent
plt.style.use( str(module_path) + '/config/mystyle.mplstyle')

class PNGmodel:
         
    def __init__(self, fid_corr, math_model, terms=None):
        """
        Initializes a PNGmodel object with which we can run MCMC parameter 
        estimation on various sets of data. 
    
        Parameters
        ----------
        fid_corr : str
            Full path/filename to file that stores the fiducial observation vector 
            describing our model. It is assumed to be a 2d fits table with one column
            being the scale, s, and other columns being the terms in the observation vector.
            This format will be changed soon.
        math_model : MathModel-like object
            An instance of any class from the MathModel.py file. This MathModel is how 
            the PNGmodel's xi_modded_base_pars will be defined to build the likelihood function.
        terms : None
            Under construction. DO NOT USE.
        """
        
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
        
        self.xi_fid, fid_terms = obs_unwrapper(self.fid_corr_filename)
        if terms == None:
            self.terms = list(dict.fromkeys(fid_terms))

        self.parameterization_files = []
        self.arrays_to_mask = ['xi_fid']
        self.N_obs_vec = len(self.xi_fid)
        return

    def load_fits(self, file, mapper=None):
        """
        Loads coefficients that define our parameterized model. This can be used to load 
        png coefficients as well as systematics coefficients. It defined the columns of the
        file to be PNGmodel attributes that will be called within the likelihood. 
        The names of the columns and/or the mapper dictionary solely determines how the attributes
        are named. The only thing that matters is that the attributes are ultimately named 
        in a way that is consistent with how they are called within your chosen MathModel.
    
        Parameters
        ----------
        file : str
            Full path/filename to csv file that stores the coefficients that will be called in the 
            likelihood function as defined by the MathModel. There are two necessary columns that the 
            file MUST have: ['s', 'term']. 's' corresponds to the scale (e.g. 50) and 'term' corresponds
            to the term of your observation vector (e.g. 'xi0', 'xi2', etc.). All other columns are 
            coefficients for the given (s,term) combination. If 'mapper' is not specified, then the columns
            will be stored as model attributes with names given by the column names. For example, column
            'c1' will be stored as PNGmodel.c1, so be careful that your naming conventions are self-consistent.
        mapper : dict, optional 
            Dictionary used to change what names the columns of 'file' are saved under when they are made
            into PNGmodel attributes. This is useful when the columns of your table do not match what they are 
            called in MathModel.xi_modded_base_pars. For example, if you are loading systematic fits files whose 
            linear and quadratic fits columns are labeled c1 and c2 respectively, 
            use mapper={'c1': 'pvar_par_B1', 'c2': 'pvar_par_A1'}, to make PNGmodel.pvar_par_B2 = file['c1'].
        """
        print('Loading model coefficients...')
        df = reorder_fits(pd.read_csv(file), self.terms)
        cols = list(df.columns)
        to_remove = ['term', 's']
        cols = [x for x in cols if x not in to_remove]
        
        if not mapper:
            mapper = {col:col for col in cols}

        
        for col in cols:
            setattr(self, mapper[col], np.asarray(df[col]))
            print(f'\tadded attribute: {mapper[col]}')
            self.arrays_to_mask.append(mapper[col])
        self.parameterization_files.append(file)
        return
    
    def load_covariance(self, cov_pkg, cov_rescale_factor=1.):
        """
        Loads the covariance matrix as a PNGmodel object. Notice! It does not compute the 
        covariance matrix so you must be sure that the order of the indices of the covariance 
        matrix matches the order of the indices in all other parts of the model (like in the 
        fiducial case, coefficients, etc.).
    
        Parameters
        ----------
        cov_pkg : str
            Full path/filename to the npy file that stores the covariance matrix.
        cov_rescale_factor : float, optional
            Float used to rescale the covariance matrix for quick comparisons between different
            survey volumes.
        """
        print('Loading covariance matrix...')
        self.cov_file = cov_pkg
        self.cov_mat = cov_rescale_factor*np.load(self.cov_file)
        return

    def xi_modded_base_pars(self, params):
        return self.math.xi_modded_base_pars(self, params)

    def util_chi2_base_pars(self, params):
        # Defines chi2 given data and params
        exp = self.xi_modded_base_pars(params)
        return -np.matmul(np.matmul(self.masked['cov_inv'],(self.obs-exp)),(self.obs-exp))

    def log_probability_base_pars(self, params):
        # Defines the log probability combining the likelihood and priors
        lp = self.log_prior_base_pars(params)
        if not np.isfinite(lp):
            return -np.inf
        return 0.5*(lp + self.util_chi2_base_pars(params))
    
    def run_sampling(self, 
                     min_type, 
                     fname_chain, 
                     data_obs=None, params_toy=None,
                     s_min=None, s_max=None, s_cutwindow=None, exclude=[], # scale/term cuts used to decide what to mask in the model
                     nwalkers=75, nsteps=20000, burn_in_steps=500, thinner=1, multiprocessing=False, # model attributes
                     plt_out=True, plt_color='green', savefig=False, fname_out=None, # optional plotting params 
                     update_priors=None,
                     **kwargs):
        """
        Runs the MCMC parameter estimation and saves the resulting chains along with their metadata.
    
        Parameters
        ----------
        min_type : str
            A string defining where your observation vector comes from. Either 'data' or 'pseudo'. 
            If 'data' is chosen then the 'data_obs' argument must also be provided. In this case the
            observation vector is taken to be the data stored in the 'data_obs' file. If 'pseudo' is 
            chosen, then the 'params_toy' argument must also be provided. In this case the observation
            vector is constructed by passing 'params_toy' to self.xi_modded_base_pars.
        fname_chain : str
            Full filepath/filename where the output chain will be saved. 
        data_obs : str, optional
            Full filepath to data (in fits format) that will be used as the observation vector. 
        params_toy : array, optional
            Array holding the parameters of interest to be used to construct the observation vector. 
        s_min : int or float, optional
            Minimum scale to be used in the model.
        s_max : int or float, optional
            Maximum scale to be used in the model.
        s_cutwindow : array of integers or floats, optional
            Length-two array representing a range of scales to be masked, e.g. [90,130] for masking BAO.
        exclude : array of strings, optional
            Array of strings corresponding to terms in your data vector to be masked. For example, if
            your data vector has the terms xi0, xi2, xi4, setting exclude=['xi4'] masks the 
            hexadecapole everywhere within the model.
        nwalkers : int, optional
            Number of walkers used in the MCMC. See: https://emcee.readthedocs.io/en/stable/user/sampler/
        nsteps : int, optional
            Number of steps used in the MCMC. See: https://emcee.readthedocs.io/en/stable/user/sampler/
        burn_in_steps : int, optional
            Number of steps to discard when saving getting chain.
            See: https://emcee.readthedocs.io/en/stable/user/sampler/
        thinner : int, optional
            Number used to thin the chain. See: https://emcee.readthedocs.io/en/stable/user/sampler/
        multiprocessing : bool, optional
            Boolean determining whether the emcee.EnsembleSampler is parallelized using Pool(). Under
            construction and defaults to False. This emcee feature is incompatible with windows 
            and so if the system used to run this code is using windows, it defaults to False.
        plt_out : bool, optional
            Boolean to decide if a plot will be made showing all the steps taken by the walkers. If 
            'savefig' is set to True but 'plt_out' is set to False, the figure will be plotted anyway. 
        plt_color : str, optional
            String indicating the color of the paths taken by the walkers in the plot.
        savefig : bool, optional
            Boolean to decide whether the walker plot figure is saved. 
        fname_out : str, optional
            Full directory/filename where the walker plot figure will be saved. 
        update_priors : None or dict, optional
            Used to update the priors for this MCMC run from their default values as they are defined in
            the chosen PNGmodel's parameter_defaults. Default is None. If a dictionary is provided, the 
            keys must be strings corresponding to the keys of the parameter_defaults dataframe. The values 
            must be arrays corresponding to the values with which the chosen parameter's prior will be replaced.
            For example, update_priors={'b1g':[1.90, 0.06, 'gauss']}
        kwargs : dict, optional
            Used to add data that is necessary to run the model but not previously loaded. The values 
            passed should correspond to those defined in the given MathModel's 'extra_parameters'.
            Anything passed to kwargs will temporarily be stored as an instance attribute to the PNGmodel,
            and then subsequently removed from the instance's attributes at the end of this function. 
        """

        self.prep_run_dependent_parts(min_type=min_type, 
                                      data_obs=data_obs, params_toy=params_toy,
                                      s_min=s_min, s_max=s_max, s_cutwindow=s_cutwindow, exclude=exclude, 
                                      update_priors=update_priors,
                                      # Now the kwargs 
                                      **kwargs)
        
        #####################################################
        ### MCMC dependent attributes
        #####################################################
        self.nwalkers = nwalkers
        self.nsteps = nsteps
        self.attrs_to_delete.extend(['nwalkers', 'nsteps'])      

        #####################################################
        ### Run the MCMC 
        #####################################################
        if platform == 'win32':# turn off mutliprocessing for windows
                multiprocessing = False
            
        start_pos = np.asarray(self.parameter_info['init'])+1e-4*np.random.randn(
                                   self.nwalkers, self.num_params)
        print('Exploring parameter space...')
        if multiprocessing:
            os.environ["OMP_NUM_THREADS"] = "1"
            with Pool(8) as pool:
            # Define and run the sampler chain
                self.sampler = emcee.EnsembleSampler(self.nwalkers,
                                                     self.num_params,
                                                     self.log_probability_base_pars,
                                                     pool=pool)
                self.sampler.run_mcmc(start_pos, self.nsteps, progress=True)
            _ = os.environ.pop("OMP_NUM_THREADS")
        else:
            self.sampler = emcee.EnsembleSampler(self.nwalkers,
                                                     self.num_params,
                                                     self.log_probability_base_pars)
            self.sampler.run_mcmc(start_pos, self.nsteps, progress=True)
        self.attrs_to_delete.append('sampler')

        #####################################################
        ### Plotting/saving walker plots
        #####################################################
        if savefig:
            plt_out = True # override plt_out=False if savefig is set.
            if fname_out is None:
                print(f'savefig = {savefig} but fname_out = None! Figure will not be saved.')
        if plt_out == True:
            self.plot_walkers(plt_color=plt_color, savefig=savefig, fname_out=fname_out)

        #####################################################
        ### Get and display parameter estimates
        #####################################################
        burn_samples = self.sampler.get_chain(
            discard=burn_in_steps,thin=thinner,flat=True)
        np.savetxt(fname_chain, burn_samples)
        self.qnts = get_ints(burn_samples)
        for i in range(self.num_params):
            param = self.parameters[i]
            print(param + ' = '+str(np.round(self.qnts[i][1],decimals=2))+' + '+
                  str(np.round(self.qnts[i][0],decimals=2))+' - '+
                  str(np.round(self.qnts[i][2],decimals=2)))
            
        #####################################################
        ### Save Meta file and delete temporary attributes
        #####################################################
        self.save_meta(fname_chain)
        self.attrs_to_delete.append('qnts')
        
        for attr in self.attrs_to_delete.copy():
            delattr(self, attr)
        return

    def compute_likelihood(self, 
                           min_type, 
                           param,
                           param_values,
                           fixed_params,
                           data_obs=None, params_toy=None,
                           s_min=None, s_max=None, s_cutwindow=None, exclude=[], # scale/term cuts used to decide what to mask in the model
                           update_priors=None,
                           **kwargs):
        """
        Complementary to run_sampling. It has very similar inputs but rather than running the MCMC
        sampling, it caclulated the chi squared as a function of one of the parameters of interest. 
        This can be helpful when debugging. Currently, only one parameter may be varied at a time,
        while all others must be fixed.
    
        Parameters
        ----------
        min_type : str
            A string defining where your observation vector comes from. Either 'data' or 'pseudo'. 
            If 'data' is chosen then the 'data_obs' argument must also be provided. In this case the
            observation vector is taken to be the data stored in the 'data_obs' file. If 'pseudo' is 
            chosen, then the 'params_toy' argument must also be provided. In this case the observation
            vector is constructed by passing 'params_toy' to self.xi_modded_base_pars.
        param : str
            Parameter that will be varied when calculating and plotting the likelihood.
        param_values: array
            Array of values that 'param' will take.
        fixed_params: dict
            A dictionary holding the values that other POIs will be fixed too. The keys are the parameter
            labels (matching the keys of MathModel.parameter_defaults), and the values are the corresponding 
            values the keys will be fixed to.
        data_obs : str, optional
            Full filepath to data (in fits format) that will be used as the observation vector. 
        params_toy : array, optional
            Array holding the parameters of interest to be used to construct the observation vector. 
        s_min : int or float, optional
            Minimum scale to be used in the model.
        s_max : int or float, optional
            Maximum scale to be used in the model.
        s_cutwindow : array of integers or floats, optional
            Length-two array representing a range of scales to be masked, e.g. [90,130] for masking BAO.
        exclude : array of strings, optional
            Array of strings corresponding to terms in your data vector to be masked. For example, if
            your data vector has the terms xi0, xi2, xi4, setting exclude=['xi4'] masks the 
            hexadecapole everywhere within the model.
        update_priors : None or dict, optional
            Used to update the priors for this MCMC run from their default values as they are defined in
            the chosen PNGmodel's parameter_defaults. Default is None. If a dictionary is provided, the 
            keys must be strings corresponding to the keys of the parameter_defaults dataframe. The values 
            must be arrays corresponding to the values with which the chosen parameter's prior will be replaced.
            For example, update_priors={'b1g':[1.90, 0.06, 'gauss']}
        kwargs : dict, optional
            Used to add data that is necessary to run the model but not previously loaded. The values 
            passed should correspond to those defined in the given MathModel's 'extra_parameters'.
            Anything passed to kwargs will temporarily be stored as an instance attribute to the PNGmodel,
            and then subsequently removed from the instance's attributes at the end of this function. 
        """

        self.prep_run_dependent_parts(min_type=min_type, 
                                      data_obs=data_obs, params_toy=params_toy,
                                      s_min=s_min, s_max=s_max, s_cutwindow=s_cutwindow, exclude=exclude,
                                      update_priors=update_priors,
                                      # Now the kwargs 
                                      **kwargs)
        
        #####################################################
        ### likelihood dependent attributes
        #####################################################
        idx = self.parameters.index(param)
        idxs_other = [self.parameters.index(key) for key in fixed_params.keys()]

        print('Computing chi squared...')
        likelihood = []
        for val in tqdm(param_values):
            params = np.zeros(len(self.parameters))
            params[idx] = val
            params[idxs_other] = list(fixed_params.values())
            likelihood.append(self.log_probability_base_pars(params))
        likelihood = np.asarray(likelihood)
       
        for attr in self.attrs_to_delete.copy():
            delattr(self, attr)
            
        return likelihood       

    @staticmethod
    def compile_log_prior(priors):
        # Separate the two prior types at compile time
        gaussian_terms = [(i, p[0], p[1]) for i, p in enumerate(priors) if p[2] == "gauss"]
        uniform_terms  = [(i, p[0], p[1]) for i, p in enumerate(priors) if p[2] == "flat"]
    
        def log_prior(params):
            # Uniform bounds check first — cheapest early exit
            for i, low, high in uniform_terms:
                if not (low <= params[i] <= high):
                    return -np.inf
    
            # Gaussian terms inlined — no function call overhead
            total = 0.0
            for i, mean, sigma in gaussian_terms:
                d = ((params[i] - mean) / sigma)**2
                total -= d
    
            return total
        return log_prior

    def prep_run_dependent_parts(self, 
                                 min_type, 
                                 data_obs=None, params_toy=None,
                                 s_min=None, s_max=None, s_cutwindow=None, exclude=[], # scale/term cuts used to decide what to mask in the model
                                 update_priors=None,
                                 **kwargs):
        attributes_initial = list(self.__dict__.keys())
        self.exclude = exclude
        self.s_min = s_min
        self.s_max = s_max
        self.s_cutwindow = s_cutwindow
        # self.attrs_to_delete = ['attrs_to_delete', 's_min', 's_max', 's_cutwindow', 'exclude']
        
        # self.prep_run_dependent_parts()
        unexpected_kwargs = ['smin', 'smax', 'scutwindow']
        for uk in unexpected_kwargs:
            if (uk in kwargs): 
                idx = uk.index('s')
                ek = uk[:idx+1] + '_' + uk[idx+1:]
                raise Exception(f'Unexpected key word argument "{uk}", did you mean "{ek}"?')
        
        self.parameter_info = self.parameter_defaults.copy()
        self.params_toy = params_toy
        self.data_obs = data_obs
        # self.attrs_to_delete.extend(['params_toy', 'parameter_info', 'data_obs'])

        #####################################################
        ### Update the priors if applicable
        #####################################################
        if update_priors is not None: 
            # Must be a dict with keys given by the parameter labels of parameter_defaults
            # and values with what the defaults will be replaced by
            # For example update_priors = {'b1g':(2, 0.03, 'gauss')}
            for key,val in update_priors.items(): 
                self.parameter_info.at[key, 'prior'] = val 

        #####################################################
        ### Define Masks for this run of the model
        #####################################################
        self.make_masked(self.s_min, self.s_max, self.s_cutwindow, self.exclude)
        print('Observable will have {} pts'.format(self.N_obs_vec_masked))
        # self.attrs_to_delete.extend(['mask', 'term_masks', 'masked', 'N_obs_vec_masked'])

        #####################################################
        ### Define the observation vector
        #####################################################
        if min_type == 'pseudo':
            self.obs = self.xi_modded_base_pars(self.params_toy)
        elif min_type == 'data':
            self.obs, _ = obs_unwrapper(self.data_obs)
            self.obs = self.obs[self.mask]
        # self.attrs_to_delete.append('obs')
        
        #####################################################
        ### Load any missing attributes
        #####################################################
        for key, value in kwargs.items():
            setattr(self, key, value)
                    
        #####################################################
        ### Define the log prior function:
        #####################################################
        priors = list(self.parameter_info['prior'])
        self.log_prior_base_pars = self.compile_log_prior(priors)  

        attributes_final = list(self.__dict__.keys())
        self.attrs_to_delete = list(set(attributes_final) - set(attributes_initial))
        return

    def plot_walkers(self, plt_color='green', savefig=False, fname_out=None):
        """
        Plots the steps taken by the walkers in the MCMC.
    
        Parameters
        ----------
        plt_color : str
            Color of the lines to be plotted.
        savefig : bool
            Boolean defining whether the figure will be saved. If True fname_out MUST
            be specified.
        fname_out : str
            Full output file path/name of figure if it is saved. 
        """
        # Plot walker output
        plt.rc('xtick', labelsize = 12)
        plt.rc('ytick', labelsize = 12)
        plt.rc('lines', lw = 1)
        plt_samples = self.sampler.get_chain()
        if self.num_params>1:
            fig, axes = plt.subplots(self.num_params, figsize=(12, 14), sharex=True)
            for i in range(self.num_params):
                ax = axes[i]
                ax.plot(plt_samples[:, :, i], plt_color, alpha=0.3)
                ax.set_xlim(0, len(plt_samples))
                ax.set_ylabel(self.parameter_info['plot_label'].iloc[i])
                ax.yaxis.set_label_coords(-0.1, 0.5)
            axes[-1].set_xlabel("step number");
        else:
            fig, ax = plt.subplots(figsize=(12, 14*(self.num_params/9.)))
            ax.plot(plt_samples[:, :, 0], plt_color, alpha=0.3)
            ax.set_xlim(0, len(plt_samples))
            ax.set_ylabel(self.parameter_info['plot_label'].iloc[0])
            ax.yaxis.set_label_coords(-0.1, 0.5)
            ax.set_xlabel("step number");
        if savefig:
            plt.savefig(fname_out)

    def save_meta(self, fname_chain):
        """
        Saves the metadata corresponding to a given run of PNGmodel.run_sampling. It 
        creates a yaml file complementary to the chain array.
    
        Parameters
        ----------
        fname_chain : str
            The filename that was passed to PNGmodel.run_sampling specifying the output file
            for the chain array. It is used to define the name for the corresponding yaml
            file that is saved when this function is called. 
        """
        fname_meta = chain_meta_fname(fname_chain)
        meta = {}
        meta['parameter_info'] = self.parameter_info.to_dict(orient='index')
        meta['fid_corr_filename'] = self.fid_corr_filename
        meta['cov_filename'] = self.cov_file
        meta['parameterization_files'] = self.parameterization_files
        meta['scale'] = {'s_min': self.s_min, 's_max': self.s_max, 's_cutwindow': self.s_cutwindow}
        meta['math_model'] = self.math.__class__.__name__
        meta['qnts'] = [[float(q) for q in tup] for tup in self.qnts]
        meta['exclude'] = self.exclude
        # meta['sys_pkg_sets'] = self.sys_pkg_sets
        with open(fname_meta, 'w') as f:
            yaml.dump(meta, f, sort_keys=False)

    def get_missing_attributes(self):
        return sorted(self.math.extra_parameters - set(vars(self)))
    
    def show_missing_attributes(self):
        missing_attributes = self.get_missing_attributes()
        print("Must pass the following as a dictionary to 'run_sampling':")
        print( missing_attributes )

    def show_parameters(self):
        print('This MathModel needs the following parameters:')
        print(self.parameters)

    def make_masked(self, s_min=None, s_max=None, s_cutwindow=None, exclude=[]):
        self.s_mask = get_2pcf_idx_slice(self.fid_corr, s_min, s_max, s_cutwindow)
        term_lengths = [len(self.fid_corr[self.fid_corr['term']==term]) for term in self.terms]
        
        self.term_masks = {term: self.fid_corr['term']==term for term in self.terms}
        
        # for i,term in enumerate(self.terms):
        #     first_idx = 0 if i==0 else term_lengths[i-1]
        #     last_idx = term_lengths[i] if i==0 else term_lengths[i]+term_lengths[i-1]
        #     print(first_idx, last_idx)
        #     self.term_masks[term][first_idx:last_idx] = True

        self.mask = self.s_mask.copy()
        for term in exclude:
            self.mask = np.logical_and(self.mask, ~self.term_masks[term])
            
        for term in self.terms:
            self.term_masks[term] = self.term_masks[term][self.mask]
        self.masked = {'cov_inv': np.linalg.inv(self.cov_mat[self.mask][:,self.mask])}
        for tm in self.arrays_to_mask:
            self.masked[tm] = getattr(self, tm)[self.mask]
        self.N_obs_vec_masked = len(self.masked['xi_fid'])