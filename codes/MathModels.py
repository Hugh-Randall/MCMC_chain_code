import numpy as np
import pandas as pd
from codes.helper_functions import *

class MathModel:
    """
    Exemplary class showing the necessary elements of a MathModel class. These classes
    are used to define how a theoretical observation vector is defined and are passed
    during initialization of a PNGmodel object. There are three necessary elements.

    Elements
    ----------
    parameter_defaults : pandas dataframe 
        dataframe defining the default values of the POIs, such as their init value for the 
        MCMC, the string used to label them in plots, their priors, etc. A row MUST
        be defined for every parameter of interest.
    extra_parameters : set
        A set of strings defining values that are necessary to computing the theoretical observation
        vector but that change depending on the data used as an observation vector (e.g. effective z).
        These values must be passed as **kwargs to PNGmodel.run_sampling each time it is called. 
        They will be stored temporarily as PNGmodel instance attributes but will ultimately be deleted 
        from the instance. These are stored long-term in the metadata file for each chain.
    xi_modded_base_pars : staticmethod
        Function defining the theoretical observation vector. It is defined at the vector-level and its
        elements will depend on the particular terms in your vector (2pcf multipoles, moments, 3pcf, etc.).
        Examples of functions meant for the 2pcf multipoles are given below. 
    """
    
    parameter_defaults = pd.DataFrame(columns=['key', 'init', 'prior', 'plot_label', 'num_decimals', 'unit'])
    parameter_defaults = parameter_defaults.set_index('key')
    parameter_defaults.loc['fNL'] = [0, [-250, 250,'flat'], r'$f_{NL}$', 0, '']

    extra_parameters = {'z_eff'}
    @staticmethod
    def xi_modded_base_pars(mod, params):
        raise NotImplementedError

class Y1:
    parameter_defaults = pd.DataFrame(columns=['key', 'init', 'prior', 'plot_label', 'num_decimals', 'unit'])
    parameter_defaults = parameter_defaults.set_index('key')
    parameter_defaults.loc['fNL'] = [0, [-250, 250,'flat'], r'$f_{NL}$', 0, '']
    parameter_defaults.loc['b1g'] = [1, [0.5, 4,'flat'], r'$b_{1g}$', 2, '']
    parameter_defaults.loc['b1h'] = [1, [1.94,0.04,'gauss'], r'$b_{1h}$', 2, '']
    parameter_defaults.loc['b1gfid'] = [1, [1.94,0.04,'gauss'], r'$b_{1g}^{fid}$', 2, '']
    parameter_defaults.loc['ph'] = [1, [1,0.1,'gauss'], r'$p_h$', 1, '']
    parameter_defaults.loc['pg'] = [1, [1,0.1,'gauss'], r'$p_g$', 1,'']
    parameter_defaults.loc['KsysSGC'] = [1, [0,10,'gauss'], r'$K_{\mathrm{SGC}}$', 1, r'\%']
    parameter_defaults.loc['KsysDEC'] = [1, [0,10,'gauss'], r'$K_{\mathrm{DEC}}$', 1, r'\%']
    parameter_defaults.loc['KsysMZLS'] = [1, [0,10,'gauss'], r'$K_{\mathrm{MZLS}}$', 1, r'\%']

    extra_parameters = {'z_eff', 'z_fid', 'z_halo', 'Om_m0_g', 'Om_m0_h'}
    
    @staticmethod
    def xi_modded_base_pars(mod, params):
        fNL, b1g, b1h, b1g_fid, ph, pg, Psys1, Psys2, Psys3 = params
        f_g = Omega_m_z(mod.z_eff,mod.Om_m0_g)**0.55
        f_fid = Omega_m_z(mod.z_fid,mod.Om_m0_g)**0.55
        f_h = Omega_m_z(mod.z_halo,mod.Om_m0_h)**0.55
        Dz_g = Dz_norm(mod.z_eff,Om_m0=mod.Om_m0_g)
        Dz_h = Dz_norm(mod.z_halo,Om_m0=mod.Om_m0_h)
        
        ### Define rescale factors ######
        r_fac_fid = np.ones(mod.N_obs_vec_masked)
        r_fac_c1 = np.ones(mod.N_obs_vec_masked)
        r_fac_c2 = np.ones(mod.N_obs_vec_masked)
        
        r_fac_fid[mod.term_masks['xi0']] = (b1g**2 + (2/3)*b1g*f_g + (f_g**2)/5)/(b1g_fid**2 + (2/3)*b1g_fid*f_fid + (f_fid**2)/5)
        r_fac_fid[mod.term_masks['xi2']] = ( (4/3)*b1g*f_g + (4/7)*(f_g**2) )/( (4/3)*b1g_fid*f_fid + (4/7)*(f_fid**2) )
        r_fac_fid[mod.term_masks['xi4']] = (f_g/f_fid)**2
    
        r_fac_c1[mod.term_masks['xi0']] = ((b1g + f_g/3)*(b1g-pg)*(mod.Om_m0_g/Dz_g))/\
                                ((b1h + f_h/3)*(b1h-ph)*(mod.Om_m0_h/Dz_h))
        r_fac_c2[mod.term_masks['xi0']] = (((b1g-pg)*(mod.Om_m0_g/Dz_g))**2)/(((b1h-ph)*(mod.Om_m0_h/Dz_h))**2)
        r_fac_c1[mod.term_masks['xi2']] = (f_g*(b1g-pg)*(mod.Om_m0_g/Dz_g))/(f_h*(b1h-ph)*(mod.Om_m0_h/Dz_h))
        
        #################################    
        fid_term = r_fac_fid*(mod.masked['xi_fid'])
        PNG_term = r_fac_c1*mod.masked['c1']*fNL + r_fac_c2*mod.masked['c2']*(fNL**2)
        sys_term = r_fac_fid*((mod.masked['pvar_par_A1']*Psys1**2+mod.masked['pvar_par_B1']*Psys1) +\
                              (mod.masked['pvar_par_A2']*Psys2**2+mod.masked['pvar_par_B2']*Psys2) +\
                              (mod.masked['pvar_par_A3']*Psys3**2+mod.masked['pvar_par_B3']*Psys3))
        return fid_term + PNG_term + sys_term

class DR2_nosys:
    parameter_defaults = pd.DataFrame(columns=['key', 'init', 'prior', 'plot_label', 'num_decimals', 'unit'])
    parameter_defaults = parameter_defaults.set_index('key')
    parameter_defaults.loc['fNL'] = [0, [-250, 250,'flat'], r'$f_{NL}$', 0, '']
    parameter_defaults.loc['b1g'] = [1, [0.5, 4,'flat'], r'$b_{1g}$', 2, '']
    parameter_defaults.loc['b1gfid'] = [1, [1.94,0.04,'gauss'], r'$b_{1g}^{fid}$', 2, '']
    parameter_defaults.loc['pfid'] = [1, [1,0.1,'gauss'], r'$p_{fid}$', 1,'']
    parameter_defaults.loc['pg'] = [1, [1,0.1,'gauss'], r'$p_g$', 1,'']

    extra_parameters = {'z_eff', 'z_fid', 'Om_m0_g', 'Om_m0_fid'}

    @staticmethod
    def xi_modded_base_pars(mod, params):
        fNL, b1g, b1g_fid, pfid, pg = params
        f_g = Omega_m_z(mod.z_eff,mod.Om_m0_g)**0.55
        f_fid = Omega_m_z(mod.z_fid,mod.Om_m0_fid)**0.55
        Dz_g = Dz_norm(mod.z_eff,Om_m0=mod.Om_m0_g)
        Dz_fid = Dz_norm(mod.z_fid,Om_m0=mod.Om_m0_fid)
        
        ### Define rescale factors ######
        r_fac_fid = np.ones(len(mod.xi_fid))
        r_fac_c1 = np.ones(len(mod.xi_fid))
        r_fac_c2 = np.ones(len(mod.xi_fid))
        
        r_fac_fid[mod.term_masks['xi0']] = (b1g**2 + (2/3)*b1g*f_g + (f_g**2)/5)/(b1g_fid**2 + (2/3)*b1g_fid*f_fid + (f_fid**2)/5)
        r_fac_fid[mod.term_masks['xi2']] = ( (4/3)*b1g*f_g + (4/7)*(f_g**2) )/( (4/3)*b1g_fid*f_fid + (4/7)*(f_fid**2) )
        r_fac_fid[mod.term_masks['xi2']] = (f_g/f_fid)**2
    
        r_fac_c1[mod.term_masks['xi0']] = ((b1g + f_g/3)*(b1g-pg)*(mod.Om_m0_g/Dz_g))/\
                                ((b1g_fid + f_fid/3)*(b1g_fid-pfid)*(mod.Om_m0_fid/Dz_fid))
        r_fac_c2[mod.term_masks['xi0']] = (((b1g-pg)*(mod.Om_m0_g/Dz_g))**2)/(((b1g_fid-pfid)*(mod.Om_m0_fid/Dz_fid))**2)
        r_fac_c1[mod.term_masks['xi2']] = (f_g*(b1g-pg)*(mod.Om_m0_g/Dz_g))/(f_fid*(b1g_fid-pfid)*(mod.Om_m0_fid/Dz_fid))
        #################################    
        fid_term = r_fac_fid*(mod.xi_fid)
        PNG_term = r_fac_c1*mod.c1*fNL + r_fac_c2*mod.c2*(fNL**2)
        return fid_term + PNG_term 

class DR2:
    parameter_defaults = pd.DataFrame(columns=['key', 'init', 'prior', 'plot_label', 'num_decimals', 'unit'])
    parameter_defaults = parameter_defaults.set_index('key')
    parameter_defaults.loc['fNL'] = [0, [-250, 250,'flat'], r'$f_{NL}$', 0, '']
    parameter_defaults.loc['b1g'] = [1, [0.5, 4,'flat'], r'$b_{1g}$', 2, '']
    parameter_defaults.loc['b1gfid'] = [1, [1.94,0.04,'gauss'], r'$b_{1g}^{fid}$', 2, '']
    parameter_defaults.loc['pfid'] = [1, [1,0.1,'gauss'], r'$p_{fid}$', 1,'']
    parameter_defaults.loc['pg'] = [1, [1,0.1,'gauss'], r'$p_g$', 1,'']
    parameter_defaults.loc['KsysSGC'] = [1, [0,10,'gauss'], r'$K_{\mathrm{SGC}}$', 1, r'\%']
    parameter_defaults.loc['KsysDEC'] = [1, [0,10,'gauss'], r'$K_{\mathrm{DEC}}$', 1, r'\%']
    parameter_defaults.loc['KsysMZLS'] = [1, [0,10,'gauss'], r'$K_{\mathrm{MZLS}}$', 1, r'\%']

    extra_parameters = {'z_eff', 'z_fid', 'Om_m0_g', 'Om_m0_fid'}

    @staticmethod
    def xi_modded_base_pars(mod, params):
        fNL, b1g, b1g_fid, pfid, pg, Psys1, Psys2, Psys3 = params
        f_g = Omega_m_z(mod.z_eff,mod.Om_m0_g)**0.55
        f_fid = Omega_m_z(mod.z_fid,mod.Om_m0_fid)**0.55
        Dz_g = Dz_norm(mod.z_eff,Om_m0=mod.Om_m0_g)
        Dz_fid = Dz_norm(mod.z_fid,Om_m0=mod.Om_m0_fid)
        
        ### Define rescale factors ######
        r_fac_fid = np.ones(len(mod.xi_fid))
        r_fac_c1 = np.ones(len(mod.xi_fid))
        r_fac_c2 = np.ones(len(mod.xi_fid))
        
        r_fac_fid[mod.term_masks['xi0']] = (b1g**2 + (2/3)*b1g*f_g + (f_g**2)/5)/(b1g_fid**2 + (2/3)*b1g_fid*f_fid + (f_fid**2)/5)
        r_fac_fid[mod.term_masks['xi2']] = ( (4/3)*b1g*f_g + (4/7)*(f_g**2) )/( (4/3)*b1g_fid*f_fid + (4/7)*(f_fid**2) )
        r_fac_fid[mod.term_masks['xi4']] = (f_g/f_fid)**2
    
        r_fac_c1[mod.term_masks['xi0']] = ((b1g + f_g/3)*(b1g-pg)*(mod.Om_m0_g/Dz_g))/\
                                ((b1g_fid + f_fid/3)*(b1g_fid-pfid)*(mod.Om_m0_fid/Dz_fid))
        r_fac_c2[mod.term_masks['xi0']] = (((b1g-pg)*(mod.Om_m0_g/Dz_g))**2)/(((b1g_fid-pfid)*(mod.Om_m0_fid/Dz_fid))**2)
        r_fac_c1[mod.term_masks['xi2']] = (f_g*(b1g-pg)*(mod.Om_m0_g/Dz_g))/(f_fid*(b1g_fid-pfid)*(mod.Om_m0_fid/Dz_fid))
        #################################    
        fid_term = r_fac_fid*(mod.xi_fid)
        PNG_term = r_fac_c1*mod.c1*fNL + r_fac_c2*mod.c2*(fNL**2)
        sys_term = r_fac_fid*((mod.pvar_par_A1*Psys1**2+mod.pvar_par_B1*Psys1) +\
                              (mod.pvar_par_A2*Psys2**2+mod.pvar_par_B2*Psys2) +\
                              (mod.pvar_par_A3*Psys3**2+mod.pvar_par_B3*Psys3))
        return fid_term + PNG_term + sys_term