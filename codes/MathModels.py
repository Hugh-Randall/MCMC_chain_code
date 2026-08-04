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
        r_fac_fid = np.ones(mod.N_obs_vec_masked)
        r_fac_c1 = np.ones(mod.N_obs_vec_masked)
        r_fac_c2 = np.ones(mod.N_obs_vec_masked)
        
        r_fac_fid[mod.term_masks['xi0']] = (b1g**2 + (2/3)*b1g*f_g + (f_g**2)/5)/(b1g_fid**2 + (2/3)*b1g_fid*f_fid + (f_fid**2)/5)
        r_fac_fid[mod.term_masks['xi2']] = ( (4/3)*b1g*f_g + (4/7)*(f_g**2) )/( (4/3)*b1g_fid*f_fid + (4/7)*(f_fid**2) )
        r_fac_fid[mod.term_masks['xi2']] = (f_g/f_fid)**2
    
        r_fac_c1[mod.term_masks['xi0']] = ((b1g + f_g/3)*(b1g-pg)*(mod.Om_m0_g/Dz_g))/\
                                ((b1g_fid + f_fid/3)*(b1g_fid-pfid)*(mod.Om_m0_fid/Dz_fid))
        r_fac_c2[mod.term_masks['xi0']] = (((b1g-pg)*(mod.Om_m0_g/Dz_g))**2)/(((b1g_fid-pfid)*(mod.Om_m0_fid/Dz_fid))**2)
        r_fac_c1[mod.term_masks['xi2']] = (f_g*(b1g-pg)*(mod.Om_m0_g/Dz_g))/(f_fid*(b1g_fid-pfid)*(mod.Om_m0_fid/Dz_fid))
        #################################    
        fid_term = r_fac_fid*(mod.masked['xi_fid'])
        PNG_term = r_fac_c1*mod.masked['c1']*fNL + r_fac_c2*mod.masked['c2']*(fNL**2)
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
        r_fac_fid = np.ones(mod.N_obs_vec_masked)
        r_fac_c1 = np.ones(mod.N_obs_vec_masked)
        r_fac_c2 = np.ones(mod.N_obs_vec_masked)
        
        r_fac_fid[mod.term_masks['xi0']] = (b1g**2 + (2/3)*b1g*f_g + (f_g**2)/5)/(b1g_fid**2 + (2/3)*b1g_fid*f_fid + (f_fid**2)/5)
        r_fac_fid[mod.term_masks['xi2']] = ( (4/3)*b1g*f_g + (4/7)*(f_g**2) )/( (4/3)*b1g_fid*f_fid + (4/7)*(f_fid**2) )
        r_fac_fid[mod.term_masks['xi4']] = (f_g/f_fid)**2
    
        r_fac_c1[mod.term_masks['xi0']] = ((b1g + f_g/3)*(b1g-pg)*(mod.Om_m0_g/Dz_g))/\
                                ((b1g_fid + f_fid/3)*(b1g_fid-pfid)*(mod.Om_m0_fid/Dz_fid))
        r_fac_c2[mod.term_masks['xi0']] = (((b1g-pg)*(mod.Om_m0_g/Dz_g))**2)/(((b1g_fid-pfid)*(mod.Om_m0_fid/Dz_fid))**2)
        r_fac_c1[mod.term_masks['xi2']] = (f_g*(b1g-pg)*(mod.Om_m0_g/Dz_g))/(f_fid*(b1g_fid-pfid)*(mod.Om_m0_fid/Dz_fid))
        #################################    
        fid_term = r_fac_fid*(mod.masked['xi_fid'])
        PNG_term = r_fac_c1*mod.masked['c1']*fNL + r_fac_c2*mod.masked['c2']*(fNL**2)
        sys_term = r_fac_fid*((mod.masked['pvar_par_A1']*Psys1**2+mod.masked['pvar_par_B1']*Psys1) +\
                              (mod.masked['pvar_par_A2']*Psys2**2+mod.masked['pvar_par_B2']*Psys2) +\
                              (mod.masked['pvar_par_A3']*Psys3**2+mod.masked['pvar_par_B3']*Psys3))
        return fid_term + PNG_term + sys_term

# class DR2_LRG:
#     parameter_defaults = pd.DataFrame(columns=['key', 'init', 'prior', 'plot_label', 'num_decimals', 'unit'])
#     parameter_defaults = parameter_defaults.set_index('key')
#     parameter_defaults.loc['fNL'] = [0, [-250, 250,'flat'], r'$f_{NL}$', 0, '']
#     parameter_defaults.loc['b0g'] = [1, [0.5, 4,'flat'], r'$b_{0g}$', 2, '']
#     parameter_defaults.loc['b0gfid'] = [1, [1.94,0.04,'gauss'], r'$b_{0g}^{fid}$', 2, '']
#     parameter_defaults.loc['b0gpng'] = [1, [1.94,0.04,'gauss'], r'$b_{0g}^{png}$', 2, '']
#     parameter_defaults.loc['pg'] = [1, [1,0.1,'gauss'], r'$p_g$', 1,'']
#     parameter_defaults.loc['ppng'] = [1, [1,0.1,'gauss'], r'$p_{png}$', 1,'']
#     parameter_defaults.loc['KsysSGC'] = [1, [0,10,'gauss'], r'$K_{\mathrm{SGC}}$', 1, r'\%']
#     parameter_defaults.loc['KsysDEC'] = [1, [0,10,'gauss'], r'$K_{\mathrm{DEC}}$', 1, r'\%']
#     parameter_defaults.loc['KsysMZLS'] = [1, [0,10,'gauss'], r'$K_{\mathrm{MZLS}}$', 1, r'\%']

#     extra_parameters = {'z_eff', 'z_fid', 'z_png', 'Om_m0_g', 'Om_m0_fid', 'Om_m0_png', 'H0', 'H0_fid', 'H0_png'}
#     # z_eff, z_fid should both be dicts

#     @staticmethod
#     def xi_modded_base_pars(mod, params):
#         fNL, b0g, b0g_fid, b0g_png, pg, ppng, Psys1, Psys2, Psys3 = params

#         ells = [0, 2, 4]        
#         f_g = {ell: Omega_m_z(mod.z_eff[ell],mod.Om_m0_g)**0.55 for ell in ells}
#         f_fid = {ell: Omega_m_z(mod.z_fid[ell],mod.Om_m0_fid)**0.55 for ell in ells}
#         f_png = {ell: Omega_m_z(mod.z_png[ell],mod.Om_m0_png)**0.55 for ell in ells}
#         Dz_g = {ell: Dz_norm(mod.z_eff[ell],Om_m0=mod.Om_m0_g) for ell in ells}
#         Dz_fid = {ell: Dz_norm(mod.z_fid[ell],Om_m0=mod.Om_m0_fid) for ell in ells}
#         Dz_png = {ell: Dz_norm(mod.z_png[ell],Om_m0=mod.Om_m0_png) for ell in ells}
        
#         ### Define rescale factors ######
#         r_fac_fid = np.ones(mod.N_obs_vec_masked)
#         r_fac_fid_png = np.ones(mod.N_obs_vec_masked)
#         r_fac_c1 = np.ones(mod.N_obs_vec_masked)
#         r_fac_c2 = np.ones(mod.N_obs_vec_masked)
        
#         r_fac_fid[mod.term_masks['xi0']] = ((b0g/Dz_g[0])**2 + (2/3)*(b0g/Dz_g[0])*f_g[0] + (f_g[0]**2)/5)/\
#                                                 ((b0g_fid/Dz_fid[0])**2 + (2/3)*(b0g_fid/Dz_fid[0])*f_fid[0] + (f_fid[0]**2)/5)
#         r_fac_fid[mod.term_masks['xi2']] = ( (4/3)*(b0g/Dz_g[2])*f_g[2] + (4/7)*(f_g[2]**2) )/\
#                                                 ( (4/3)*(b0g_fid/Dz_fid[2])*f_fid[2] + (4/7)*(f_fid[2]**2) )
#         r_fac_fid[mod.term_masks['xi4']] = (f_g[4]/f_fid[4])**2

#         r_fac_fid_png[mod.term_masks['xi0']] = ((b0g/Dz_g[0])**2 + (2/3)*(b0g/Dz_g[0])*f_g[0] + (f_g[0]**2)/5)/\
#                                                 ((b0g_png/Dz_png[0])**2 + (2/3)*(b0g_png/Dz_png[0])*f_png[0] + (f_png[0]**2)/5)
#         r_fac_fid_png[mod.term_masks['xi2']] = ( (4/3)*(b0g/Dz_g[2])*f_g[2] + (4/7)*(f_g[2]**2) )/\
#                                                 ( (4/3)*(b0g_png/Dz_png[2])*f_png[2] + (4/7)*(f_png[2]**2) )
#         r_fac_fid_png[mod.term_masks['xi4']] = (f_g[4]/f_png[4])**2
    
#         r_fac_c1[mod.term_masks['xi0']] = (((b0g/Dz_g[0]) + f_g[0]/3)*((b0g/Dz_g[0])-pg)*((mod.Om_m0_g*mod.H0**2)/Dz_g[0]))/\
#                                                 (((b0g_png/Dz_png[0]) + f_png[0]/3)*((b0g_png/Dz_png[0])-ppng)*\
#                                                      ((mod.Om_m0_png*mod.H0_png**2)/Dz_png[0]))
#         r_fac_c2[mod.term_masks['xi0']] = ((((b0g/Dz_g[0])-pg)*((mod.Om_m0_g*mod.H0**2)/Dz_g[0]))**2)/\
#                                                 ((((b0g_png/Dz_png[0])-ppng)*((mod.Om_m0_png*mod.H0_png**2)/Dz_png[0]))**2)
#         r_fac_c1[mod.term_masks['xi2']] = (f_g[2]*((b0g/Dz_g[2])-pg)*((mod.Om_m0_g*mod.H0**2)/Dz_g[2]))/\
#                                                 (f_png[2]*((b0g_png/Dz_png[2])-ppng)*((mod.Om_m0_png*mod.H0_png**2)/Dz_png[2]))
#         #################################     
#         fid_term = r_fac_fid*(mod.masked['xi_fid'])
#         PNG_term = r_fac_c1*mod.masked['c1']*fNL + r_fac_c2*mod.masked['c2']*(fNL**2)
#         sys_term = r_fac_fid_png*((mod.masked['pvar_par_A1']*Psys1**2+mod.masked['pvar_par_B1']*Psys1) +\
#                               (mod.masked['pvar_par_A2']*Psys2**2+mod.masked['pvar_par_B2']*Psys2) +\
#                               (mod.masked['pvar_par_A3']*Psys3**2+mod.masked['pvar_par_B3']*Psys3))
#         return fid_term + PNG_term + sys_term

# class DR2_QSO:
#     parameter_defaults = pd.DataFrame(columns=['key', 'init', 'prior', 'plot_label', 'num_decimals', 'unit'])
#     parameter_defaults = parameter_defaults.set_index('key')
#     parameter_defaults.loc['fNL'] = [0, [-250, 250,'flat'], r'$f_{NL}$', 0, '']
#     parameter_defaults.loc['b0g'] = [1, [0.5, 4,'flat'], r'$b_{0g}$', 2, '']
#     parameter_defaults.loc['b0gfid'] = [1, [1.94,0.04,'gauss'], r'$b_{0g}^{fid}$', 2, '']
#     parameter_defaults.loc['b0gpng'] = [1, [1.94,0.04,'gauss'], r'$b_{0g}^{png}$', 2, '']
#     parameter_defaults.loc['pg'] = [1, [1,0.1,'gauss'], r'$p_g$', 1,'']
#     parameter_defaults.loc['ppng'] = [1, [1,0.1,'gauss'], r'$p_{png}$', 1,'']
#     parameter_defaults.loc['KsysSGC'] = [1, [0,10,'gauss'], r'$K_{\mathrm{SGC}}$', 1, r'\%']
#     parameter_defaults.loc['KsysDEC'] = [1, [0,10,'gauss'], r'$K_{\mathrm{DEC}}$', 1, r'\%']
#     parameter_defaults.loc['KsysMZLS'] = [1, [0,10,'gauss'], r'$K_{\mathrm{MZLS}}$', 1, r'\%']
#     parameter_defaults.loc['KsysDES'] = [1, [0,10,'gauss'], r'$K_{\mathrm{DES}}$', 1, r'\%']
    

#     extra_parameters = {'z_eff', 'z_fid', 'z_png', 'Om_m0_g', 'Om_m0_fid', 'Om_m0_png', 'H0', 'H0_fid', 'H0_png'}
#     # z_eff, z_fid should both be dicts

#     @staticmethod
#     def xi_modded_base_pars(mod, params):
#         fNL, b0g, b0g_fid, b0g_png, pg, ppng, Psys1, Psys2, Psys3, Psys4 = params
        
#         ells = [0, 2, 4]        
#         f_g = {ell: Omega_m_z(mod.z_eff[ell],mod.Om_m0_g)**0.55 for ell in ells}
#         f_fid = {ell: Omega_m_z(mod.z_fid[ell],mod.Om_m0_fid)**0.55 for ell in ells}
#         f_png = {ell: Omega_m_z(mod.z_png[ell],mod.Om_m0_png)**0.55 for ell in ells}
#         Dz_g = {ell: Dz_norm(mod.z_eff[ell],Om_m0=mod.Om_m0_g) for ell in ells}
#         Dz_fid = {ell: Dz_norm(mod.z_fid[ell],Om_m0=mod.Om_m0_fid) for ell in ells}
#         Dz_png = {ell: Dz_norm(mod.z_png[ell],Om_m0=mod.Om_m0_png) for ell in ells}
        
#         ### Define rescale factors ######
#         r_fac_fid = np.ones(mod.N_obs_vec_masked)
#         r_fac_fid_png = np.ones(mod.N_obs_vec_masked)
#         r_fac_c1 = np.ones(mod.N_obs_vec_masked)
#         r_fac_c2 = np.ones(mod.N_obs_vec_masked)
        
#         r_fac_fid[mod.term_masks['xi0']] = ((b0g/Dz_g[0])**2 + (2/3)*(b0g/Dz_g[0])*f_g[0] + (f_g[0]**2)/5)/\
#                                                 ((b0g_fid/Dz_fid[0])**2 + (2/3)*(b0g_fid/Dz_fid[0])*f_fid[0] + (f_fid[0]**2)/5)
#         r_fac_fid[mod.term_masks['xi2']] = ( (4/3)*(b0g/Dz_g[2])*f_g[2] + (4/7)*(f_g[2]**2) )/\
#                                                 ( (4/3)*(b0g_fid/Dz_fid[2])*f_fid[2] + (4/7)*(f_fid[2]**2) )
#         r_fac_fid[mod.term_masks['xi4']] = (f_g[4]/f_fid[4])**2

#         r_fac_fid_png[mod.term_masks['xi0']] = ((b0g/Dz_g[0])**2 + (2/3)*(b0g/Dz_g[0])*f_g[0] + (f_g[0]**2)/5)/\
#                                                 ((b0g_png/Dz_png[0])**2 + (2/3)*(b0g_png/Dz_png[0])*f_png[0] + (f_png[0]**2)/5)
#         r_fac_fid_png[mod.term_masks['xi2']] = ( (4/3)*(b0g/Dz_g[2])*f_g[2] + (4/7)*(f_g[2]**2) )/\
#                                                 ( (4/3)*(b0g_png/Dz_png[2])*f_png[2] + (4/7)*(f_png[2]**2) )
#         r_fac_fid_png[mod.term_masks['xi4']] = (f_g[4]/f_png[4])**2
    
#         r_fac_c1[mod.term_masks['xi0']] = (((b0g/Dz_g[0]) + f_g[0]/3)*((b0g/Dz_g[0])-pg)*((mod.Om_m0_g*mod.H0**2)/Dz_g[0]))/\
#                                                 (((b0g_png/Dz_png[0]) + f_png[0]/3)*((b0g_png/Dz_png[0])-ppng)*\
#                                                      ((mod.Om_m0_png*mod.H0_png**2)/Dz_png[0]))
#         r_fac_c2[mod.term_masks['xi0']] = ((((b0g/Dz_g[0])-pg)*((mod.Om_m0_g*mod.H0**2)/Dz_g[0]))**2)/\
#                                                 ((((b0g_png/Dz_png[0])-ppng)*((mod.Om_m0_png*mod.H0_png**2)/Dz_png[0]))**2)
#         r_fac_c1[mod.term_masks['xi2']] = (f_g[2]*((b0g/Dz_g[2])-pg)*((mod.Om_m0_g*mod.H0**2)/Dz_g[2]))/\
#                                                 (f_png[2]*((b0g_png/Dz_png[2])-ppng)*((mod.Om_m0_png*mod.H0_png**2)/Dz_png[2]))
#         #################################    
#         fid_term = r_fac_fid*(mod.masked['xi_fid'])
#         PNG_term = r_fac_c1*mod.masked['c1']*fNL + r_fac_c2*mod.masked['c2']*(fNL**2)
#         sys_term = r_fac_fid_png*((mod.masked['pvar_par_A1']*Psys1**2+mod.masked['pvar_par_B1']*Psys1) +\
#                               (mod.masked['pvar_par_A2']*Psys2**2+mod.masked['pvar_par_B2']*Psys2) +\
#                               (mod.masked['pvar_par_A3']*Psys3**2+mod.masked['pvar_par_B3']*Psys3) +\
#                               (mod.masked['pvar_par_A4']*Psys3**2+mod.masked['pvar_par_B4']*Psys3))
#         return fid_term + PNG_term + sys_term

class DR2_LRG:
    parameter_defaults = pd.DataFrame(columns=['key', 'init', 'prior', 'plot_label', 'num_decimals', 'unit'])
    parameter_defaults = parameter_defaults.set_index('key')
    parameter_defaults.loc['fNL'] = [0, [-250, 250,'flat'], r'$f_{NL}$', 0, '']
    parameter_defaults.loc['b0g'] = [1, [0.5, 4,'flat'], r'$b_{0g}$', 2, '']
    parameter_defaults.loc['b0gfid'] = [1, [1.94,0.04,'gauss'], r'$b_{0g}^{fid}$', 2, '']
    parameter_defaults.loc['b0gpng'] = [1, [1.94,0.04,'gauss'], r'$b_{0g}^{png}$', 2, '']
    parameter_defaults.loc['KsysSGC'] = [1, [0,10,'gauss'], r'$K_{\mathrm{SGC}}$', 1, r'\%']
    parameter_defaults.loc['KsysDEC'] = [1, [0,10,'gauss'], r'$K_{\mathrm{DEC}}$', 1, r'\%']
    parameter_defaults.loc['KsysMZLS'] = [1, [0,10,'gauss'], r'$K_{\mathrm{MZLS}}$', 1, r'\%']
    parameter_defaults.loc['Kregr'] = [0, [0,0.1,'gauss'], r'$K_{\mathrm{regr}}$', 1, r'\%']

    extra_parameters = {'z_eff', 'z_fid', 'z_png', 'Om_m0_g', 'Om_m0_fid', 'Om_m0_png', 'H0', 'H0_fid', 'H0_png', 'pg', 'ppng'}
    # z_eff, z_fid should both be dicts

    @staticmethod
    def xi_modded_base_pars(mod, params):
        fNL, b0g, b0g_fid, b0g_png, Psys1, Psys2, Psys3, Kregr = params

        ells = [0, 2, 4]        
        f_g = {ell: Omega_m_z(mod.z_eff[ell],mod.Om_m0_g)**0.55 for ell in ells}
        f_fid = {ell: Omega_m_z(mod.z_fid[ell],mod.Om_m0_fid)**0.55 for ell in ells}
        f_png = {ell: Omega_m_z(mod.z_png[ell],mod.Om_m0_png)**0.55 for ell in ells}
        Dz_g = {ell: Dz_norm(mod.z_eff[ell],Om_m0=mod.Om_m0_g) for ell in ells}
        Dz_fid = {ell: Dz_norm(mod.z_fid[ell],Om_m0=mod.Om_m0_fid) for ell in ells}
        Dz_png = {ell: Dz_norm(mod.z_png[ell],Om_m0=mod.Om_m0_png) for ell in ells}
        
        ### Define rescale factors ######
        r_fac_fid = np.ones(mod.N_obs_vec_masked)
        r_fac_fid_png = np.ones(mod.N_obs_vec_masked)
        r_fac_c1 = np.ones(mod.N_obs_vec_masked)
        r_fac_c2 = np.ones(mod.N_obs_vec_masked)
        
        r_fac_fid[mod.term_masks['xi0']] = ((b0g/Dz_g[0])**2 + (2/3)*(b0g/Dz_g[0])*f_g[0] + (f_g[0]**2)/5)/\
                                                ((b0g_fid/Dz_fid[0])**2 + (2/3)*(b0g_fid/Dz_fid[0])*f_fid[0] + (f_fid[0]**2)/5)
        r_fac_fid[mod.term_masks['xi2']] = ( (4/3)*(b0g/Dz_g[2])*f_g[2] + (4/7)*(f_g[2]**2) )/\
                                                ( (4/3)*(b0g_fid/Dz_fid[2])*f_fid[2] + (4/7)*(f_fid[2]**2) )
        r_fac_fid[mod.term_masks['xi4']] = (f_g[4]/f_fid[4])**2

        r_fac_fid_png[mod.term_masks['xi0']] = ((b0g/Dz_g[0])**2 + (2/3)*(b0g/Dz_g[0])*f_g[0] + (f_g[0]**2)/5)/\
                                                ((b0g_png/Dz_png[0])**2 + (2/3)*(b0g_png/Dz_png[0])*f_png[0] + (f_png[0]**2)/5)
        r_fac_fid_png[mod.term_masks['xi2']] = ( (4/3)*(b0g/Dz_g[2])*f_g[2] + (4/7)*(f_g[2]**2) )/\
                                                ( (4/3)*(b0g_png/Dz_png[2])*f_png[2] + (4/7)*(f_png[2]**2) )
        r_fac_fid_png[mod.term_masks['xi4']] = (f_g[4]/f_png[4])**2
    
        r_fac_c1[mod.term_masks['xi0']] = (((b0g/Dz_g[0]) + f_g[0]/3)*((b0g/Dz_g[0])-mod.pg)*((mod.Om_m0_g*mod.H0**2)/Dz_g[0]))/\
                                                (((b0g_png/Dz_png[0]) + f_png[0]/3)*((b0g_png/Dz_png[0])-mod.ppng)*\
                                                     ((mod.Om_m0_png*mod.H0_png**2)/Dz_png[0]))
        r_fac_c2[mod.term_masks['xi0']] = ((((b0g/Dz_g[0])-mod.pg)*((mod.Om_m0_g*mod.H0**2)/Dz_g[0]))**2)/\
                                                ((((b0g_png/Dz_png[0])-mod.ppng)*((mod.Om_m0_png*mod.H0_png**2)/Dz_png[0]))**2)
        r_fac_c1[mod.term_masks['xi2']] = (f_g[2]*((b0g/Dz_g[2])-mod.pg)*((mod.Om_m0_g*mod.H0**2)/Dz_g[2]))/\
                                                (f_png[2]*((b0g_png/Dz_png[2])-mod.ppng)*((mod.Om_m0_png*mod.H0_png**2)/Dz_png[2]))
        #################################     
        fid_term = r_fac_fid*(mod.masked['xi_fid'])
        PNG_term = r_fac_c1*mod.masked['c1']*fNL + r_fac_c2*mod.masked['c2']*(fNL**2)
        sys_term = r_fac_fid_png*((mod.masked['pvar_par_A1']*Psys1**2+mod.masked['pvar_par_B1']*Psys1) +\
                              (mod.masked['pvar_par_A2']*Psys2**2+mod.masked['pvar_par_B2']*Psys2) +\
                              (mod.masked['pvar_par_A3']*Psys3**2+mod.masked['pvar_par_B3']*Psys3))
        regression_term = r_fac_fid_png*mod.masked['pvar_par_Kregr']*Kregr
        return fid_term + PNG_term + sys_term + regression_term

class DR2_QSO:
    parameter_defaults = pd.DataFrame(columns=['key', 'init', 'prior', 'plot_label', 'num_decimals', 'unit'])
    parameter_defaults = parameter_defaults.set_index('key')
    parameter_defaults.loc['fNL'] = [0, [-250, 250,'flat'], r'$f_{NL}$', 0, '']
    parameter_defaults.loc['b0g'] = [1, [0.5, 4,'flat'], r'$b_{0g}$', 2, '']
    parameter_defaults.loc['b0gfid'] = [1, [1.94,0.04,'gauss'], r'$b_{0g}^{fid}$', 2, '']
    parameter_defaults.loc['b0gpng'] = [1, [1.94,0.04,'gauss'], r'$b_{0g}^{png}$', 2, '']
    parameter_defaults.loc['KsysSGC'] = [1, [0,10,'gauss'], r'$K_{\mathrm{SGC}}$', 1, r'\%']
    parameter_defaults.loc['KsysDEC'] = [1, [0,10,'gauss'], r'$K_{\mathrm{DEC}}$', 1, r'\%']
    parameter_defaults.loc['KsysMZLS'] = [1, [0,10,'gauss'], r'$K_{\mathrm{MZLS}}$', 1, r'\%']
    parameter_defaults.loc['KsysDES'] = [1, [0,10,'gauss'], r'$K_{\mathrm{DES}}$', 1, r'\%']
    parameter_defaults.loc['Kregr'] = [0, [0,0.1,'gauss'], r'$K_{\mathrm{regr}}$', 1, r'\%']
    
    extra_parameters = {'z_eff', 'z_fid', 'z_png', 'Om_m0_g', 'Om_m0_fid', 'Om_m0_png', 'H0', 'H0_fid', 'H0_png', 'pg', 'ppng'}
    # z_eff, z_fid should both be dicts

    @staticmethod
    def xi_modded_base_pars(mod, params):
        fNL, b0g, b0g_fid, b0g_png, Psys1, Psys2, Psys3, Psys4, Kregr = params
        
        ells = [0, 2, 4]        
        f_g = {ell: Omega_m_z(mod.z_eff[ell],mod.Om_m0_g)**0.55 for ell in ells}
        f_fid = {ell: Omega_m_z(mod.z_fid[ell],mod.Om_m0_fid)**0.55 for ell in ells}
        f_png = {ell: Omega_m_z(mod.z_png[ell],mod.Om_m0_png)**0.55 for ell in ells}
        Dz_g = {ell: Dz_norm(mod.z_eff[ell],Om_m0=mod.Om_m0_g) for ell in ells}
        Dz_fid = {ell: Dz_norm(mod.z_fid[ell],Om_m0=mod.Om_m0_fid) for ell in ells}
        Dz_png = {ell: Dz_norm(mod.z_png[ell],Om_m0=mod.Om_m0_png) for ell in ells}
        
        ### Define rescale factors ######
        r_fac_fid = np.ones(mod.N_obs_vec_masked)
        r_fac_fid_png = np.ones(mod.N_obs_vec_masked)
        r_fac_c1 = np.ones(mod.N_obs_vec_masked)
        r_fac_c2 = np.ones(mod.N_obs_vec_masked)
        
        r_fac_fid[mod.term_masks['xi0']] = ((b0g/Dz_g[0])**2 + (2/3)*(b0g/Dz_g[0])*f_g[0] + (f_g[0]**2)/5)/\
                                                ((b0g_fid/Dz_fid[0])**2 + (2/3)*(b0g_fid/Dz_fid[0])*f_fid[0] + (f_fid[0]**2)/5)
        r_fac_fid[mod.term_masks['xi2']] = ( (4/3)*(b0g/Dz_g[2])*f_g[2] + (4/7)*(f_g[2]**2) )/\
                                                ( (4/3)*(b0g_fid/Dz_fid[2])*f_fid[2] + (4/7)*(f_fid[2]**2) )
        r_fac_fid[mod.term_masks['xi4']] = (f_g[4]/f_fid[4])**2

        r_fac_fid_png[mod.term_masks['xi0']] = ((b0g/Dz_g[0])**2 + (2/3)*(b0g/Dz_g[0])*f_g[0] + (f_g[0]**2)/5)/\
                                                ((b0g_png/Dz_png[0])**2 + (2/3)*(b0g_png/Dz_png[0])*f_png[0] + (f_png[0]**2)/5)
        r_fac_fid_png[mod.term_masks['xi2']] = ( (4/3)*(b0g/Dz_g[2])*f_g[2] + (4/7)*(f_g[2]**2) )/\
                                                ( (4/3)*(b0g_png/Dz_png[2])*f_png[2] + (4/7)*(f_png[2]**2) )
        r_fac_fid_png[mod.term_masks['xi4']] = (f_g[4]/f_png[4])**2
    
        r_fac_c1[mod.term_masks['xi0']] = (((b0g/Dz_g[0]) + f_g[0]/3)*((b0g/Dz_g[0])-mod.pg)*((mod.Om_m0_g*mod.H0**2)/Dz_g[0]))/\
                                                (((b0g_png/Dz_png[0]) + f_png[0]/3)*((b0g_png/Dz_png[0])-mod.ppng)*\
                                                     ((mod.Om_m0_png*mod.H0_png**2)/Dz_png[0]))
        r_fac_c2[mod.term_masks['xi0']] = ((((b0g/Dz_g[0])-mod.pg)*((mod.Om_m0_g*mod.H0**2)/Dz_g[0]))**2)/\
                                                ((((b0g_png/Dz_png[0])-mod.ppng)*((mod.Om_m0_png*mod.H0_png**2)/Dz_png[0]))**2)
        r_fac_c1[mod.term_masks['xi2']] = (f_g[2]*((b0g/Dz_g[2])-mod.pg)*((mod.Om_m0_g*mod.H0**2)/Dz_g[2]))/\
                                                (f_png[2]*((b0g_png/Dz_png[2])-mod.ppng)*((mod.Om_m0_png*mod.H0_png**2)/Dz_png[2]))
        #################################    
        fid_term = r_fac_fid*(mod.masked['xi_fid'])
        PNG_term = r_fac_c1*mod.masked['c1']*fNL + r_fac_c2*mod.masked['c2']*(fNL**2)
        sys_term = r_fac_fid_png*((mod.masked['pvar_par_A1']*Psys1**2+mod.masked['pvar_par_B1']*Psys1) +\
                              (mod.masked['pvar_par_A2']*Psys2**2+mod.masked['pvar_par_B2']*Psys2) +\
                              (mod.masked['pvar_par_A3']*Psys3**2+mod.masked['pvar_par_B3']*Psys3) +\
                              (mod.masked['pvar_par_A4']*Psys3**2+mod.masked['pvar_par_B4']*Psys3))
        regression_term = r_fac_fid_png*mod.masked['pvar_par_Kregr']*Kregr
        return fid_term + PNG_term + sys_term + regression_term

class DR2_cross:
    parameter_defaults = pd.DataFrame(columns=['key', 'init', 'prior', 'plot_label', 'num_decimals', 'unit'])
    parameter_defaults = parameter_defaults.set_index('key')
    parameter_defaults.loc['fNL'] = [0, [-250, 250,'flat'], r'$f_{NL}$', 0, '']
    parameter_defaults.loc['b0g_LRG'] = [1, [0, 5,'flat'], r'$b_{LRG}$', 2, ''] # made upper bound higher because QSO b w/OQE ~3
    parameter_defaults.loc['b0gfid_LRG'] = [1, [1.94,0.04,'gauss'], r'$b_{LRG}^{fid}$', 2, '']
    parameter_defaults.loc['b0gpng_LRG'] = [1, [1.94,0.04,'gauss'], r'$b_{LRG}^{PNG}$', 2, '']
    parameter_defaults.loc['b0g_QSO'] = [1, [0, 5,'flat'], r'$b_{QSO}$', 2, ''] # made upper bound higher because QSO b w/OQE ~3
    parameter_defaults.loc['b0gfid_QSO'] = [1, [1.94,0.04,'gauss'], r'$b_{QSO}^{fid}$', 2, '']
    parameter_defaults.loc['b0gpng_QSO'] = [1, [1.94,0.04,'gauss'], r'$b_{QSO}^{PNG}$', 2, '']
    parameter_defaults.loc['Kregr'] = [0, [0,0.1,'gauss'], r'$K_{\mathrm{regr}}$', 1, r'\%']
    
    extra_parameters = {'z_eff', 'z_fid', 'z_png', 'Om_m0_g', 'Om_m0_fid', 'Om_m0_png', 'H0', 'H0_fid', 'H0_png',
                        'pg_LRG', 'pg_QSO', 'ppng_LRG', 'ppng_QSO'}
    # z_eff, z_fid should both be dicts. Here zeff represents zeff of the cross tracer sample
    
    @staticmethod
    def xi_modded_base_pars(mod, params):
        fNL, b0g_LRG, b0g_fid_LRG, b0g_png_LRG, b0g_QSO, b0g_fid_QSO, b0g_png_QSO, Kregr = params
        
        ells = [0, 2, 4]        
        f_g = {ell: Omega_m_z(mod.z_eff[ell],mod.Om_m0_g)**0.55 for ell in ells}
        f_fid = {ell: Omega_m_z(mod.z_fid[ell],mod.Om_m0_fid)**0.55 for ell in ells}
        f_png = {ell: Omega_m_z(mod.z_png[ell],mod.Om_m0_png)**0.55 for ell in ells}
        Dz_g = {ell: Dz_norm(mod.z_eff[ell],Om_m0=mod.Om_m0_g) for ell in ells}
        Dz_fid = {ell: Dz_norm(mod.z_fid[ell],Om_m0=mod.Om_m0_fid) for ell in ells}
        Dz_png = {ell: Dz_norm(mod.z_png[ell],Om_m0=mod.Om_m0_png) for ell in ells}
        
        ### Define rescale factors ######
        r_fac_fid = np.ones(mod.N_obs_vec_masked)
        r_fac_fid_png = np.ones(mod.N_obs_vec_masked)
        r_fac_c1 = np.ones(mod.N_obs_vec_masked)
        r_fac_c2 = np.ones(mod.N_obs_vec_masked)
        
        r_fac_fid[mod.term_masks['xi0']] = ((b0g_LRG*b0g_QSO)/(Dz_g[0]**2) + \
                                            (b0g_LRG + b0g_QSO)*(f_g[0]/Dz_g[0])/3 + (f_g[0]**2)/5)/\
                                           ((b0g_fid_LRG*b0g_fid_QSO)/(Dz_fid[0]**2) + \
                                            (b0g_fid_LRG + b0g_fid_QSO)*(f_fid[0]/Dz_fid[0])/3 +(f_fid[0]**2)/5)
        r_fac_fid[mod.term_masks['xi2']] = ( (2/3)*(b0g_LRG + b0g_QSO)*(f_g[2]/Dz_g[2]) + (4/7)*(f_g[2]**2) )/\
                                                ( (2/3)*(b0g_fid_LRG+b0g_fid_QSO)*(f_fid[2]/Dz_fid[2]) + (4/7)*(f_fid[2]**2) )
        r_fac_fid[mod.term_masks['xi4']] = (f_g[4]/f_fid[4])**2

        r_fac_fid_png[mod.term_masks['xi0']] = ((b0g_LRG*b0g_QSO)/(Dz_g[0]**2) + \
                                            (b0g_LRG + b0g_QSO)*(f_g[0]/Dz_g[0])/3 + (f_g[0]**2)/5)/\
                                           ((b0g_png_LRG*b0g_png_QSO)/(Dz_png[0]**2) + \
                                            (b0g_png_LRG + b0g_png_QSO)*(f_png[0]/Dz_png[0])/3 +(f_png[0]**2)/5)
        r_fac_fid_png[mod.term_masks['xi2']] = ( (2/3)*(b0g_LRG + b0g_QSO)*(f_g[2]/Dz_g[2]) + (4/7)*(f_g[2]**2) )/\
                                                ( (2/3)*(b0g_png_LRG+b0g_png_QSO)*(f_png[2]/Dz_png[2]) + (4/7)*(f_png[2]**2) )
        r_fac_fid_png[mod.term_masks['xi4']] = (f_g[4]/f_png[4])**2
    
        r_fac_c1[mod.term_masks['xi0']] = ((((b0g_LRG/Dz_g[0])+f_g[0]/3)*((b0g_QSO/Dz_g[0])-mod.pg_QSO)+\
                                            ((b0g_QSO/Dz_g[0])+f_g[0]/3)*((b0g_LRG/Dz_g[0])-mod.pg_LRG))*\
                                            ((mod.Om_m0_g*mod.H0**2)/Dz_g[0]) )/\
                                            ((((b0g_png_LRG/Dz_png[0]) + f_png[0]/3)*((b0g_png_QSO/Dz_png[0])-mod.ppng_QSO) +\
                                             ((b0g_png_QSO/Dz_png[0]) + f_png[0]/3)*((b0g_png_LRG/Dz_png[0])-mod.ppng_LRG))*\
                                            ((mod.Om_m0_png*mod.H0_png**2)/Dz_png[0]))
        
        r_fac_c2[mod.term_masks['xi0']] = (((b0g_LRG/Dz_g[0])-mod.pg_LRG)*((b0g_QSO/Dz_g[0])-mod.pg_QSO)*\
                                           ((mod.Om_m0_g*mod.H0**2)/Dz_g[0])**2)/\
                                            (((b0g_png_LRG/Dz_png[0])-mod.ppng_LRG)*((b0g_png_QSO/Dz_png[0])-mod.ppng_QSO)*\
                                             ((mod.Om_m0_png*mod.H0_png**2)/Dz_png[0])**2)
    
        r_fac_c1[mod.term_masks['xi2']] = (f_g[2]*(((b0g_LRG/Dz_g[2])-mod.pg_LRG)+((b0g_QSO/Dz_g[2])-mod.pg_QSO))*\
                                           ((mod.Om_m0_g*mod.H0**2)/Dz_g[2]))/\
                                        (f_png[2]*(((b0g_png_LRG/Dz_png[2])-mod.ppng_LRG)+((b0g_png_QSO/Dz_png[2])-mod.ppng_QSO))*\
                                         ((mod.Om_m0_png*mod.H0_png**2)/Dz_png[2]))
        #################################    
        fid_term = r_fac_fid*(mod.masked['xi_fid'])
        PNG_term = r_fac_c1*mod.masked['c1']*fNL + r_fac_c2*mod.masked['c2']*(fNL**2)
        regression_term = r_fac_fid_png*mod.masked['pvar_par_Kregr']*Kregr
        return fid_term + PNG_term + regression_term

class DR2_cross_all:
    parameter_defaults = pd.DataFrame(columns=['key', 'init', 'prior', 'plot_label', 'num_decimals', 'unit'])
    parameter_defaults = parameter_defaults.set_index('key')
    parameter_defaults.loc['fNL'] = [0, [-250, 250,'flat'], r'$f_{NL}$', 0, '']
    parameter_defaults.loc['b0g_LRG'] = [1, [0, 5,'flat'], r'$b_{LRG}$', 2, ''] # made upper bound higher because QSO b w/OQE ~3
    parameter_defaults.loc['b0gfid_LRG'] = [1, [1.94,0.04,'gauss'], r'$b_{LRG}^{fid}$', 2, '']
    parameter_defaults.loc['b0gpng_LRG'] = [1, [1.94,0.04,'gauss'], r'$b_{LRG}^{PNG}$', 2, '']
    parameter_defaults.loc['b0g_QSO'] = [1, [0, 5,'flat'], r'$b_{QSO}$', 2, ''] # made upper bound higher because QSO b w/OQE ~3
    parameter_defaults.loc['b0gfid_QSO'] = [1, [1.94,0.04,'gauss'], r'$b_{QSO}^{fid}$', 2, '']
    parameter_defaults.loc['b0gpng_QSO'] = [1, [1.94,0.04,'gauss'], r'$b_{QSO}^{PNG}$', 2, '']
    parameter_defaults.loc['KsysSGC_LRG'] = [1, [0,10,'gauss'], r'$K_{\mathrm{SGC}^\mathrm{LRG}}$', 1, r'\%']
    parameter_defaults.loc['KsysDEC_LRG'] = [1, [0,10,'gauss'], r'$K_{\mathrm{DEC}^\mathrm{LRG}}$', 1, r'\%']
    parameter_defaults.loc['KsysMZLS_LRG'] = [1, [0,10,'gauss'], r'$K_{\mathrm{MZLS}^\mathrm{LRG}}$', 1, r'\%']
    parameter_defaults.loc['KsysSGC_QSO'] = [1, [0,10,'gauss'], r'$K_{\mathrm{SGC}^\mathrm{QSO}}$', 1, r'\%']
    parameter_defaults.loc['KsysDEC_QSO'] = [1, [0,10,'gauss'], r'$K_{\mathrm{DEC}^\mathrm{QSO}}$', 1, r'\%']
    parameter_defaults.loc['KsysMZLS_QSO'] = [1, [0,10,'gauss'], r'$K_{\mathrm{MZLS}^\mathrm{QSO}}$', 1, r'\%']
    parameter_defaults.loc['KsysDES_QSO'] = [1, [0,10,'gauss'], r'$K_{\mathrm{DES}^\mathrm{QSO}}$', 1, r'\%']
    parameter_defaults.loc['Kregr'] = [0, [0,0.1,'gauss'], r'$K_{\mathrm{regr}}$', 1, r'\%']
    
    extra_parameters = {'z_eff', 'z_fid', 'z_png', 'Om_m0_g', 'Om_m0_fid', 'Om_m0_png', 'H0', 'H0_fid', 'H0_png',
                        'pg_LRG', 'pg_QSO', 'ppng_LRG', 'ppng_QSO'}
    
    @staticmethod
    def xi_modded_base_pars(mod, params):
        fNL, b0g_LRG, b0g_fid_LRG, b0g_png_LRG, b0g_QSO, b0g_fid_QSO, b0g_png_QSO, \
                Psys1_LRG, Psys2_LRG, Psys3_LRG, Psys1_QSO, Psys2_QSO, Psys3_QSO, Psys4_QSO, Kregr = params
        
        ells = [0, 2, 4]        
        f_g = {term: Omega_m_z(mod.z_eff[term],mod.Om_m0_g)**0.55 for term in mod.terms}
        f_fid = {term: Omega_m_z(mod.z_fid[term],mod.Om_m0_fid)**0.55 for term in mod.terms}
        f_png = {term: Omega_m_z(mod.z_png[term],mod.Om_m0_png)**0.55 for term in mod.terms}
        Dz_g = {term: Dz_norm(mod.z_eff[term],Om_m0=mod.Om_m0_g) for term in mod.terms}
        Dz_fid = {term: Dz_norm(mod.z_fid[term],Om_m0=mod.Om_m0_fid) for term in mod.terms}
        Dz_png = {term: Dz_norm(mod.z_png[term],Om_m0=mod.Om_m0_png) for term in mod.terms}
        
        ### Define rescale factors ######
        r_fac_fid = np.ones(mod.N_obs_vec_masked)    
        r_fac_c1 = np.ones(mod.N_obs_vec_masked)
        r_fac_c2 = np.ones(mod.N_obs_vec_masked)
        r_fac_fid_png = np.ones(mod.N_obs_vec_masked)
        
        r_fac_fid[mod.term_masks['LRG_ell0']] = ((b0g_LRG/Dz_g['LRG_ell0'])**2 + \
                                                 (2/3)*(b0g_LRG/Dz_g['LRG_ell0'])*f_g['LRG_ell0'] + (f_g['LRG_ell0']**2)/5)/\
                                                ((b0g_fid_LRG/Dz_fid['LRG_ell0'])**2 + \
                                                 (2/3)*(b0g_fid_LRG/Dz_fid['LRG_ell0'])*f_fid['LRG_ell0'] + (f_fid['LRG_ell0']**2)/5)
        r_fac_fid[mod.term_masks['LRGxQSO_ell0']] = ((b0g_LRG*b0g_QSO)/(Dz_g['LRGxQSO_ell0']**2) + \
                                                     (b0g_LRG + b0g_QSO)*(f_g['LRGxQSO_ell0']/Dz_g['LRGxQSO_ell0'])/3 +\
                                                     (f_g['LRGxQSO_ell0']**2)/5)/\
                                             ((b0g_fid_LRG*b0g_fid_QSO)/(Dz_fid['LRGxQSO_ell0']**2) + \
                                             (b0g_fid_LRG + b0g_QSO)*(f_fid['LRGxQSO_ell0']/Dz_fid['LRGxQSO_ell0'])/3 +\
                                              (f_fid['LRGxQSO_ell0']**2)/5)
        r_fac_fid[mod.term_masks['QSO_ell0']] = ((b0g_QSO/Dz_g['QSO_ell0'])**2 + \
                                                 (2/3)*(b0g_QSO/Dz_g['QSO_ell0'])*f_g['QSO_ell0'] + (f_g['QSO_ell0']**2)/5)/\
                                                ((b0g_fid_QSO/Dz_fid['QSO_ell0'])**2 + \
                                                 (2/3)*(b0g_fid_QSO/Dz_fid['QSO_ell0'])*f_fid['QSO_ell0'] + (f_fid['QSO_ell0']**2)/5)

        r_fac_fid[mod.term_masks['LRG_ell2']] = ( (4/3)*(b0g_LRG/Dz_g['LRG_ell2'])*f_g['LRG_ell2'] + (4/7)*(f_g['LRG_ell2']**2) )/\
                                                ( (4/3)*(b0g_fid_LRG/Dz_fid['LRG_ell2'])*f_fid['LRG_ell2'] + (4/7)*(f_fid['LRG_ell2']**2) )
        r_fac_fid[mod.term_masks['LRGxQSO_ell2']] = ( (2/3)*(b0g_LRG + b0g_QSO)*\
                                                      (f_g['LRGxQSO_ell2']/Dz_g['LRGxQSO_ell2']) + (4/7)*(f_g['LRGxQSO_ell2']**2) )/\
                                                ( (2/3)*(b0g_fid_LRG+b0g_fid_QSO)*\
                                                  (f_fid['LRGxQSO_ell2']/Dz_fid['LRGxQSO_ell2']) + (4/7)*(f_fid['LRGxQSO_ell2']**2) )
        r_fac_fid[mod.term_masks['QSO_ell2']] = ( (4/3)*(b0g_QSO/Dz_g['QSO_ell2'])*f_g['QSO_ell2'] + (4/7)*(f_g['QSO_ell2']**2) )/\
                                                ( (4/3)*(b0g_fid_QSO/Dz_fid['QSO_ell2'])*f_fid['QSO_ell2'] + (4/7)*(f_fid['QSO_ell2']**2) )

        r_fac_fid[mod.term_masks['LRG_ell4']] = (f_g['LRG_ell4']/f_fid['LRG_ell4'])**2
        r_fac_fid[mod.term_masks['LRGxQSO_ell4']] = (f_g['LRGxQSO_ell4']/f_fid['LRGxQSO_ell4'])**2
        r_fac_fid[mod.term_masks['QSO_ell4']] = (f_g['QSO_ell4']/f_fid['QSO_ell4'])**2

        r_fac_fid_png[mod.term_masks['LRG_ell0']] = ((b0g_LRG/Dz_g['LRG_ell0'])**2 + \
                                                     (2/3)*(b0g_LRG/Dz_g['LRG_ell0'])*f_g['LRG_ell0'] + (f_g['LRG_ell0']**2)/5)/\
                                                ((b0g_png_LRG/Dz_png['LRG_ell0'])**2 + \
                                                 (2/3)*(b0g_png_LRG/Dz_png['LRG_ell0'])*f_png['LRG_ell0'] + (f_png['LRG_ell0']**2)/5)
        r_fac_fid_png[mod.term_masks['LRGxQSO_ell0']] = ((b0g_LRG*b0g_QSO)/(Dz_g['LRGxQSO_ell0']**2) + \
                                                         (b0g_LRG + b0g_QSO)*(f_g['LRGxQSO_ell0']/Dz_g['LRGxQSO_ell0'])/3 + \
                                                         (f_g['LRGxQSO_ell0']**2)/5)/\
                                                        ((b0g_png_LRG*b0g_png_QSO)/(Dz_fid['LRGxQSO_ell0']**2) + \
                                                         (b0g_png_LRG + b0g_png_QSO)*\
                                                         (f_png['LRGxQSO_ell0']/Dz_png['LRGxQSO_ell0'])/3 +(f_png['LRGxQSO_ell0']**2)/5)
        r_fac_fid_png[mod.term_masks['QSO_ell0']] = ((b0g_QSO/Dz_g['QSO_ell0'])**2 + \
                                                     (2/3)*(b0g_QSO/Dz_g['QSO_ell0'])*f_g['QSO_ell0'] + (f_g['QSO_ell0']**2)/5)/\
                                                ((b0g_png_QSO/Dz_png['QSO_ell0'])**2 + \
                                                 (2/3)*(b0g_png_QSO/Dz_png['QSO_ell0'])*f_png['QSO_ell0'] + (f_png['QSO_ell0']**2)/5)

        r_fac_fid_png[mod.term_masks['LRG_ell2']] = ( (4/3)*(b0g_LRG/Dz_g['LRG_ell2'])*f_g['LRG_ell2'] + (4/7)*(f_g['LRG_ell2']**2) )/\
                                                ( (4/3)*(b0g_png_LRG/Dz_png['LRG_ell2'])*f_png['LRG_ell2'] + (4/7)*(f_png['LRG_ell2']**2) )
        r_fac_fid_png[mod.term_masks['LRGxQSO_ell2']] = ( (2/3)*(b0g_LRG + b0g_QSO)*\
                                                          (f_g['LRGxQSO_ell2']/Dz_g['LRGxQSO_ell2']) + (4/7)*(f_g['LRGxQSO_ell2']**2) )/\
                                                ( (2/3)*(b0g_png_LRG+b0g_png_QSO)*\
                                                  (f_png['LRGxQSO_ell2']/Dz_png['LRGxQSO_ell2']) + (4/7)*(f_png['LRGxQSO_ell2']**2) )
        r_fac_fid_png[mod.term_masks['QSO_ell2']] = ( (4/3)*(b0g_QSO/Dz_g['QSO_ell2'])*f_g['QSO_ell2'] + (4/7)*(f_g['QSO_ell2']**2) )/\
                                                ( (4/3)*(b0g_png_QSO/Dz_png['QSO_ell2'])*f_png['QSO_ell2'] + (4/7)*(f_png['QSO_ell2']**2) )

        r_fac_fid_png[mod.term_masks['LRG_ell4']] = (f_g['LRG_ell4']/f_png['LRG_ell4'])**2
        r_fac_fid_png[mod.term_masks['LRGxQSO_ell4']] = (f_g['LRGxQSO_ell4']/f_png['LRGxQSO_ell4'])**2
        r_fac_fid_png[mod.term_masks['QSO_ell4']] = (f_g['QSO_ell4']/f_png['QSO_ell4'])**2
        
        r_fac_c1[mod.term_masks['LRG_ell0']] = (((b0g_LRG/Dz_g['LRG_ell0']) + f_g['LRG_ell0']/3)*\
                                                ((b0g_LRG/Dz_g['LRG_ell0'])-mod.pg_LRG)*((mod.Om_m0_g*mod.H0**2)/Dz_g['LRG_ell0']))/\
                                                (((b0g_png_LRG/Dz_png['LRG_ell0']) + f_png['LRG_ell0']/3)*\
                                                 ((b0g_png_LRG/Dz_png['LRG_ell0'])-mod.ppng_LRG)*\
                                                     ((mod.Om_m0_png*mod.H0_png**2)/Dz_png['LRG_ell0']))
        r_fac_c1[mod.term_masks['LRGxQSO_ell0']] = ((((b0g_LRG/Dz_g['LRGxQSO_ell0'])+f_g['LRGxQSO_ell0']/3)*\
                                                     ((b0g_QSO/Dz_g['LRGxQSO_ell0'])-mod.pg_QSO)+\
                                            ((b0g_QSO/Dz_g['LRGxQSO_ell0'])+f_g['LRGxQSO_ell0']/3)*\
                                                     ((b0g_LRG/Dz_g['LRGxQSO_ell0'])-mod.pg_LRG))*\
                                            ((mod.Om_m0_g*mod.H0**2)/Dz_g['LRGxQSO_ell0']) )/\
                                            ((((b0g_png_LRG/Dz_png['LRGxQSO_ell0']) + f_png['LRGxQSO_ell0']/3)*\
                                              ((b0g_png_QSO/Dz_png['LRGxQSO_ell0'])-mod.ppng_QSO) +\
                                             ((b0g_png_QSO/Dz_png['LRGxQSO_ell0']) + f_png['LRGxQSO_ell0']/3)*\
                                              ((b0g_png_LRG/Dz_png['LRGxQSO_ell0'])-mod.ppng_LRG))*\
                                            ((mod.Om_m0_png*mod.H0_png**2)/Dz_png['LRGxQSO_ell0']))
        r_fac_c1[mod.term_masks['QSO_ell0']] = (((b0g_QSO/Dz_g['QSO_ell0']) + f_g['QSO_ell0']/3)*\
                                                ((b0g_QSO/Dz_g['QSO_ell0'])-mod.pg_QSO)*((mod.Om_m0_g*mod.H0**2)/Dz_g['QSO_ell0']))/\
                                        (((b0g_png_QSO/Dz_png['QSO_ell0']) + f_png['QSO_ell0']/3)*\
                                         ((b0g_png_QSO/Dz_png['QSO_ell0'])-mod.ppng_QSO)*\
                                             ((mod.Om_m0_png*mod.H0_png**2)/Dz_png['QSO_ell0']))

        r_fac_c2[mod.term_masks['LRG_ell0']] = ((((b0g_LRG/Dz_g['LRG_ell0'])-mod.pg_LRG)*((mod.Om_m0_g*mod.H0**2)/Dz_g['LRG_ell0']))**2)/\
                                                ((((b0g_png_LRG/Dz_png['LRG_ell0'])-mod.ppng_LRG)*\
                                                  ((mod.Om_m0_png*mod.H0_png**2)/Dz_png['LRG_ell0']))**2)
        r_fac_c2[mod.term_masks['LRGxQSO_ell0']] = (((b0g_LRG/Dz_g['LRGxQSO_ell0'])-mod.pg_LRG)*\
                                                    ((b0g_QSO/Dz_g['LRGxQSO_ell0'])-mod.pg_QSO)*\
                                                    ((mod.Om_m0_g*mod.H0**2)/Dz_g['LRGxQSO_ell0'])**2)/\
                                                    (((b0g_png_LRG/Dz_png['LRGxQSO_ell0'])-mod.ppng_LRG)*\
                                                     ((b0g_png_QSO/Dz_png['LRGxQSO_ell0'])-mod.ppng_QSO)*\
                                                     ((mod.Om_m0_png*mod.H0_png**2)/Dz_png['LRGxQSO_ell0'])**2)
        r_fac_c2[mod.term_masks['QSO_ell0']] = ((((b0g_QSO/Dz_g['QSO_ell0'])-mod.pg_QSO)*\
                                                 ((mod.Om_m0_g*mod.H0**2)/Dz_g['QSO_ell0']))**2)/\
                                                ((((b0g_png_QSO/Dz_png['QSO_ell0'])-mod.ppng_QSO)*\
                                                  ((mod.Om_m0_png*mod.H0_png**2)/Dz_png['QSO_ell0']))**2)

        r_fac_c1[mod.term_masks['LRG_ell2']] = (f_g['LRG_ell2']*((b0g_LRG/Dz_g['LRG_ell2'])-mod.pg_LRG)*\
                                                ((mod.Om_m0_g*mod.H0**2)/Dz_g['LRG_ell2']))/\
                                                (f_png['LRG_ell2']*((b0g_png_LRG/Dz_png['LRG_ell2'])-mod.ppng_LRG)*\
                                                 ((mod.Om_m0_png*mod.H0_png**2)/Dz_png['LRG_ell2']))
        r_fac_c1[mod.term_masks['LRGxQSO_ell2']] = (f_g['LRGxQSO_ell2']*(((b0g_LRG/Dz_g['LRGxQSO_ell2'])-mod.pg_LRG)+\
                                                                         ((b0g_QSO/Dz_g['LRGxQSO_ell2'])-mod.pg_QSO))*\
                                                    ((mod.Om_m0_g*mod.H0**2)/Dz_g['LRGxQSO_ell2']))/\
                                                (f_png['LRGxQSO_ell2']*(((b0g_png_LRG/Dz_png['LRGxQSO_ell2'])-mod.ppng_LRG)+\
                                                                        ((b0g_png_QSO/Dz_png['LRGxQSO_ell2'])-mod.ppng_QSO))*\
                                                 ((mod.Om_m0_png*mod.H0_png**2)/Dz_png['LRGxQSO_ell2']))
        r_fac_c1[mod.term_masks['QSO_ell2']] = (f_g['QSO_ell2']*((b0g_QSO/Dz_g['QSO_ell2'])-mod.pg_QSO)*\
                                                ((mod.Om_m0_g*mod.H0**2)/Dz_g['QSO_ell2']))/\
                                                (f_png['QSO_ell2']*((b0g_png_QSO/Dz_png['QSO_ell2'])-mod.ppng_QSO)*\
                                                 ((mod.Om_m0_png*mod.H0_png**2)/Dz_png['QSO_ell2']))
        #################################    
        fid_term = r_fac_fid*(mod.masked['xi_fid'])
        PNG_term = r_fac_c1*mod.masked['c1']*fNL + r_fac_c2*mod.masked['c2']*(fNL**2)
        sys_term = r_fac_fid_png*((mod.masked['A1_LRG']*Psys1_LRG**2+mod.masked['B1_LRG']*Psys1_LRG) +\
                                  (mod.masked['A2_LRG']*Psys2_LRG**2+mod.masked['B2_LRG']*Psys2_LRG) +\
                                  (mod.masked['A3_LRG']*Psys3_LRG**2+mod.masked['B3_LRG']*Psys3_LRG) +\
                                  (mod.masked['A1_QSO']*Psys1_QSO**2+mod.masked['B1_QSO']*Psys1_QSO) +\
                                  (mod.masked['A2_QSO']*Psys2_QSO**2+mod.masked['B2_QSO']*Psys2_QSO) +\
                                  (mod.masked['A3_QSO']*Psys3_QSO**2+mod.masked['B3_QSO']*Psys3_QSO) +\
                                  (mod.masked['A4_QSO']*Psys4_QSO**2+mod.masked['B4_QSO']*Psys4_QSO))
        regression_term = r_fac_fid_png*mod.masked['pvar_par_Kregr']*Kregr
        return fid_term + PNG_term + sys_term + regression_term

class DR2_LRG_QSO:
    parameter_defaults = pd.DataFrame(columns=['key', 'init', 'prior', 'plot_label', 'num_decimals', 'unit'])
    parameter_defaults = parameter_defaults.set_index('key')
    parameter_defaults.loc['fNL'] = [0, [-250, 250,'flat'], r'$f_{NL}$', 0, '']
    parameter_defaults.loc['b0g_LRG'] = [1, [0, 5,'flat'], r'$b_{LRG}$', 2, ''] # made upper bound higher because QSO b w/OQE ~3
    parameter_defaults.loc['b0gfid_LRG'] = [1, [1.94,0.04,'gauss'], r'$b_{LRG}^{fid}$', 2, '']
    parameter_defaults.loc['b0gpng_LRG'] = [1, [1.94,0.04,'gauss'], r'$b_{LRG}^{PNG}$', 2, '']
    parameter_defaults.loc['b0g_QSO'] = [1, [0, 5,'flat'], r'$b_{QSO}$', 2, ''] # made upper bound higher because QSO b w/OQE ~3
    parameter_defaults.loc['b0gfid_QSO'] = [1, [1.94,0.04,'gauss'], r'$b_{QSO}^{fid}$', 2, '']
    parameter_defaults.loc['b0gpng_QSO'] = [1, [1.94,0.04,'gauss'], r'$b_{QSO}^{PNG}$', 2, '']
    parameter_defaults.loc['KsysSGC_LRG'] = [1, [0,10,'gauss'], r'$K_{\mathrm{SGC}^\mathrm{LRG}}$', 1, r'\%']
    parameter_defaults.loc['KsysDEC_LRG'] = [1, [0,10,'gauss'], r'$K_{\mathrm{DEC}^\mathrm{LRG}}$', 1, r'\%']
    parameter_defaults.loc['KsysMZLS_LRG'] = [1, [0,10,'gauss'], r'$K_{\mathrm{MZLS}^\mathrm{LRG}}$', 1, r'\%']
    parameter_defaults.loc['KsysSGC_QSO'] = [1, [0,10,'gauss'], r'$K_{\mathrm{SGC}^\mathrm{QSO}}$', 1, r'\%']
    parameter_defaults.loc['KsysDEC_QSO'] = [1, [0,10,'gauss'], r'$K_{\mathrm{DEC}^\mathrm{QSO}}$', 1, r'\%']
    parameter_defaults.loc['KsysMZLS_QSO'] = [1, [0,10,'gauss'], r'$K_{\mathrm{MZLS}^\mathrm{QSO}}$', 1, r'\%']
    parameter_defaults.loc['KsysDES_QSO'] = [1, [0,10,'gauss'], r'$K_{\mathrm{DES}^\mathrm{QSO}}$', 1, r'\%']
    parameter_defaults.loc['Kregr'] = [0, [0,0.1,'gauss'], r'$K_{\mathrm{regr}}$', 1, r'\%']
    
    extra_parameters = {'z_eff', 'z_fid', 'z_png', 'Om_m0_g', 'Om_m0_fid', 'Om_m0_png', 'H0', 'H0_fid', 'H0_png',
                        'pg_LRG', 'pg_QSO', 'ppng_LRG', 'ppng_QSO'}
    
    @staticmethod
    def xi_modded_base_pars(mod, params):
        fNL, b0g_LRG, b0g_fid_LRG, b0g_png_LRG, b0g_QSO, b0g_fid_QSO, b0g_png_QSO, \
                Psys1_LRG, Psys2_LRG, Psys3_LRG, Psys1_QSO, Psys2_QSO, Psys3_QSO, Psys4_QSO, Kregr = params
        
        ells = [0, 2, 4]        
        f_g = {term: Omega_m_z(mod.z_eff[term],mod.Om_m0_g)**0.55 for term in mod.terms}
        f_fid = {term: Omega_m_z(mod.z_fid[term],mod.Om_m0_fid)**0.55 for term in mod.terms}
        f_png = {term: Omega_m_z(mod.z_png[term],mod.Om_m0_png)**0.55 for term in mod.terms}
        Dz_g = {term: Dz_norm(mod.z_eff[term],Om_m0=mod.Om_m0_g) for term in mod.terms}
        Dz_fid = {term: Dz_norm(mod.z_fid[term],Om_m0=mod.Om_m0_fid) for term in mod.terms}
        Dz_png = {term: Dz_norm(mod.z_png[term],Om_m0=mod.Om_m0_png) for term in mod.terms}
        
        ### Define rescale factors ######
        r_fac_fid = np.ones(mod.N_obs_vec_masked)    
        r_fac_c1 = np.ones(mod.N_obs_vec_masked)
        r_fac_c2 = np.ones(mod.N_obs_vec_masked)
        r_fac_fid_png = np.ones(mod.N_obs_vec_masked)
        
        r_fac_fid[mod.term_masks['LRG_ell0']] = ((b0g_LRG/Dz_g['LRG_ell0'])**2 + \
                                                 (2/3)*(b0g_LRG/Dz_g['LRG_ell0'])*f_g['LRG_ell0'] + (f_g['LRG_ell0']**2)/5)/\
                                                ((b0g_fid_LRG/Dz_fid['LRG_ell0'])**2 + \
                                                 (2/3)*(b0g_fid_LRG/Dz_fid['LRG_ell0'])*f_fid['LRG_ell0'] + (f_fid['LRG_ell0']**2)/5)
        r_fac_fid[mod.term_masks['QSO_ell0']] = ((b0g_QSO/Dz_g['QSO_ell0'])**2 + \
                                                 (2/3)*(b0g_QSO/Dz_g['QSO_ell0'])*f_g['QSO_ell0'] + (f_g['QSO_ell0']**2)/5)/\
                                                ((b0g_fid_QSO/Dz_fid['QSO_ell0'])**2 + \
                                                 (2/3)*(b0g_fid_QSO/Dz_fid['QSO_ell0'])*f_fid['QSO_ell0'] + (f_fid['QSO_ell0']**2)/5)

        r_fac_fid[mod.term_masks['LRG_ell2']] = ( (4/3)*(b0g_LRG/Dz_g['LRG_ell2'])*f_g['LRG_ell2'] + (4/7)*(f_g['LRG_ell2']**2) )/\
                                                ( (4/3)*(b0g_fid_LRG/Dz_fid['LRG_ell2'])*f_fid['LRG_ell2'] + (4/7)*(f_fid['LRG_ell2']**2) )
        r_fac_fid[mod.term_masks['QSO_ell2']] = ( (4/3)*(b0g_QSO/Dz_g['QSO_ell2'])*f_g['QSO_ell2'] + (4/7)*(f_g['QSO_ell2']**2) )/\
                                                ( (4/3)*(b0g_fid_QSO/Dz_fid['QSO_ell2'])*f_fid['QSO_ell2'] + (4/7)*(f_fid['QSO_ell2']**2) )

        r_fac_fid[mod.term_masks['LRG_ell4']] = (f_g['LRG_ell4']/f_fid['LRG_ell4'])**2
        r_fac_fid[mod.term_masks['QSO_ell4']] = (f_g['QSO_ell4']/f_fid['QSO_ell4'])**2

        r_fac_fid_png[mod.term_masks['LRG_ell0']] = ((b0g_LRG/Dz_g['LRG_ell0'])**2 + \
                                                     (2/3)*(b0g_LRG/Dz_g['LRG_ell0'])*f_g['LRG_ell0'] + (f_g['LRG_ell0']**2)/5)/\
                                                ((b0g_png_LRG/Dz_png['LRG_ell0'])**2 + \
                                                 (2/3)*(b0g_png_LRG/Dz_png['LRG_ell0'])*f_png['LRG_ell0'] + (f_png['LRG_ell0']**2)/5)
        r_fac_fid_png[mod.term_masks['QSO_ell0']] = ((b0g_QSO/Dz_g['QSO_ell0'])**2 + \
                                                     (2/3)*(b0g_QSO/Dz_g['QSO_ell0'])*f_g['QSO_ell0'] + (f_g['QSO_ell0']**2)/5)/\
                                                ((b0g_png_QSO/Dz_png['QSO_ell0'])**2 + \
                                                 (2/3)*(b0g_png_QSO/Dz_png['QSO_ell0'])*f_png['QSO_ell0'] + (f_png['QSO_ell0']**2)/5)

        r_fac_fid_png[mod.term_masks['LRG_ell2']] = ( (4/3)*(b0g_LRG/Dz_g['LRG_ell2'])*f_g['LRG_ell2'] + (4/7)*(f_g['LRG_ell2']**2) )/\
                                                ( (4/3)*(b0g_png_LRG/Dz_png['LRG_ell2'])*f_png['LRG_ell2'] + (4/7)*(f_png['LRG_ell2']**2) )
        r_fac_fid_png[mod.term_masks['QSO_ell2']] = ( (4/3)*(b0g_QSO/Dz_g['QSO_ell2'])*f_g['QSO_ell2'] + (4/7)*(f_g['QSO_ell2']**2) )/\
                                                ( (4/3)*(b0g_png_QSO/Dz_png['QSO_ell2'])*f_png['QSO_ell2'] + (4/7)*(f_png['QSO_ell2']**2) )

        r_fac_fid_png[mod.term_masks['LRG_ell4']] = (f_g['LRG_ell4']/f_png['LRG_ell4'])**2
        r_fac_fid_png[mod.term_masks['QSO_ell4']] = (f_g['QSO_ell4']/f_png['QSO_ell4'])**2
        
        r_fac_c1[mod.term_masks['LRG_ell0']] = (((b0g_LRG/Dz_g['LRG_ell0']) + f_g['LRG_ell0']/3)*\
                                                ((b0g_LRG/Dz_g['LRG_ell0'])-mod.pg_LRG)*((mod.Om_m0_g*mod.H0**2)/Dz_g['LRG_ell0']))/\
                                                (((b0g_png_LRG/Dz_png['LRG_ell0']) + f_png['LRG_ell0']/3)*\
                                                 ((b0g_png_LRG/Dz_png['LRG_ell0'])-mod.ppng_LRG)*\
                                                     ((mod.Om_m0_png*mod.H0_png**2)/Dz_png['LRG_ell0']))
        r_fac_c1[mod.term_masks['QSO_ell0']] = (((b0g_QSO/Dz_g['QSO_ell0']) + f_g['QSO_ell0']/3)*\
                                                ((b0g_QSO/Dz_g['QSO_ell0'])-mod.pg_QSO)*((mod.Om_m0_g*mod.H0**2)/Dz_g['QSO_ell0']))/\
                                        (((b0g_png_QSO/Dz_png['QSO_ell0']) + f_png['QSO_ell0']/3)*\
                                         ((b0g_png_QSO/Dz_png['QSO_ell0'])-mod.ppng_QSO)*\
                                             ((mod.Om_m0_png*mod.H0_png**2)/Dz_png['QSO_ell0']))

        r_fac_c2[mod.term_masks['LRG_ell0']] = ((((b0g_LRG/Dz_g['LRG_ell0'])-mod.pg_LRG)*((mod.Om_m0_g*mod.H0**2)/Dz_g['LRG_ell0']))**2)/\
                                                ((((b0g_png_LRG/Dz_png['LRG_ell0'])-mod.ppng_LRG)*\
                                                  ((mod.Om_m0_png*mod.H0_png**2)/Dz_png['LRG_ell0']))**2)
        r_fac_c2[mod.term_masks['QSO_ell0']] = ((((b0g_QSO/Dz_g['QSO_ell0'])-mod.pg_QSO)*\
                                                 ((mod.Om_m0_g*mod.H0**2)/Dz_g['QSO_ell0']))**2)/\
                                                ((((b0g_png_QSO/Dz_png['QSO_ell0'])-mod.ppng_QSO)*\
                                                  ((mod.Om_m0_png*mod.H0_png**2)/Dz_png['QSO_ell0']))**2)

        r_fac_c1[mod.term_masks['LRG_ell2']] = (f_g['LRG_ell2']*((b0g_LRG/Dz_g['LRG_ell2'])-mod.pg_LRG)*\
                                                ((mod.Om_m0_g*mod.H0**2)/Dz_g['LRG_ell2']))/\
                                                (f_png['LRG_ell2']*((b0g_png_LRG/Dz_png['LRG_ell2'])-mod.ppng_LRG)*\
                                                 ((mod.Om_m0_png*mod.H0_png**2)/Dz_png['LRG_ell2']))
        r_fac_c1[mod.term_masks['QSO_ell2']] = (f_g['QSO_ell2']*((b0g_QSO/Dz_g['QSO_ell2'])-mod.pg_QSO)*\
                                                ((mod.Om_m0_g*mod.H0**2)/Dz_g['QSO_ell2']))/\
                                                (f_png['QSO_ell2']*((b0g_png_QSO/Dz_png['QSO_ell2'])-mod.ppng_QSO)*\
                                                 ((mod.Om_m0_png*mod.H0_png**2)/Dz_png['QSO_ell2']))
        #################################    
        fid_term = r_fac_fid*(mod.masked['xi_fid'])
        PNG_term = r_fac_c1*mod.masked['c1']*fNL + r_fac_c2*mod.masked['c2']*(fNL**2)
        sys_term = r_fac_fid_png*((mod.masked['A1_LRG']*Psys1_LRG**2+mod.masked['B1_LRG']*Psys1_LRG) +\
                                  (mod.masked['A2_LRG']*Psys2_LRG**2+mod.masked['B2_LRG']*Psys2_LRG) +\
                                  (mod.masked['A3_LRG']*Psys3_LRG**2+mod.masked['B3_LRG']*Psys3_LRG) +\
                                  (mod.masked['A1_QSO']*Psys1_QSO**2+mod.masked['B1_QSO']*Psys1_QSO) +\
                                  (mod.masked['A2_QSO']*Psys2_QSO**2+mod.masked['B2_QSO']*Psys2_QSO) +\
                                  (mod.masked['A3_QSO']*Psys3_QSO**2+mod.masked['B3_QSO']*Psys3_QSO) +\
                                  (mod.masked['A4_QSO']*Psys4_QSO**2+mod.masked['B4_QSO']*Psys4_QSO))
        regression_term = r_fac_fid_png*mod.masked['pvar_par_Kregr']*Kregr
        return fid_term + PNG_term + sys_term + regression_term

# class DR2_nosys_oqe:
#     parameter_defaults = pd.DataFrame(columns=['key', 'init', 'prior', 'plot_label', 'num_decimals', 'unit'])
#     parameter_defaults = parameter_defaults.set_index('key')
#     parameter_defaults.loc['fNL'] = [0, [-250, 250,'flat'], r'$f_{NL}$', 0, '']
#     parameter_defaults.loc['b0g'] = [1, [0.5, 4,'flat'], r'$b_{0g}$', 2, '']
#     parameter_defaults.loc['b0gfid'] = [1, [1.94,0.04,'gauss'], r'$b_{0g}^{fid}$', 2, '']
#     parameter_defaults.loc['pfid'] = [1, [1,0.1,'gauss'], r'$p_{fid}$', 1,'']
#     parameter_defaults.loc['pg'] = [1, [1,0.1,'gauss'], r'$p_g$', 1,'']

#     extra_parameters = {'z_eff', 'z_fid', 'Om_m0_g', 'Om_m0_fid', 'H0', 'H0_fid'}
#     # z_eff, z_fid should both be dicts
    
#     @staticmethod
#     def xi_modded_base_pars(mod, params):
#         fNL, b0g, b0g_fid, pfid, pg = params
        
#         ells = [0, 2, 4]        
#         f_g = {ell: Omega_m_z(mod.z_eff[ell],mod.Om_m0_g)**0.55 for ell in ells}
#         f_fid = {ell: Omega_m_z(mod.z_fid[ell],mod.Om_m0_fid)**0.55 for ell in ells}
#         Dz_g = {ell: Dz_norm(mod.z_eff[ell],Om_m0=mod.Om_m0_g) for ell in ells}
#         Dz_fid = {ell: Dz_norm(mod.z_fid[ell],Om_m0=mod.Om_m0_fid) for ell in ells}
        
#         ### Define rescale factors ######
#         r_fac_fid = np.ones(mod.N_obs_vec_masked)
#         r_fac_c1 = np.ones(mod.N_obs_vec_masked)
#         r_fac_c2 = np.ones(mod.N_obs_vec_masked)
        
#         r_fac_fid[mod.term_masks['xi0']] = ((b0g/Dz_g[0])**2 + (2/3)*(b0g/Dz_g[0])*f_g[0] + (f_g[0]**2)/5)/\
#                                                 ((b0g_fid/Dz_fid[0])**2 + (2/3)*(b0g_fid/Dz_fid[0])*f_fid[0] + (f_fid[0]**2)/5)
#         r_fac_fid[mod.term_masks['xi2']] = ( (4/3)*(b0g/Dz_g[2])*f_g[2] + (4/7)*(f_g[2]**2) )/\
#                                                 ( (4/3)*(b0g_fid/Dz_fid[2])*f_fid[2] + (4/7)*(f_fid[2]**2) )
#         r_fac_fid[mod.term_masks['xi4']] = (f_g[4]/f_fid[4])**2
    
#         r_fac_c1[mod.term_masks['xi0']] = (((b0g/Dz_g[0]) + f_g[0]/3)*((b0g/Dz_g[0])-pg)*((mod.Om_m0_g*mod.H0**2)/Dz_g[0]))/\
#                                                 (((b0g_fid/Dz_fid[0]) + f_fid[0]/3)*((b0g_fid/Dz_fid[0])-pfid)*\
#                                                      ((mod.Om_m0_fid*mod.H0_fid**2)/Dz_fid[0]))
#         r_fac_c2[mod.term_masks['xi0']] = ((((b0g/Dz_g[0])-pg)*((mod.Om_m0_g*mod.H0**2)/Dz_g[0]))**2)/\
#                                                 ((((b0g_fid/Dz_fid[0])-pfid)*((mod.Om_m0_fid*mod.H0_fid**2)/Dz_fid[0]))**2)
#         r_fac_c1[mod.term_masks['xi2']] = (f_g[2]*((b0g/Dz_g[2])-pg)*((mod.Om_m0_g*mod.H0**2)/Dz_g[2]))/\
#                                                 (f_fid[2]*((b0g_fid/Dz_fid[2])-pfid)*((mod.Om_m0_fid*mod.H0_fid**2)/Dz_fid[2]))
#         #################################    
#         fid_term = r_fac_fid*(mod.masked['xi_fid'])
#         PNG_term = r_fac_c1*mod.masked['c1']*fNL + r_fac_c2*mod.masked['c2']*(fNL**2)
#         return fid_term + PNG_term 

class DR2_nosys_oqe_ab:
    parameter_defaults = pd.DataFrame(columns=['key', 'init', 'prior', 'plot_label', 'num_decimals', 'unit'])
    parameter_defaults = parameter_defaults.set_index('key')
    parameter_defaults.loc['fNL'] = [0, [-250, 250,'flat'], r'$f_{NL}$', 0, '']
    parameter_defaults.loc['b0g'] = [1, [0.5, 5,'flat'], r'$b_{0g}$', 2, ''] # made upper bound higher because QSO b w/OQE ~3
    parameter_defaults.loc['b0gfid'] = [1, [1.94,0.04,'gauss'], r'$b_{0g}^{fid}$', 2, '']
    parameter_defaults.loc['b0gpng'] = [1, [1.94,0.04,'gauss'], r'$b_{0g}^{PNG}$', 2, '']
    # parameter_defaults.loc['pg'] = [1, [1,0.1,'gauss'], r'$p_g$', 1,'']
    # parameter_defaults.loc['ppng'] = [1, [1,0.1,'gauss'], r'$p_g$', 1,'']
    
    # extra_parameters = {'z_eff', 'z_fid', 'z_png', 'Om_m0_g', 'Om_m0_fid', 'Om_m0_png', 'H0', 'H0_fid', 'H0_png'}
    extra_parameters = {'z_eff', 'z_fid', 'z_png', 'Om_m0_g', 'Om_m0_fid', 'Om_m0_png', 'H0', 'H0_fid', 'H0_png', 'pg', 'ppng'}
    
    # z_eff, z_fid should both be dicts
    
    @staticmethod
    def xi_modded_base_pars(mod, params):
        # fNL, b0g, b0g_fid, b0g_png, pg, ppng = params
        fNL, b0g, b0g_fid, b0g_png = params
        
        
        ells = [0, 2, 4]        
        f_g = {ell: Omega_m_z(mod.z_eff[ell],mod.Om_m0_g)**0.55 for ell in ells}
        f_fid = {ell: Omega_m_z(mod.z_fid[ell],mod.Om_m0_fid)**0.55 for ell in ells}
        f_png = {ell: Omega_m_z(mod.z_png[ell],mod.Om_m0_png)**0.55 for ell in ells}
        Dz_g = {ell: Dz_norm(mod.z_eff[ell],Om_m0=mod.Om_m0_g) for ell in ells}
        Dz_fid = {ell: Dz_norm(mod.z_fid[ell],Om_m0=mod.Om_m0_fid) for ell in ells}
        Dz_png = {ell: Dz_norm(mod.z_png[ell],Om_m0=mod.Om_m0_png) for ell in ells}
        
        ### Define rescale factors ######
        r_fac_fid = np.ones(mod.N_obs_vec_masked)
        r_fac_c1 = np.ones(mod.N_obs_vec_masked)
        r_fac_c2 = np.ones(mod.N_obs_vec_masked)
        
        r_fac_fid[mod.term_masks['xi0']] = ((b0g/Dz_g[0])**2 + (2/3)*(b0g/Dz_g[0])*f_g[0] + (f_g[0]**2)/5)/\
                                                ((b0g_fid/Dz_fid[0])**2 + (2/3)*(b0g_fid/Dz_fid[0])*f_fid[0] + (f_fid[0]**2)/5)
        r_fac_fid[mod.term_masks['xi2']] = ( (4/3)*(b0g/Dz_g[2])*f_g[2] + (4/7)*(f_g[2]**2) )/\
                                                ( (4/3)*(b0g_fid/Dz_fid[2])*f_fid[2] + (4/7)*(f_fid[2]**2) )
        r_fac_fid[mod.term_masks['xi4']] = (f_g[4]/f_fid[4])**2
    
        r_fac_c1[mod.term_masks['xi0']] = (((b0g/Dz_g[0]) + f_g[0]/3)*((b0g/Dz_g[0])-mod.pg)*((mod.Om_m0_g*mod.H0**2)/Dz_g[0]))/\
                                                (((b0g_png/Dz_png[0]) + f_png[0]/3)*((b0g_png/Dz_png[0])-mod.ppng)*\
                                                     ((mod.Om_m0_png*mod.H0_png**2)/Dz_png[0]))
        r_fac_c2[mod.term_masks['xi0']] = ((((b0g/Dz_g[0])-mod.pg)*((mod.Om_m0_g*mod.H0**2)/Dz_g[0]))**2)/\
                                                ((((b0g_png/Dz_png[0])-mod.ppng)*((mod.Om_m0_png*mod.H0_png**2)/Dz_png[0]))**2)
        r_fac_c1[mod.term_masks['xi2']] = (f_g[2]*((b0g/Dz_g[2])-mod.pg)*((mod.Om_m0_g*mod.H0**2)/Dz_g[2]))/\
                                                (f_png[2]*((b0g_png/Dz_png[2])-mod.ppng)*((mod.Om_m0_png*mod.H0_png**2)/Dz_png[2]))
        #################################    
        fid_term = r_fac_fid*(mod.masked['xi_fid'])
        PNG_term = r_fac_c1*mod.masked['c1']*fNL + r_fac_c2*mod.masked['c2']*(fNL**2)
        return fid_term + PNG_term 

class fNL_only:
    parameter_defaults = pd.DataFrame(columns=['key', 'init', 'prior', 'plot_label', 'num_decimals', 'unit'])
    parameter_defaults = parameter_defaults.set_index('key')
    parameter_defaults.loc['fNL'] = [0, [-250, 250,'flat'], r'$f_{NL}$', 0, '']

    extra_parameters = set()

    @staticmethod
    def xi_modded_base_pars(mod, params):
        fNL = params
        
        fid_term = mod.masked['xi_fid']
        PNG_term = mod.masked['c1']*fNL + mod.masked['c2']*(fNL**2)
        return fid_term + PNG_term 