import numpy as np
import pandas as pd
from codes.helper_functions import *

class MathModel:
    def xi_modded_base_pars(self, mod, params):
        raise NotImplementedError
        
    def util_chi2_base_pars(self, mod, params):
        raise NotImplementedError

    def log_prior_base_pars(self, mod, params):
        raise NotImplementedError

    def log_probability_base_pars(self, mod, params):
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
        r_fac_fid = np.ones(len(mod.xi_fid))
        r_fac_c1 = np.ones(len(mod.xi_fid))
        r_fac_c2 = np.ones(len(mod.xi_fid))
        
        r_fac_fid[mod.xi0_cond] = (b1g**2 + (2/3)*b1g*f_g + (f_g**2)/5)/(b1g_fid**2 + (2/3)*b1g_fid*f_fid + (f_fid**2)/5)
        r_fac_fid[mod.xi2_cond] = ( (4/3)*b1g*f_g + (4/7)*(f_g**2) )/( (4/3)*b1g_fid*f_fid + (4/7)*(f_fid**2) )
        r_fac_fid[mod.xi4_cond] = (f_g/f_fid)**2
    
        r_fac_c1[mod.xi0_cond] = ((b1g + f_g/3)*(b1g-pg)*(mod.Om_m0_g/Dz_g))/\
                                ((b1h + f_h/3)*(b1h-ph)*(mod.Om_m0_h/Dz_h))
        r_fac_c2[mod.xi0_cond] = (((b1g-pg)**2)*(mod.Om_m0_g/Dz_g))/(((b1h-ph)**2)*(mod.Om_m0_h/Dz_h))
        r_fac_c1[mod.xi2_cond] = (f_g*(b1g-pg)*(mod.Om_m0_g/Dz_g))/(f_h*(b1h-ph)*(mod.Om_m0_h/Dz_h))
        #################################    
        fid_term = r_fac_fid*(mod.xi_fid)
        PNG_term = r_fac_c1*mod.c1*fNL + r_fac_c2*mod.c2*(fNL**2)
        sys_term = r_fac_fid*((mod.pvar_par_A1*Psys1**2+mod.pvar_par_B1*Psys1) +\
                              (mod.pvar_par_A2*Psys2**2+mod.pvar_par_B2*Psys2) + (mod.pvar_par_A3*Psys3**2+mod.pvar_par_B3*Psys3))
        return fid_term + PNG_term + sys_term

    @staticmethod
    def util_chi2_base_pars(mod, params):
        # Defines chi2 given data and params
        exp = mod.xi_modded_base_pars(params)
        cov_inv = np.linalg.inv(mod.cov_mat)
        return -0.5*np.matmul(np.matmul(cov_inv,(mod.obs-exp)),(mod.obs-exp))

    @staticmethod
    def log_prior_base_pars(mod, params):
        fNL, b1g, b1h, b1g_fid, ph, pg, Psys1, Psys2, Psys3 = params
        if mod.poi_hard_lims[0][0] < fNL < mod.poi_hard_lims[0][1] and \
            mod.poi_hard_lims[1][0] < b1g < mod.poi_hard_lims[1][1]:
            return -(Psys1-mod.Psys1_gauss_prior[0])**2/(mod.Psys1_gauss_prior[1])**2-\
            (Psys2-mod.Psys2_gauss_prior[0])**2/(mod.Psys2_gauss_prior[1])**2-\
            (Psys3-mod.Psys3_gauss_prior[0])**2/(mod.Psys3_gauss_prior[1])**2-\
            (b1h-mod.gauss_priors[0][0])**2/(mod.gauss_priors[0][1])**2-\
            (b1g_fid-mod.gauss_priors[1][0])**2/(mod.gauss_priors[1][1])**2-\
            (ph-mod.gauss_priors[2][0])**2/(mod.gauss_priors[2][1])**2-\
            (pg-mod.gauss_priors[3][0])**2/(mod.gauss_priors[3][1])**2
        return -np.inf

    @staticmethod
    def log_probability_base_pars(mod, params):
        # Defines the log probability combining the likelihood and priors
        lp = 0.5*mod.log_prior_base_pars(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + mod.util_chi2_base_pars(params)

class DR2_nosys:
    parameter_defaults = pd.DataFrame(columns=['key', 'init', 'prior', 'plot_label', 'num_decimals', 'unit'])
    parameter_defaults = parameter_defaults.set_index('key')
    parameter_defaults.loc['fNL'] = [0, [-250, 250,'flat'], r'$f_{NL}$', 0, '']
    parameter_defaults.loc['b1g'] = [1, [0.5, 4,'flat'], r'$b_{1g}$', 2, '']
    parameter_defaults.loc['b1gfid'] = [1, [1.94,0.04,'gauss'], r'$b_{1g}^{fid}$', 2, '']
    parameter_defaults.loc['pfid'] = [1, [1,0.1,'gauss'], r'$p_{fid}$', 1,'']
    parameter_defaults.loc['pg'] = [1, [1,0.1,'gauss'], r'$p_g$', 1,'']
    
    def xi_modded_base_pars(self, mod, params):
        fNL, b1g, b1g_fid, pfid, pg = params
        f_g = Omega_m_z(mod.z_eff,mod.Om_m0_g)**0.55
        f_fid = Omega_m_z(mod.z_fid,mod.Om_m0_fid)**0.55
        Dz_g = Dz_norm(mod.z_eff,Om_m0=mod.Om_m0_g)
        Dz_fid = Dz_norm(mod.z_fid,Om_m0=mod.Om_m0_fid)
        
        ### Define rescale factors ######
        r_fac_fid = np.ones(len(mod.xi_fid))
        r_fac_c1 = np.ones(len(mod.xi_fid))
        r_fac_c2 = np.ones(len(mod.xi_fid))
        
        r_fac_fid[mod.xi0_cond] = (b1g**2 + (2/3)*b1g*f_g + (f_g**2)/5)/(b1g_fid**2 + (2/3)*b1g_fid*f_fid + (f_fid**2)/5)
        r_fac_fid[mod.xi2_cond] = ( (4/3)*b1g*f_g + (4/7)*(f_g**2) )/( (4/3)*b1g_fid*f_fid + (4/7)*(f_fid**2) )
        r_fac_fid[mod.xi4_cond] = (f_g/f_fid)**2
    
        r_fac_c1[mod.xi0_cond] = ((b1g + f_g/3)*(b1g-pg)*(mod.Om_m0_g/Dz_g))/\
                                ((b1g_fid + f_fid/3)*(b1g_fid-pfid)*(mod.Om_m0_fid/Dz_fid))
        r_fac_c2[mod.xi0_cond] = (((b1g-pg)**2)*(mod.Om_m0_g/Dz_g))/(((b1g_fid-pfid)**2)*(mod.Om_m0_fid/Dz_fid))
        r_fac_c1[mod.xi2_cond] = (f_g*(b1g-pg)*(mod.Om_m0_g/Dz_g))/(f_fid*(b1g_fid-pfid)*(mod.Om_m0_fid/Dz_fid))
        #################################    
        fid_term = r_fac_fid*(mod.xi_fid)
        PNG_term = r_fac_c1*mod.c1*fNL + r_fac_c2*mod.c2*(fNL**2)
        return fid_term + PNG_term 
    
    def util_chi2_base_pars(self, mod, params):
        ...
        # Think about how to pull this from Y1 since its functionally the same
        ...
        exp = mod.xi_modded_base_pars(params)
        cov_inv = np.linalg.inv(mod.cov_mat)
        return -0.5*np.matmul(np.matmul(cov_inv,(mod.obs-exp)),(mod.obs-exp))
    
    def log_prior_base_pars(self, mod, params):
        fNL, b1g, b1g_fid, pfid, pg = params
        if mod.poi_hard_lims[0][0] < fNL < mod.poi_hard_lims[0][1] and \
            mod.poi_hard_lims[1][0] < b1g < mod.poi_hard_lims[1][1]:
            return -(b1g_fid-mod.gauss_priors[0][0])**2/(mod.gauss_priors[0][1])**2-\
            (pfid-mod.gauss_priors[1][0])**2/(mod.gauss_priors[1][1])**2-\
            (pg-mod.gauss_priors[2][0])**2/(mod.gauss_priors[2][1])**2
        return -np.inf
    
    def log_probability_base_pars(self, mod, params):
        # Defines the log probability combining the likelihood and priors
        lp = 0.5*mod.log_prior_base_pars(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + mod.util_chi2_base_pars(params)