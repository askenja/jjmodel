"""
Created on Thu Feb  4 12:14:54 2016
@author: Skevja

This file contains definitions of the JJ model input functions: 
    - Age-metllicity relation (AMR)
    - Age-velocity dispersion relation (AVR)
    - Star formation rate (SFR)
    - Initial mass function (IMF)
    - Radial density profiles of the different Galactic components. 
"""

import os
import numpy as np
import scipy.signal as sgn
from scipy.special import erf, kv, iv
from scipy.optimize import curve_fit
from scipy.integrate import dblquad, quad
from .iof import tab_reader, dir_tree
from .tools import transition_2curves, ConvertAxes
from .constants import tp, tr, GA, G, GYR, PC, SIGMA_E, KM, M_SUN
from . import localpath


def hgr(p,a):
    """
    Schale heights of the atomic and molecular gas components as 
    a function of Galactocentric distance R. Taken from Nakanishi 
    and Sofue (2016). 
    
    Parameters
    ----------
    p : namedtuple
        Set of the model parameters from the parameter file. 
    a : namedtuple
        Collection of the fixed model parameters, useful quantities 
        and arrays.

    Returns
    -------
    (hg1,hg10) : ndarray, float
        Molecular gas scale height (in pc) at Galactocentric 
        distances a.R. hg10 is the scale height at the Solar radius. 
    (hg2,hg20) : ndarray, float
        Atomic gas scale height (in pc) at Galactocentric distances 
        a.R. hg10 is the scale height at the Solar radius. 
    """
    
    T = dir_tree(p)
    H2,HI = tab_reader(['H2','HI'],p,T)
    
    exp_law = lambda x,a,b: a*np.exp(x*b)
    popt1,pcov1 = curve_fit(exp_law,H2[0],H2[1]/2)
    popt2,pcov2 = curve_fit(exp_law,HI[0],HI[1]/2)
    hg10, hg20 = np.round(exp_law(p.Rsun,*popt1),1), np.round(exp_law(p.Rsun,*popt2),1)
    
    if p.run_mode==1:
        hg1, hg2 = np.round(exp_law(a.R,*popt1),1), np.round(exp_law(a.R,*popt2),1)
        return (hg1,hg10),(hg2,hg20)
    else:
        return (hg10, hg20)



class RadialPotential():
    
    def __init__(self,Rsun,R):
        """
        Class instance is initialized by the two parameters:
        
        Rsun : float
            Solar Galactocentric distance, kpc.
        R : float or array_like
            Galactocentric distance(s), where the matter density 
            has to be calculated, kpc.  
        """
        self.R, self.Rsun = R, Rsun
        
    def exp_disk(self,SigmaR0,Rd):
    
        y = self.R/2/Rd
        Fi = -np.pi*G*SigmaR0*M_SUN/PC**2*self.R*1e3*PC*\
            (iv(0,y)*kv(1,y) - iv(1,y)*kv(0,y)) 
        return Fi
    
    def pow_law(self,rho0,alpha):
    
        Fi = 4*np.pi*G*rho0*M_SUN/PC**3/(3-alpha)/(2-alpha)*\
            (self.Rsun/self.R)**alpha*(self.R*1e3*PC)**2 
        return Fi
        
    def cored_iso_sphere(self,rho0,ah):
    
        C = (ah**2 + self.Rsun**2)/ah**2
        Fi = 4*np.pi*G*rho0*C*ah**2*(ah/self.R*np.arctan(self.R/ah) +\
                                     0.5*np.log(1 + (self.R/ah)**2))*M_SUN/PC
        return Fi



class RadialDensity():
    """
    Class for defining radial density profiles of the different 
    Galactic components. 
    """
    
    def __init__(self,Rsun,zsun,R):
        """
        Class instance is initialized by the two parameters:
        
        Rsun : float
            Solar Galactocentric distance, kpc.
        zsun : float
            Height of Sun above the plane, pc.
        R : float or array_like
            Galactocentric distance(s), where the matter density 
            has to be calculated, kpc.  
        """
        self.Rsun = Rsun
        self.zsun = zsun*1e-3
        self.rsun = np.sqrt(Rsun**2 + (zsun*1e-3)**2)
        self.R = R
        
    def rho_disk(self,rho0,Rd):
        """
        Mass density of an exponential disk in the Galactic plane.
    
        Parameters
        ----------
        rho0 : scalar
            Local mass density, Msun/pc^3.       
        Rd : scalar
            Radial scale length of the disk, kpc. 
            
        Returns
        -------
        rho : float
            Mass density of the disk at Galactocentric distance R, 
            Msun/pc^3. 
        """
        rho = rho0*np.exp(-(np.subtract(self.R,self.Rsun))/Rd)
        return rho
    
    
    def sigma_disk(self,sigma0,Rd):
        """
        Surface density of an exponential disk.
    
        Parameters
        ----------
        sigma0 : scalar
            Local surface density, Msun/pc^2.       
        Rd : scalar
            Radial scale length of the disk, kpc. 
            
        Returns
        -------
        sigma : float
            Surface density of the disk at Galactocentric distance 
            R, Msun/pc^2. 
        """
        
        sigma = sigma0*np.exp(-(np.subtract(self.R,self.Rsun))/Rd)
        return sigma
        
    
    def rho_dm_halo(self,z,rho0,ah):
        """
        3D mass density of an isothermal DM sphere.
    
        Parameters
        ----------
        z : scalar
            Height above the plane, kpc. 
        rho0 : scalar
            Local mass density of the DM halo, Msun/pc^3.
        ah : scalar
            DM scaling parameter, kpc. 
            
        Returns
        -------
        rho : float
            DM halo mass density at Galactocentric 
            distance R and height z, in Msun/pc^3.   
        """
        
        '''
        #rho = rho0*(np.divide(self.Rsun,self.R))**2
        C = (ah**2 + self.Rsun**2)/ah**2
        rho = rho0*C*ah**2/(ah**2 + self.R**2)   
        '''
        
        r = np.sqrt(self.R**2 + z**2)
        
        rhoc = rho0*(ah**2 + self.rsun**2)/ah**2
        rho = rhoc*ah**2/(ah**2 + r**2)   
    
        return rho
    
    
    def sigma_dm_halo(self,zmax,sigma0,ah):
        """
        Surface density of an isothermal DM sphere.
    
        Parameters
        ----------
        zmax : scalar
            Max height above the plane, kpc. 
        sigma0 : scalar
            Local surface density, Msun/pc^2.
        ah : scalar
            DM scaling parameter, kpc. 
            
        Returns
        -------
        sigma : float
            DM halo surface density at Galactocentric 
            distance R (up to height zmax), in Msun/pc^2.   
        """
        
        tan_term = np.arctan(zmax/np.sqrt(ah**2 + self.R**2))/np.arctan(zmax/np.sqrt(ah**2 + self.Rsun**2))
        
        sigma = sigma0*np.sqrt(ah**2 + self.Rsun**2)/np.sqrt(ah**2 + self.R**2)*tan_term    
    
        return sigma

    
    def rho_stellar_halo(self,z,rho0,a_sh):
        """
        3D mass density of a spherical stellar halo.
        Flattening ignored, profile is a power law. 
        
        Parameters
        ----------
        z : scalar
            Height above the plane, kpc. 
        rho0 : scalar
            Local mass density of the halo, Msun/pc^3. 
        a_sh : scalar
            Slope of the halo profile. Usually about -2.5 (see 
            Bland-Hawthorn 2016). 
            
        Returns
        -------
        rho : float
            Midplane halo mass density at Galactocentric distance 
            R, Msun/pc^3.  
        """
        
        '''
        rho = rho0*(np.divide(self.Rsun,self.R))**(-a_sh)
        '''
        
        r = np.sqrt(self.R**2 + z**2)
        rho = rho0*(self.rsun/r)**(-a_sh)
        
        return rho
    
    
    def sigma_stellar_halo(self,zmax,sigma0,a_sh):
        """
        Surface density of a spherical stellar halo.
        Flattening ignored, profile is a power law. 
        
        Parameters
        ----------
        zmax : scalar
            Max height above the plane, kpc. 
        sigma0 : scalar
            Local surface density, Msun/pc^2. 
        a_sh : scalar
            Slope of the halo profile. Usually about -2.5 (see 
            Bland-Hawthorn 2016). 
            
        Returns
        -------
        sigma : float
            Halo surface density at Galactocentric distance 
            R up to zmax, Msun/pc^2.  
        """
        
        rho0 = sigma0/2/quad(lambda z: (self.Rsun/np.sqrt(self.Rsun**2 + z**2))**(-a_sh),0,zmax)[0]
        
        Rbins = len(self.R)
        sigma = np.zeros((Rbins))
        
        for i in range(Rbins):
            sigma[i] = 2*quad(lambda z: rho0*\
                              (np.divide(self.Rsun,np.sqrt(self.R[i]**2 + z**2)))**(-a_sh),0,zmax)[0]
        
        return sigma

    

class RotCurve():
    """
    Circular velocity as a function of R as follows from
    the assumed density laws for the model components.
    All methods of this class return velocity values for a set 
    of Galactocentric distnces, in km/s. 
    """
    
    def __init__(self,Rsun,R):
        """
        Class instance is initialized by the two parameters:
        
        Rsun : float
            Solar Galactocentric distance, kpc.
        R : float or array_like
            Galactocentric distance(s), where the matter density 
            has to be calculated, kpc.  
        """
        self.Rsun = Rsun
        self.R = R
        self.m_r = M_SUN/PC/1e3
        self.rho_r2 = M_SUN/PC*1e6
        self.sigma_r = M_SUN/PC*1e3
        
    def vc_bulge(self,Mb):
        """
        Rotation velocity for a point-mass bulge. 
        
        Parameters
        ----------
        Mb : scalar
            Mass of bulge.
        """
        
        vc = np.sqrt(G*Mb/self.R*self.m_r)/1e3
        return vc

    def vc_disk(self,sigma0,Rd,R0):
        """
        Rotation velocity for an infinitely thin exponential disk 
        (Eq. 2-169 in Binney & Tremaine).
        
        Parameters
        ----------
        sigma0 : scalar
            Surface density at Rsun, Msun/pc^2.
        Rd : scalar
            Radial scale length, kpc.
        R0 : scalar
            Radius of the inner hole, kpc. 
        """
        
        sigma = sigma0*np.exp((self.Rsun - R0)/Rd)
        R = self.R - R0
        vc = np.sqrt(np.pi*G*sigma*R**2/Rd*\
                     (iv(0,R/2/Rd)*kv(0,R/2/Rd)-iv(1,R/2/Rd)*kv(1,R/2/Rd))*self.sigma_r)/1e3
        vc[vc*0!=0]=0
        return vc

    def vc_halo_nfw(self,rho0,ah):
        """
        Rotation velocity for DM halo with NWF profile.
        
        Parameters
        ----------
        rho0 : scalar
            DM density at Rsun, Msun/pc^3.
        ah : scalar
            Scaling parameter, kpc.
        """
        
        C = (self.Rsun/ah)*(1 + self.Rsun/ah)**2
        #vc = np.sqrt(4*np.pi*G*rho0*C*ah**3/self.R*\
        #             (np.log(1+self.R/ah)-self.R/ah/(1+self.R/ah))*self.rho_r2)/1e3
        vc = np.sqrt(4*np.pi*G*rho0*C*ah**2*\
                     (np.log(1+self.R/ah)/self.R*ah - 1/(1 + self.R/ah))*self.rho_r2)/1e3
        return vc

    def vc_halo_cored_iso_sphere(self,rho0,ah):
        """
        Rotation velocity for DM halo, which is a cored isothermal 
        sphere.
        
        Parameters
        ----------
        rho0 : scalar
            DM density at Rsun, Msun/pc^3.
        ah : scalarM_SUN/(1e3*PC)
            Scaling parameter, kpc.
        """
        
        C = (ah**2 + self.Rsun**2)/ah**2
        vc = np.sqrt(4*np.pi*G*rho0*C*ah**2*(1 - ah/self.R*np.arctan(self.R/ah))*self.rho_r2)/1e3
        
        return vc
    
    def _vc_halo_power_law_(self,rho0,alpha):
        """
        Rotation velocity for stellar halo with a power-law profile.
        Not physical, gives too large enclosed halo mass. Must be 
        truncated at some R near GC, or  core has to be added. 
        
        Parameters
        ----------
        rho0 : scalar
            Halo density at Rsun, Msun/pc^3.
        alpha : scalar
            Power-law slope (rho ~ 1/R^alpha, make sure that 
            you give alpha > 0).
        """
    
        vc = np.sqrt(4*np.pi*G*rho0*self.Rsun**alpha/(3 - alpha)*self.R**(2 - alpha)*self.rho_r2)/1e3
        return vc
            
    
    def vc_tot(self,vc_array):
        """
        Total rotation velocity, quadratic sum of all velocity 
        components. Contribution of stellar halo is ignored. 
        
        Parameters
        ----------
        vc_array : array-like
            List of lists, list of arrays or 2d-array. Here lists,
            arrays or array rows contain Vc(R) of the different
            components. Output units are the same as for the 
            input velocities. 
        """
        
        vc_array = np.array(vc_array)
        if len(vc_array.shape)==1:
            vc = np.sqrt(np.sum(vc_array**2))
        else:
            N = vc_array.shape[1]
            vc = np.array([np.sqrt(np.sum(vc_array[:,i]**2)) for i in range(N)])
        return vc
    
    def vc0(self,vc_tot):
        """
        Calculates tocal Vc.  
        
        Parameters
        ----------
        vc_tot : array-like
            Array of Vc corresponding to R grid.  
        """
        ind_r0 = np.where(np.abs(self.R-self.Rsun)==np.amin(np.abs(self.R-self.Rsun)))[0]
        vc0 = vc_tot[ind_r0]
        return vc0



def heffr(p,a,heffd0):
    """
    Thin-disk half-thickness as a function of the Galactocentric 
    distance R. 
    
    Parameters
    ----------
    p : namedtuple
        Set of the model parameters from the parameter file. 
    a : namedtuple
        Collection of the fixed model parameters, useful quantities 
        and arrays.
    heffd0 : scalar
        Thin-disk half-thickness at the Solar radius.

    Returns
    -------
    hdeff : ndarray
        Thin-disk half-thickness (pc) at Galactocentric distances 
        a.R.
    """
    
    if p.Rf < p.Rmax:
        indR = np.where(np.abs(a.R-p.Rf)==np.amin(np.abs(a.R-p.Rf)))[0][0]
        if p.Rf < p.Rsun:
            hdeff_rf = heffd0*np.exp((p.Rf-p.Rsun)/p.Rdf)
        else:
            hdeff_rf = heffd0
        hdeff_r1 = [hdeff_rf for i in a.R[:indR]]
        hdeff_r2 = [hdeff_rf*np.exp((i-p.Rf)/p.Rdf) for i in a.R[indR:]] 
        hdeff = np.concatenate((hdeff_r1,hdeff_r2),axis=0)
        epsilon_R = 0.5     # kpc
        hdeff = transition_2curves(epsilon_R,p.Rf,a.R,hdeff)
    else:
        hdeff = [heffd0 for i in a.R]
    
    return hdeff


class AMR():
    """
    Collection of functions, which are used 
    to model metallicities and an age-metallicity relation.  
    """
        
    def amrd_jj10(self,t,tp,q,r,FeH_0,FeH_p):
        """
        Thin-disk age-metallicity relation (AMR) equation as 
        defined in Just and Jahreiss (2010).
        
        Parameters
        ----------
        t : scalar or array_like
            Galactic time, Gyr.  
        tp : scalar
            Present-day age of the MW disk (i.e., present-day 
            Galactic time, Gyr). 
        q,r : scalar
            Power indices in the AMR equation.
        FeH_0,FeH_p : scalar
            Initial and present-day metallicities of the MW disk 
            in the Solar neighbourhood. 
           
        Returns
        -------
        FeH : float or ndarray
            Metallicity of the local MW-disk populations, which 
            were born at Galactic time t. 
        """
        
        a = 2.67
        OH_0, OH_p = FeH_0/a, FeH_p/a
        ZO_0, ZO_p=10**OH_0,10**OH_p
        ZO = ZO_0 + (ZO_p-ZO_0)*np.log(1+q*(np.divide(t,tp)**r))/np.log(1+q)
        FeH = np.log10(ZO)*a
        
        return FeH
    

    def amrd_jj10_default(self,t,**kwargs):
        """
        Thin-disk age-metallicity relation (AMR) equation as 
        defined in Just and Jahreiss (2010) with the best 
        parameters (model A). 
        
        Parameters
        ----------
        t : scalar or array_like
            Galactic time, Gyr.
        **kwargs : dict, optional keyword arguments
            stretch: boolean
                If True, uses an extended relative to model A 
                time scale by setting a present-day age of the MW 
                disk to tp = 13 Gyr. By default is False, tp = 
                12 Gyr, as in Just and Jahreiss (2010). 
                 
        Returns
        -------
        FeH : float or ndarray
            Metallicity of the local MW-disk populations, which 
            were born at Galactic time t. 
        """
        
        q, r = 2.0, 0.55
        FeH_0, FeH_p = -0.6, 0.02
        
        if 'stretch' in kwargs and kwargs['stretch']==True:
            tp_ = tp 
        else: 
            tp_ = 12 

        return self.amrd_jj10(t,tp_,q,r,FeH_0,FeH_p)
    
    
    def amrd_sj21(self,t,tp,q,r,FeH_0,FeH_p):
        """
        Thin-disk age-metallicity relation (AMR) equation as 
        defined in Sysoliatina and Just (2021).
        
        Parameters
        ----------
        t : scalar or array_like
            Galactic time, Gyr.
        tp : scalar
            Present-day age of the MW disk (i.e., present-day 
            Galactic time, Gyr).
        q, r : scalar
            Power indices in the AMR equation. 
        FeH_0,FeH_p : scalar
            Initial and present-day metallicities of the MW disk 
            in the Solar neighbourhood. 
           
        Returns
        -------
        FeH : float or ndarray
            Metallicity of the local MW-disk populations, which 
            were born at Galactic time t. 
        """
        
        FeH = FeH_0 + (FeH_p - FeH_0)*np.log(1+q*(np.divide(t,tp)**r))/np.log(1+q)
        return FeH
    
    
    def amrd_sj21_default(self,t):
        """
        Thin-disk age-metallicity relation (AMR) equation as defined 
        in Sysoliatina and Just (2021), with the best parameters 
        (Table 3).
        
        Parameters
        ----------
        t : scalar or array_like
            Galactic time, Gyr. 
           
        Returns
        -------
        FeH : float or ndarray
            Metallicity of the local MW-disk populations, which 
            were born at Galactic time t. 
        """
        
        tp_ = tp
        q, r = -0.72, 0.34
        FeH_0, FeH_p = -0.7, 0.29
        return self.amrd_sj21(t,tp_,q,r,FeH_0,FeH_p)
    
    
    def amrd_global_sj22(self,t,t01,t02,r1,r2,alpha_w,FeH_0,FeH_p):
        """
        Thin-disk age-metallicity relation (AMR) equation as 
        defined in Sysoliatina and Just (2022). All parameters 
        must correspond to the same R (not necessarily to Rsun). 
        
        Parameters
        ----------
        t : scalar or array_like
            Galactic time, Gyr.
        t01, t02 : scalar
            Time-scale parameters, Gyr. 
        r1, r2 : scalar
            Power indices of tanh-terms. 
        FeH_0, FeH_p : scalar
            Initial and present-day metallicity. 
           
        Returns
        -------
        FeH : float or ndarray
            Metallicity of the MW-disk populations born 
            at Galactic time t. 
        """
        
        f_t = (alpha_w*np.tanh(np.divide(t,t01))**r1/np.tanh(np.divide(tp,t01))**r1 +
              (1 - alpha_w)*np.tanh(np.divide(t,t02))**r2/np.tanh(np.divide(tp,t02))**r2)
        FeH = FeH_0 + (FeH_p - FeH_0)*f_t
        return FeH
    
    
    def amrd_global_sj22_custom(self,t,R,p):
        """
        Thin-disk age-metallicity relation (AMR) equation as 
        defined in Sysoliatina and Just (2022). All parameters 
        must correspond to the same R (not necessarily to Rsun). 
        
        Parameters
        ----------
        t : scalar or array_like
            Galactic time, Gyr.
           
        Returns
        -------
        FeH : float or ndarray
            Metallicity of the MW-disk populations born 
            at Galactic time t. 
        """
        
        alpha_w = p.k_alphaw*R/p.Rsun + p.b_alphaw
        
        if R <= p.Rbr1: 
            FeH_p = p.k1_FeHdp*p.Rbr1/p.Rsun + p.b1_FeHdp
        else:
            FeH_p = p.k1_FeHdp*R/p.Rsun + p.b1_FeHdp
        
        if R <= p.Rbr2:
            t01 = p.k1_t01*R/p.Rsun + p.b1_t01
        else:
            if R <= p.Rbr3:
                t01 = p.k2_t01*R/p.Rsun + p.b2_t01
            else:
                t01 = p.k3_t01*R/p.Rsun + p.b3_t01
                
        if R <= p.Rbr1:
            t02 = p.k1_t02*R/p.Rsun + p.b1_t02
        else:
            if R <= p.Rbr3:
                t02 = p.k2_t02*R/p.Rsun + p.b2_t02
            else:
                t02 = p.k3_t02*R/p.Rsun + p.b3_t02    
        
        return self.amrd_global_sj22(t,t01,t02,p.rd1,p.rd2,alpha_w,p.FeHd0,FeH_p)
    
    
    def amrd_global_sj22_default(self,t,R,p):
        """
        Thin-disk age-metallicity relation (AMR) equation as defined 
        in Sysoliatina and Just (2022) with the best parameters.
        
        Parameters
        ----------
        t : scalar or array_like
            Galactic time, Gyr. 
        R : scalar
            Galactocentric distance, kpc. 
           
        Returns
        -------
        FeH : float or ndarray
            Metallicity of the local MW-disk populations, born 
            at Galactic time t at radius R. 
        """
        
        # Break radii (kpc) for t01 and t02 AMRd parameters
        # for Eq. self.amrd_global_sj22. 
        R_br1 = 6 
        R_br2 = 7.5 
        R_br3 = 9.75 
        
        r1 = 0.5 
        r2 = 1.5 
        FeH_0 = -0.81
        alpha_w = -0.39*R/p.Rsun + 0.96
        
        if R <= R_br1: 
            FeH_p = 0.39
        else:
            FeH_p = -0.58*R/p.Rsun + 0.81

        if R <= R_br2:
            t01 = 2.04*R/p.Rsun - 0.29
        else:
            if R <= R_br3:
                t01 = -2.69*R/p.Rsun + 5.03
            else:
                t01 = -3.49*R/p.Rsun + 6.7 
                
        if R <= R_br1:
            t02 = 18.06*R/p.Rsun - 6.57
        else:
            if R <= R_br3:
                t02 = 8.85*R/p.Rsun - 3.65
            else:
                t02 = -2.54*R/p.Rsun + 10.15         
                
        return self.amrd_global_sj22(t,t01,t02,r1,r2,alpha_w,FeH_0,FeH_p)
        
        
    def amrt_sj21(self,t,t0,r,FeH_0,FeH_p):
        """
        Thick-disk age-metallicity relation (AMR) equation as 
        defined in Sysoliatina and Just (2021).
        
        Parameters
        ----------
        t : scalar or array_like
            Galactic time, Gyr. 
        t0 : scalar
            Time-scale parameter, Gyr. 
        r : scalar
            Power index in the AMR equation. 
        FeH_0, FeH_p: scalar
            Initial and present-day metallicities of the MW thick 
            disk in the Solar neighbourhood. 
           
        Returns
        -------
        FeH : float or ndarray
            Metallicity of the local MW thick-disk populations, 
            which were born at Galactic time t.
        """
        
        FeH = FeH_0 + (FeH_p - FeH_0)*np.tanh(np.divide(t,t0))**r
        return FeH
    
    
    def amrt_sj21_default(self,t):
        """
        Thick-disk age-metallicity relation (AMR) equation as defined 
        in Sysoliatina and Just (2021), with best parameters 
        (Table 3). 
        
        Parameters
        ----------
        t : scalar or array_like
            Galactic time, Gyr. 
           
        Returns
        -------
        FeH : float or ndarray
            Metallicity of the local MW thick-disk populations, 
            which were born at Galactic time t.
        """
        
        FeH_0, FeH_p = -0.94, 0.04
        r = 0.77
        t0 = 0.97
        return self.amrt_sj21(t,t0,r,FeH_0,FeH_p)
    
    
    def amrt_sj22(self,t,t0,r,FeH_0,FeH_p):
        """
        Thick-disk age-metallicity relation (AMR) equation as 
        defined in Sysoliatina and Just (2022). Difference from 
        SJ21 is in normalization (impact negligible). 
        
        Parameters
        ----------
        t : scalar or array_like
            Galactic time, Gyr. 
        t0 : scalar
            Time-scale parameter, Gyr. 
        r : scalar
            Power index in the AMR equation. 
        FeH_0, FeH_p: scalar
            Initial and present-day metallicities of the MW thick 
            disk in the Solar neighbourhood (or some other R). 
           
        Returns
        -------
        FeH : float or ndarray
            Metallicity of the local MW thick-disk populations 
            born at Galactic time t.
        """
        
        FeH = FeH_0 + (FeH_p - FeH_0)*np.tanh(np.divide(t,t0))**r/np.tanh(np.divide(tp,t0))**r
        return FeH


    def amrt_sj22_default(self,t):
        """
        Thick-disk age-metallicity relation (AMR) equation as defined 
        in Sysoliatina and Just (2022) with the best parameters 
        (applicable for R = 4-14 kpc). 
        
        Parameters
        ----------
        t : scalar or array_like
            Galactic time, Gyr. 
           
        Returns
        -------
        FeH : float or ndarray
            Metallicity of the local MW thick-disk populations 
            born at Galactic time t.
        """
        
        FeH_0, FeH_p = -0.89, 0.04
        r = 1.04
        t0 = 0.96
        return self.amrt_sj22(t,t0,r,FeH_0,FeH_p)

    
    def amrr(self,p,a):
        """
        MW disk age-metallicity relation (AMR). Thin-disk AMR 
        parameters are assumed to be linear functions of Galactocentric 
        distance. Thick-disk AMR is the same across the disk. 
        
        Parameters
        ----------
        p : namedtuple
            Set of the model parameters from the parameter file. 
        a : namedtuple
            Collection of the fixed model parameters, useful 
            quantities and arrays.

        Returns
        -------
        amrd : ndarray
            Thin-disk AMR, array size is (a.Rbins,a.jd).
        """
    
        amrd = np.zeros((a.Rbins,a.jd))
        
        if p.fehkey==0:
            for i in range(a.Rbins):
                FeH_0 = p.FeHd0 + p.k_FeHd0*(a.R[i]-p.Rsun)
                FeH_p = p.FeHdp + p.k_FeHdp*(a.R[i]-p.Rsun)
                q = p.q + p.k_q*(a.R[i]-p.Rsun)
                r = p.rd + p.k_rd*(a.R[i]-p.Rsun)
                amrd[i] = self.amrd_sj21(a.t,tp,q,r,FeH_0,FeH_p)
        else:
            for i in range(a.Rbins):
                if p.fehkey==1:
                    amrd[i] = self.amrd_global_sj22_custom(a.t,a.R[i],p)
                if p.fehkey==2:
                    amrd[i] = self.amrd_global_sj22_default(a.t,a.R[i],p)
        return amrd
    
    
    def get_amr(self,fe_ax,nfe_cum,t_ax,nt_cum,a):
        """
        Reconstructs age-metallicity relation (AMR) from the 
        normalized cumulative distributions (NCD) of (1) observed 
        metallicities and (2) Galactic birth time (tp - modelled 
        ages) of the same populations. More in Sysoliatina and 
        Just (2021). 
        
        Parameters
        ----------
        fe_ax : array_like
            Bin centers of the metallicity NCD.
        nfe_cum : array_like
            y-values of the metallicity NCD.
        t_ax : array_like
            Bin centers of the birth time NCD.
        nt_cum : array_like
            y-values of the birth time NCD.
        a : namedtuple
            Collection of the fixed model parameters, useful 
            quantities and arrays.
            
        Returns
        -------
        derived_amr : ndarray
            Metallicity as a function of Galactic time, corresponds 
            to the time array a.t.
        """
    
        n_cum_common = np.linspace(0,1,a.jd)
        cnv = ConvertAxes()
        fe_range = np.where((nfe_cum > 0)&(nfe_cum < 1))[0]
        new_fe = cnv.get_closest(n_cum_common,nfe_cum[fe_range],fe_ax[fe_range])
        
        t_range = np.where(nt_cum < 1)[0]
        if len(t_range) < a.jd:
            new_t = cnv.get_closest(n_cum_common,nt_cum[t_range],t_ax[t_range])
            derived_amr = cnv.interpolate(a.t,new_t,new_fe)[t_range]
            p1 = np.poly1d(np.polyfit(a.t[len(t_range)-10:len(t_range)],derived_amr[len(t_range)-10:],1))
            amr_end = [p1(i) for i in a.t[len(t_range):]]
            derived_amr = np.concatenate((derived_amr,amr_end),axis=-1)
        else:
            new_t = cnv.get_closest(n_cum_common,nt_cum,t_ax)
            derived_amr = cnv.interpolate(a.t,new_t,new_fe)
                
        return derived_amr
    
    
    def chemical_disks(self,tab,feh_br,alpha_br):
        """
        Separation of the two populations in the [Fe/H]-[Alpha/Fe] 
        chemical abundance plane. Form of the separating border 
        is chosen based on the APOGEE Red Clump data and may be 
        not optimal for other data samples.
        The equation defining the border between high-alpha and 
        low-alpha populations is:
                     | AlphaFe1,        if [Fe/H] > Fe2, 
        [Alpha/Fe] = | k*[Fe/H] + b,    if Fe1 < [Fe/H] < Fe2, 
                     | AlphaFe2,        if [Fe/H] < Fe1
        
        Parameters
        ----------
        tab : list of array_likes
            [Fe/H] and [Alpha/Fe] data columns.
        feh_br : list
            Fe1 and Fe2. 
        alpha_br : list
            AlphaFe1 and AlphaFe2. 

        Returns
        -------
        low_alpha : ndarray
            Array indices of the low-alpha population.
        high_alpha : ndarray 
            Array indices of the high-alpha population.
        """
        
        FeH, AlphaFe = tab
        FeH1, FeH2 = feh_br
        AlphaFe1, AlphaFe2 = alpha_br
        k = (AlphaFe2 - AlphaFe1)/(FeH2 - FeH1)
        b = AlphaFe2 - k*FeH2
        
        n = len(FeH)
        disk_flag = np.linspace(1,1,n)
        
        for i in range(n):
            if FeH[i] > FeH2 and AlphaFe[i] < AlphaFe2:
                disk_flag[i] = 0
            if FeH[i] < FeH2 and FeH[i] > FeH1 and AlphaFe[i] < k*FeH[i] + b:
                disk_flag[i] = 0
            if FeH[i] < FeH1 and AlphaFe[i] < AlphaFe1:
                disk_flag[i] = 0
                
        low_alpha = np.where(disk_flag==0)[0]
        high_alpha = np.where(disk_flag==1)[0]
        
        return (low_alpha, high_alpha)
    
    
    def chemical_disks_sj21(self,tab):
        """
        Separation of the two populations in the [Fe/H]-[Alpha/Fe] 
        chemical abundance plane as used in Sysoliatina and Just 
        (2021) (Eq.6). The separating border was chosen based 
        on the APOGEE RC DR14, and can be not optimal for other 
        samples (also for other releases of this RC catalogue). 
        
        Parameters
        ----------
        tab : list of array_likes
            [Fe/H] and [Alpha/Fe] columns.

        Returns
        -------
        low_alpha : ndarray
            Array indices of the low-alpha population.
        high_alpha : ndarray 
            Array indices of the high-alpha population.
        """
        
        low_alpha, high_alpha = self.chemical_disks(tab,[-0.69,0.0],[0.18,0.07])
        return (low_alpha, high_alpha)
    
    
    def chemical_disks_mg(self,tab):
        """
        Separation of the two populations in the [Fe/H]-[Mg/Fe] 
        chemical abundance plane. Separating border is chosen based 
        on the RAVE data, can be not optimal for other samples. 
        
        Parameters
        ----------
        tab : list of array_likes
            [Fe/H] and [Mg/Fe] columns.

        Returns
        -------
        low_alpha : ndarray
            Array indices of the low-alpha population.
        high_alpha : ndarray 
            Array indices of the high-alpha population.
        """
        
        FeH, MgFe = tab
        
        n = len(FeH)
        disk_flag = np.linspace(1,1,n)
        for i in range(n):
            if FeH[i] > -1.0 and MgFe[i] < 0.2:
                disk_flag[i] = 0
                            
        low_alpha = np.where(disk_flag==0)[0]
        high_alpha = np.where(disk_flag==1)[0]
        
        return (low_alpha, high_alpha)
    
    
    def get_metcum(self,Rlim,zlim,tab,a,**kwargs):   
        """
        Calculates a normalized cumulative metallicity distribution 
        (NCMD) from the data. 
            
        Parameters
        ----------
        Rlim : list of scalars
            Range of Galactocentric distances, same units as in 
            column R in tab.
        zlim : list of scalars
            Range of heights above the Galactic plane, same units 
            as in column z in tab.
        tab : ndarray
            Array of shape (n,4), where n is an arbitrary length 
            of columns and 4 is the number of columns, which must 
            be given in the following order - (R,D,z,Fe). R and D 
            are Galactocentric and heliocentric distances (both 
            in the same units), respectively. z is the height 
            above/below the Galactic plane (can be absolute value). 
            Fe is metallicity. 
        a : namedtuple
            Collection of the fixed model parameters, useful 
            quantities and arrays.
        **kwargs : dict, optional keyword arguments
            Dmax : scalar
                Maximum heliocentric distance, must be in the same 
                units as column D in tab. 
                
        Returns
        -------
        FeH_bins : ndarray
            Histogram bin centrers.
        Ncum : ndarray
            y-values of the NCMD.
        Ncum_err : ndarray
            Errors of the y-values (Poisson noise in bins).
        FeH_mean : float
            Mean metallicity of the selected sample.
        """

        R1,R2 = Rlim
        zlow,zup = zlim
        R,D,z,Fe = tab.T
        
        if 'Dmax' in kwargs:
            tab = tab[np.logical_and.reduce([R>=R1,R<R2,np.abs(z)>=zlow,np.abs(z)<=zup,
                                             D<=kwargs['Dmax']])]
        else:
            tab = tab[np.logical_and.reduce([R>=R1,R<R2,np.abs(z)>=zlow,np.abs(z)<=zup])]
        R,D,z,Fe = tab.T
        n = len(R)
        
        FeH_mean = np.mean(Fe)              
        N, FeH_bins = np.histogram(Fe,a.jd)      

        dn = 50
        N1 = np.concatenate((np.linspace(0,0,dn),N,np.linspace(0,0,dn)),axis=-1)   
        N_smoothed = sgn.savgol_filter(N1,101,3)[dn:-dn]
        N_smoothed[N_smoothed < 0] = 0 
        N_noise = np.sqrt(N_smoothed) 
        Ncum = np.cumsum(N_smoothed)/np.cumsum(N_smoothed)[-1]
        FeH_bins = FeH_bins[:-1] + np.diff(FeH_bins)/2
        Ncum_err = np.array([np.sqrt(np.nansum((N_noise[:k]/n)**2)) for k in a.jd_array])
               
        return (FeH_bins, Ncum, Ncum_err, FeH_mean)
          
    
    def conv(self,x,k,b,sigma):
        """
        Analytical convolution of the linear function (k*x + b) 
        with the Gaussian kernel. See Eq.(17) in Sysoliatina and 
        Just (2021). 
        
        Parameters
        ----------
        x : array_like
            x-coordinates.
        k : scalar
            Slope of the linear function.
        b : scalar
            Intercept of the linear function.
        sigma : scalar
            Standard deviation, in the same units as x.

        Returns
        -------
        s : ndarray
            y-coordinates of the convolution.
        """
        
        s = ((-1/2*(k*x + b)*erf((b + k*x - 1)/np.sqrt(2)/k/sigma) 
             + 1/2*(k*x + b)*erf((k*x + b)/np.sqrt(2)/k/sigma) 
             + k*sigma/np.sqrt(2*np.pi)*(np.exp(-(b/k + x)**2/2/sigma**2)
             - np.exp(-(b + k*x - 1)**2/2/k**2/sigma**2)))
             + 1/2*(1 + erf((b + k*x - 1)/np.sqrt(2)/k/sigma))
             )
        return s


    def get_convolved(self,x,ycum,sigma,y_linpart):
        """
        Convolution of the normalized cumulative metallicity 
        distribution (NCMD) with the Gaussian kernel based on 
        analytic formula from self.conv. Only the upper part of 
        NCMD (ycum > 0.5) is convolved. 
        
        Parameters
        ----------
        x : array_like
            x-coordinates of NCMD (centers of metallicity bins).
        ycum : array_like
            y-coordinates of NCMD.
        sigma : scalar
            Standard deviation, in the same units as x.

        Returns
        -------
        ycum_convolved : ndarray
            y-coordinates of the NCMD convelved with the Gaussian 
            kernel.
        """
        
        y1, y2 = y_linpart       # Linear part of the NCMD
        dx = 0.00001
        
        ind1 = np.where(abs(ycum-y1)==np.amin(abs(ycum-y1)))[0][0]
        ind2 = np.where(abs(ycum-y2)==np.amin(abs(ycum-y2)))[0][0]
        
        line = np.poly1d(np.polyfit(x[ind1:ind2],ycum[ind1:ind2],1))
        linear_extrapolation = line(x)
        x1 = x[np.where(linear_extrapolation>0)[0][0]]
        x2 = x[np.where(linear_extrapolation<1)[0][-1]]
    
        x1min, x2max = x1 - dx, x2 + dx
        x1max, x2min = x1 + dx, x2 - dx
        k1, k2 = 1/(x2max-x1min), 1/(x2min-x1max)
        b1, b2 = -k1*x1min, -k2*x1max 
        k, b = np.sort([k1,k2]), np.sort([b1,b2])
        
        ind_mean1 = np.where(np.abs(ycum-0.5)==np.amin(np.abs(ycum-0.5)))[0][0]
        
        ycum_convolved_part = [self.conv(i,np.mean(k),np.mean(b),sigma) for i in x[ind_mean1:]]
        #ycum_convolved_part = [self.conv(i,line[0],line[1],sigma) for i in x[ind_mean1:]]
        ycum_convolved = np.concatenate((ycum[:ind_mean1],ycum_convolved_part),axis=-1)
        
        return ycum_convolved
        
        
    def get_deconvolved(self,x,ycum,y_linpart):
        """
        Reconstructs the 'true' normalized cumulative metallicity 
        distribution (NCMD), assuming that the observed NCMD is a
        convolution of the 'true' distribution with the Gaussian 
        kernel (which can be related to observational errors). See 
        Sysoliatina and Just (2021) for more details. 
        
        Parameters
        ----------
        x : array_like
            x-coordinates of NCMD (centers of metallicity bins).
        ycum : array_like
            y-coordinates of NCMD.
        sigma : scalar
            Standard deviation, in the same units as x.

        Returns
        -------
        ycum_deconvolved : ndarray
            y-coordinates of the reconstructed 'true' NCMD.
        ycum_convolved : ndarray
            ycum_deonvolved convolved with the Gaussian kernel 
            (with standard deviation sigma that is another output 
            quantity). Should reasonably reproduce the observed 
            NCMD. 
        sigma : float
            Standard deviation of the Gaussian kernel that has to 
            be convolved with the 'true' NCMD (ycum_deconvolved)  
            to reproduce the observed NCMD (ycum_convolved, ycum).
        """
        
        y1, y2 = y_linpart               # Linear part of the NCMD
        dx = 0.00001
        
        ind1 = np.where(abs(ycum-y1)==np.amin(abs(ycum-y1)))[0][0]
        ind2 = np.where(abs(ycum-y2)==np.amin(abs(ycum-y2)))[0][0]
        
        line = np.poly1d(np.polyfit(x[ind1:ind2],ycum[ind1:ind2],1))
        linear_extrapolation = line(x)
        x1 = x[np.where(linear_extrapolation>0)[0][0]]
        x2 = x[np.where(linear_extrapolation<1)[0][-1]]
    
        x1min, x2max = x1 - dx, x2 + dx
        x1max, x2min = x1 + dx, x2 - dx
        k1, k2 = 1/(x2max-x1min), 1/(x2min-x1max)
        b1, b2 = -k1*x1min, -k2*x1max 
        k, b = np.sort([k1,k2]), np.sort([b1,b2])
    
        ind_mean1 = np.where(np.abs(ycum-0.5)==np.amin(np.abs(ycum-0.5)))[0][0]
        
        popt,pcov = curve_fit(self.conv,x[ind_mean1:],ycum[ind_mean1:],
                              bounds=([k[0],b[0],0],[k[1],b[1],0.25]))                              
        sigma = popt[2]
        '''
        def conv(x,sigma):
            return self.conv(x,line[1],line[0],sigma)

        popt, pcov = curve_fit(conv,x[ind_mean1:],ycum[ind_mean1:],bounds=([0],[0.25]))                              
        sigma = popt
        ycum_linear_part = line(x[ind_mean1:])
        ycum_convolved_part = [conv(i,*popt) for i in x[ind_mean1:]]
        '''
        ycum_linear_part = [popt[0]*i+popt[1] for i in x[ind_mean1:]]
        ycum_convolved_part = [self.conv(i,*popt) for i in x[ind_mean1:]]
        ycum_deconvolved = np.concatenate((ycum[:ind_mean1],ycum_linear_part),axis=-1)
        ycum_convolved = np.concatenate((ycum[:ind_mean1],ycum_convolved_part),axis=-1)
        
        ycum_deconvolved[ycum_deconvolved > 1] = 1
        ycum_deconvolved[ycum_deconvolved < 0] = 0
        
        return (ycum_deconvolved, ycum_convolved, sigma)


    def get_deconvolved1(self,x,ycum,y_linpart):
        """
        Reconstructs the 'true' normalized cumulative metallicity 
        distribution (NCMD), assuming that the observed NCMD is a
        convolution of the 'true' distribution with the Gaussian 
        kernel (which can be related to observational errors). See 
        Sysoliatina and Just (2021) for more details. 
        
        Parameters
        ----------
        x : array_like
            x-coordinates of NCMD (centers of metallicity bins).
        ycum : array_like
            y-coordinates of NCMD.
        sigma : scalar
            Standard deviation, in the same units as x.

        Returns
        -------
        ycum_deconvolved : ndarray
            y-coordinates of the reconstructed 'true' NCMD.
        ycum_convolved : ndarray
            ycum_deonvolved convolved with the Gaussian kernel 
            (with standard deviation sigma that is another output 
            quantity). Should reasonably reproduce the observed 
            NCMD. 
        sigma : float
            Standard deviation of the Gaussian kernel that has to 
            be convolved with the 'true' NCMD (ycum_deconvolved)  
            to reproduce the observed NCMD (ycum_convolved, ycum).
        """
        
        y1, y2 = y_linpart              # Linear part of the NCMD
        dx = 0.00001
        
        ind1 = np.where(abs(ycum-y1)==np.amin(abs(ycum-y1)))[0][0]
        ind2 = np.where(abs(ycum-y2)==np.amin(abs(ycum-y2)))[0][0]
        
        line = np.poly1d(np.polyfit(x[ind1:ind2],ycum[ind1:ind2],1))
        linear_extrapolation = line(x)
        x1 = x[np.where(linear_extrapolation>0)[0][0]]
        x2 = x[np.where(linear_extrapolation<1)[0][-1]]
        
        ind_mean1 = np.where(np.abs(ycum-0.5)==np.amin(np.abs(ycum-0.5)))[0][0]
        '''
        x1min, x2max = x1 - dx, x2 + dx
        x1max, x2min = x1 + dx, x2 - dx
        k1, k2 = 1/(x2max-x1min), 1/(x2min-x1max)
        b1, b2 = -k1*x1min, -k2*x1max 
        k, b = np.sort([k1,k2]), np.sort([b1,b2])
    
        
        
        popt,pcov = curve_fit(self.conv,x[ind_mean1:],ycum[ind_mean1:],
                              bounds=([k[0],b[0],0],[k[1],b[1],0.25]))                              
        sigma = popt[2]
        '''
        def conv(x,sigma):
            return self.conv(x,line[1],line[0],sigma)

        popt, pcov = curve_fit(conv,x[ind_mean1:],ycum[ind_mean1:],bounds=([0],[0.25]))                              
        sigma = popt
        ycum_linear_part = line(x[ind_mean1:])
        ycum_convolved_part = [conv(i,*popt) for i in x[ind_mean1:]]
        
        #ycum_linear_part = [popt[0]*i+popt[1] for i in x[ind_mean1:]]
        #ycum_convolved_part = [self.conv(i,*popt) for i in x[ind_mean1:]]
        ycum_deconvolved = np.concatenate((ycum[:ind_mean1],ycum_linear_part),axis=-1)
        ycum_convolved = np.concatenate((ycum[:ind_mean1],ycum_convolved_part),axis=-1)
        
        ycum_deconvolved[ycum_deconvolved > 1] = 1
        ycum_deconvolved[ycum_deconvolved < 0] = 0
        
        return (ycum_deconvolved, ycum_convolved, sigma)

    
    def mass_loss_jj10_default(self):
        """
        Thin-disk mass loss function calculated with Chempy code, 
        consistent with the three-slope broken power law IMF from 
        Rybizki and Just (2015) and AMR from Just and Jahreiss 
        (2010). This mass loss faction is close to the one used 
        in Just and Jahreiss (2010). 
        
        Returns
        -------
        gd_jj10_default : ndarray
            Mass loss as a function of Galactic time, a.t.
        """
        
        t,gd_jj10_default = np.loadtxt(os.path.join(localpath,
                                                    'input','mass_loss','gd_jj10_default.txt')).T
        return gd_jj10_default


    def mass_loss_sj21_default(self):
        """
        Thin-disk mass loss function calculated with Chempy code, 
        consistent with the four-slope broken power law IMF and AMR 
        from Sysoliatina and Just (2021).
        
        Returns
        -------
        gd_sj21_default : ndarray
            Thin-disk mass loss as a function of Galactic time, 
            a.t.
        gt_sj21_default : ndarray
            Thick-disk mass loss as a function of Galactic time, 
            a.t[:a.jt].
        """
        
        t,gd_sj21_default = np.loadtxt(os.path.join(localpath,'input','mass_loss','gd_sj20_default.txt')).T
        gt_sj21_default = np.loadtxt(os.path.join(localpath,'input','mass_loss','gt_sj20_default.txt')).T[1]
        
        return (gd_sj21_default, gt_sj21_default)
        
    
    def mass_loss(self,t,FeH):
        """
        Mass loss for an arbitrary AMR (IMF is a four-slope broken 
        power law from Sysoliatina & Just 2021). 
        
        Parameters
        ----------
        t : array_like
            Galactic time, in Gyr.
        FeH : array_like
            Metallicity of stellar populations born at Galactic 
            time t.

        Returns
        -------
        g : ndarray
            Mass loss function (fraction in stars and remnants). 
        """
        
        if len(t)!=len(FeH):
            print('Error in mass_loss()! Lengths of time- and [Fe/H]-array must be equal!')
            return np.nan
        
        grid = np.load(os.path.join(localpath,'input','mass_loss','g_grid.npy'))
        
        dt = 0.025              # Gyr, time step used in the pre-calculated mass loss grid
        fe0, dfe = -2.0, 0.02   # Min metallicity and step used in the pre-calculated mass loss grid
        indt = map(int,t//dt)
        indfe = map(int,FeH//dfe - fe0//dfe + 1) 
                
        g = np.array([grid[k][i] for i,k in zip(indt,indfe)])
                
        return g        
        
    
    def z2fe(self,Z):
        """
        Function to convert mass fraction of metals Z to abundance 
        [Fe/H]. Formulae are taken from Choi et al. (2016). 
        The approach adopts a primordial helium abundance Yp = 0.249
        (Planck Collaboration et al. 2015) determined by combining 
        the Planck power spectra, Planck lensing, and a number of 
        `external data` such as baryonic acoustic oscillations. 
        In the equations below, a linear enrichment law to the 
        protosolar helium abundance, Y_protosolar = 0.2703 
        (Asplund et al.2009), is assumed. Once Y is computed for 
        a desired value of Z, X and [Fe/H] is trivial to compute.
        
        Parameters
        ----------
        Z : scalar or array_like
            Mass fraction of metals in a star.
     
        Returns
        -------
        FeH : float
            Chemical abundance of metals.
        """
        
        Yp = 0.249 
        Y_protosolar, Z_protosolar = 0.2703, 0.0142  
        Z_sun, X_sun = 0.0152, 0.7381   # Present-day solar abundaces from Asplund et al. (2009)
        
        Y = Yp + np.multiply((Y_protosolar-Yp)/Z_protosolar,Z) 
        X = 1 - Y - Z
        FeH = np.log10(np.divide(Z,X)/(Z_sun/X_sun)) 
            
        return FeH


    def fe2z(self,FeH):
        """
        Function to convert abundance [Fe/H] a to mass fraction 
        of metals Z. Formulae are taken Choi et al. (2016). 
        The  approach adopts a primordial helium abundance 
        Yp = 0.249 (Planck Collaboration et al. 2015) determined 
        by combining the Planck power spectra, Planck lensing,  
        and a number of `external data` such as baryonic acoustic 
        oscillations. In the equations below, a linear enrichment 
        law to the protosolar helium abundance, Y_protosolar = 
        0.2703 (Asplund et al.2009), is assumed.  
        
        Parameters
        ----------
        FeH : scalar or array_like
            Chemical abundance of metals. 
     
        Returns
        -------
        Z : float
            Mass fraction of metals in a star. 
        """
        
        Yp = 0.249 
        Y_protosolar, Z_protosolar = 0.2703, 0.0142 
        Z_sun, X_sun = 0.0152, 0.7381   # Present-day solar abundaces from Asplund et al. (2009)
    
        up = Z_sun/X_sun*np.power(10,FeH)*(1-Yp)
        down = 1 + Z_sun/X_sun*np.power(10,FeH)*(1+(Y_protosolar - Yp)/Z_protosolar)
        Z = up/down
        
        return Z
    
  
     
class AVR():
    """
    Collection of functions to work with the age-velocity dispersion 
    relation (AVR). 
    """
    
    def avr_jj10(self,t,tp,sigma_e,tau0,alpha):
        """
        Age-velocity dispersion relation (AVR) of the MW thin disk 
        in the Solar neighbourhood, as defined in Just and Jahreiss 
        (2010).
        
        Parameters
        ----------
        t : float or array_like
            Galactic time, Gyr. 
        tp : float
            Present-day age of the MW disk (i.e., present-day 
            Galactic time, in Gyr).
        sigma_e : float
            AVR scaling factor, W-velocity dispersion of the 
            oldest thin-disk stellar population in the Solar 
            neighbourhood, in km/s.
        tau0 : float
            AVR parameter, Gyr.
        alpha : float
            AVR power index.  
    
        Returns
        -------
        sigma_W : float or ndarray
            W-velocity dispersion of the thin-disk stellar 
            population born (tp - t) Gyr ago, km/s.  
        """
        
        sigma_W = sigma_e*(np.add(np.subtract(tp,t),tau0)/(tp + tau0))**alpha
        return sigma_W

    
    def avr_jj10_default(self,t,**kwargs):
        """
        Age-velocity dispersion relation (AVR) of the MW thin disk 
        in the Solar neighbourhood, as defined in Just and Jahreiss
         (2010), calculated with the best parameters (model A).
    
        Parameters
        ----------
        t : float or array_like
            Galactic time, Gyr.   
        **kwargs : dict, optional keyword arguments
            stretch : boolean
                If True, an extended time-scale relative to the old 
                model A is used. The present-day age of the MW 
                disk tp==13 Gyr, as in the updated JJ model from 
                Sysoliatina and Just (2021). By default is False, 
                tp==12 Gyr, as it was in the first paper of the  
                series, Just and Jahreiss (2010). 
 
        Returns
        -------
        sigma_W : float or ndarray
            W-velocity dispersion of the thin-disk stellar 
            population born (tp - t) Gyr ago, km/s.  
        """
        sigma_e = 25    # model A default parameters
        tau0 = 0.17 
        alpha = 0.375 
        
        if 'stretch' in kwargs and 'stretch'==True:
            tp_ = tp
        else: 
            tp_ = 12    
            
        sigma_W = self.avr_jj10(t,tp_,sigma_e,tau0,alpha)
        
        return sigma_W



class SFR():
    """
    Class with different definitions of the star formation rate 
    (SFR) function. 
    """
    
    def __init__(self):
        """
        Initialization includes reading several tables with the 
        default mass loss functions.  
        """
        g_jj10_table = np.loadtxt(os.path.join(localpath,'input','mass_loss','gd_jj10_default.txt')).T
        self.t, self.gd_jj10_default = g_jj10_table
        self.dt = np.diff(self.t)[0]
        self.gd_jj10_default_stretch = np.loadtxt(os.path.join(localpath,'input','mass_loss',
                                                               'gd_jj10_default_stretch.txt')).T[1]
        self.gd_sj21_default = np.loadtxt(os.path.join(localpath,'input','mass_loss',
                                                       'gd_sj21_default.txt')).T[1]
        self.gt_sj21_default = np.loadtxt(os.path.join(localpath,'input','mass_loss',
                                                       'gt_sj21_default.txt')).T[1]
        
    def sfrd_jj10(self,t,t0,t1,sigma,**kwargs):
        """
        SFR equation as defined in Just and Jahreiss (2010) 
        (model A). 
        
        Parameters
        ----------
        t : scalar or array_like
            Galactic time, Gyr.           
        tp: scalar
            Present-day age of the MW disk (i.e., present-day 
            Galactic time), Gyr.        
        t0, t1 : scalar
            SFR parameters, Gyr.            
        sigma : scalar
            Midplane present-day surface density of the thin disk, 
            Msun/pc^2.
        **kwargs : dict, optional keyword arguments
            g : array_like
                Mass loss function (for the whole Galatic time 
                range, [0,tp] Gyr). 
            
        Returns
        -------
        SFR : float or ndarray
            Star formation rate at time t, in Msun/pc^2/Gyr.             
        NSFR : float or ndarray
            Star formation rate at time t normalized on the SFR 
            averaged over the whole time range of the disk evolution 
            [0,tp] Gyr. 
        """
        
        if 'g' in kwargs:
            g0 = kwargs['g']
        else:
            g0 = self.gd_jj10_default
                
        normalized_sfr = np.add(t,t0)/np.power(np.add(np.power(t,2),t1**2),2) 
        normalized_sfr_full = np.add(self.t,t0)/np.power(np.add(np.power(self.t,2),t1**2),2) 
        SFR_scale = sigma/np.sum(g0*normalized_sfr_full*self.dt)
        SFR = SFR_scale*normalized_sfr
        SFR_mean = np.mean(SFR)
        
        return (SFR, SFR/SFR_mean)


    def sfrd_jj10_default(self,t):
        """
        SFR equation as defined in Just and Jahreiss (2010) 
        (model A) with its best parameters.
        
        Parameters
        ----------
        t : scalar or array_like
            Galactic time, Gyr.     
        **kwargs : dict, optional keyword arguments
            stretch : boolean
                If True, an extended time scale is used by setting 
                a present-day age of the MW disk to tp==13 Gyr as 
                in the updated JJ model in Sysoliatina and Just 
                (2021). By default is False, tp==12 Gyr, as in the 
                first paper of this series, Just and Jahreiss (2010). 
                   
        Returns
        -------
        SFR : float or ndarray
            Star formation rate at time t, in Msun/pc^2/Gyr. 
        NSFR : float or ndarray
            Star formation rate at time t normalized on the SFR 
            averaged over the whole time range of the thin disk 
            evolution [0,tp] Gyr. 
        """
        
        t0, t1 = 5.6, 8.2    
        sigma0 = 29.4 
            
        return self.sfrd_jj10(t,t0,t1,sigma0)
        

    def sfrd_sj21(self,t,dzeta,eta,t1,t2,sigma,**kwargs):
        """
        SFR of the thin disk, as defined in Sysoliatina and Just 
        (2021).
        
        Parameters
        ----------
        t : scalar or array_like
            Galactic time, Gyr.          
        dzeta, eta : scalar
            SFR power indices. 
        t1, t2 : scalar
            Parameters that define the SFR initial time-point 
            (by default, t1==0 Gyr, i.e., the thin disk starts to 
            form without a delay), and the SFR shape (parameter t2).            
        sigma : scalar
            Midplane present-day thin-disk surface density, 
            Msun/pc^2.
        **kwargs : dict, optional keyword arguments
            g : array_like
                Mass loss function (for the whole Galatic time 
                range, [0,tp] Gyr). 
            
        Returns
        -------
        SFR : float or ndarray
            Star formation rate at time t, in Msun/pc^2/Gyr. 
        NSFR : float or ndarray
            Star formation rate at time t normalized on the SFR 
            averaged over the whole time range of the thin disk 
            evolution [0,tp] Gyr. 
        """
        
        if 'g' in kwargs:
            g = kwargs['g']
        else:
            g = self.gd_sj20_default
        normalized_sfr = (np.power(t,2)-t1**2)**dzeta/np.add(t,t2)**eta
        SFR_scale = sigma/np.sum(g[int(t1/self.dt):]*normalized_sfr*self.dt)
        SFR = SFR_scale*normalized_sfr
        SFR_mean = np.mean(SFR)
        
        return (SFR, SFR/SFR_mean)
    
    
    def sfrd_sj21_mono_default(self,t):
        """
        SFR of the thin disk, as defined in Sysoliatina and Just 
        (2021), calculated with the best-fit parameters 
        (model MCMC1).
        
        Parameters
        ----------
        t : scalar or array_like
            Galactic time, Gyr.          
            
        Returns
        -------
        SFR : float or ndarray
            Star formation rate at time t, in Msun/pc^2/Gyr. 
        NSFR : float or ndarray
            Star formation rate at time t normalized on the SFR 
            averaged over the whole time range of the thin disk 
            evolution [0,tp] Gyr. 
        """
        
        dzeta, eta = 0.8, 5.6
        t1, t2 = 0, 7.8
        sigma = 29.4 
        return self.sfrd_sj21(t,dzeta,eta,t1,t2,sigma)


    def sfrd_sj21_multipeak(self,tp,tr,t,dzeta,eta,t1,t2,sigma,sigmap,tpk,dtp,**kwargs):
        """
        SFR of the thin disk, as defined in Sysoliatina and Just 
        (2021): a power-law function that prescribes a peak at old
        ages, a monotonously declining continuum after that, with 
        one or several overlaying Gaussian peaks. 
        
        Parameters
        ----------
        tp : scalar
            Present-day MW disk age, Gyr. 
        tr : scalar
            Time resolution of the model, Gyr.
        t : scalar or array_like
            Galactic time, Gyr.           
        dzeta, eta : scalar
            SFR power indices. 
        t1, t2 : scalar
            Parameters that define the SFR initial time-point (by 
            default, t1==0 Gyr, i.e., the thin disk starts to form 
            without a delay), and the SFR shape (parameter t2).            
        sigma : scalar
            Midplane present-day thin-disk surface density, 
            Msun/pc^2.
        sigmap : scalar or array_like 
            Amplitude-related parameter(s) of the additional 
            Gaussian peak(s), Msun/pc^2. 
        tpk : scalar or array_like
            Mean Galactic time(s) of the Gaussian peak(s) (where 
            they are centered along the time axis), Gyr. 
        dtp : scalar or array_like
            Dispersion(s) of the Gaussian peak(s), Gyr
        **kwargs : dict, optional keyword arguments
            g : array_like
                Mass loss function (for the whole Galatic time 
                range, [0,tp] Gyr). 
            
        Returns
        -------
        SFR : float or ndarray
            Star formation rate at time t, in Msun/pc^2/Gyr. 
        NSFR : float or ndarray
            Star formation rate at time t normalized on the SFR 
            averaged over the whole time range of the thin disk  
            evolution [0,tp] Gyr. 
        """
        
        jd = int((tp-t1)/tr)
        
        if 'g' in kwargs:
            g = kwargs['g']
        else:
            g = self.gd_sj21_default
        
        try:
            ind_peak = np.int(np.round(np.divide(tpk-t1,self.dt)))
        except:
            ind_peak = np.array(np.round(np.divide(tpk-t1,self.dt)),dtype=np.int)
        ind_peak_max = np.amax(ind_peak)        # The youngest peak population
        
        if ind_peak_max >= jd:                  # Peak center can be outside of the time axis
            t_max = self.dt*(ind_peak_max+1)
            t_long = np.arange(t1+self.dt/2,t_max+self.dt/2,self.dt)
        else: 
            t_long = t
        
        SFR_mono_long, normalized_sfr_mono_long = self.sfrd_sj21(t_long,dzeta,eta,t1,t2,sigma,**kwargs)
        SFR_mono_max = np.amax(SFR_mono_long)
        sfr_at_peak = normalized_sfr_mono_long[ind_peak]

        if type(sigmap)==float or type(sigmap)==int:
            peaks = sfr_at_peak*sigmap/SFR_mono_max*\
                    np.exp(-np.subtract(t_long,tpk)**2/2/dtp**2)
            normalized_sfr = np.add(normalized_sfr_mono_long[:jd],peaks[:jd])
        else:
            peaks = np.array([i*k/SFR_mono_max*np.exp(-np.subtract(t_long,l)**2/2/m**2) 
                              for i,k,l,m in zip(sfr_at_peak,sigmap,tpk,dtp)])
            
            normalized_sfr = np.add(normalized_sfr_mono_long[:jd],np.sum(peaks,axis=0)[:jd])
                                
        SFR_scale = sigma/np.sum(g[int(t1/self.dt):]*normalized_sfr*self.dt)
        SFR = SFR_scale*normalized_sfr
        SFR_mean = np.mean(SFR)
        
        return (SFR, SFR/SFR_mean, peaks/normalized_sfr)
    
    
    def sfrd_sj21_multipeak_default(self,t):   
        """
        SFR of the thin disk, as defined in Sysoliatina and Just 
        (2021): a power-law function that prescribes a peak at old 
        ages, a monotonously declining continuum after that, with 
        one or several overlaying Gaussian peaks. Calculated with 
        the best-fit parameters (model MCMC1). 
        
        Parameters
        ----------
        t : scalar or array_like
            Galactic time, Gyr.                  
             
        Returns
        -------
        SFR : float or ndarray
            Star formation rate at time t, in Msun/pc^2/Gyr. 
        NSFR : float or ndarray
            Star formation rate at time t normalized on the SFR 
            averaged over the whole time range of the thin disk 
            evolution [0,tp] Gyr. 
        """
        
        tp_, tr_ = 13, 0.025 
        dzeta, eta = 0.83, 5.59
        t1, t2 = 0, 7.8
        sigma = 29.3
        sigmap, tpk, dtp = [3.5,1.4], [10,12.5], [0.7,0.25]
        return self.sfrd_sj21_multipeak(tp_,tr_,t,gamma,beta,t1,t2,sigma,sigmap,tpk,dtp)
    
    
    def sfrt_sj21(self,t,gamma,beta,t1,t2,sigma,**kwargs):
        """
        SFR of the thick disk, as defined in Sysoliatina and Just (2021). 
        
        Parameters
        ----------
        t : scalar or array_like
            Galactic time, Gyr.            
        gamma, beta : scalar
            SFR power- and exponent- indices. 
        t1, t2 : scalar
            Parameters that define the initial and final time-
            points, in Gyr. By default, t1 == 0 Gyr, i.e., thick 
            disk starts to form without a delay, and t2 == 4 Gyr, 
            i.e., t2-t1 is the thick-disk formation duration.         
        sigma : scalar
            Midplane present-day thick-disk surface density, 
            Msun/pc^2.
        **kwargs : dict, optional keyword arguments
            g : array_like
                Mass loss function (for the thick-disk formation 
                duration time range, [t1,t2] Gyr). 
            
        Returns
        -------
        SFR : float or ndarray
            Star formation rate at time t, in Msun/pc^2/Gyr. 
        NSFR : float or ndarray
            Star formation rate at time t normalized on the SFR 
            averaged over the whole time range of the thick disk 
            evolution [t1,t2] Gyr. 
        """
        
        if 'g' in kwargs:
            g = kwargs['g'][:int(t2/self.dt)]
        else:
            g = self.gt_sj20_default[:int(t2/self.dt)]
            
        normalized_sfr = np.add(t,t1)**gamma*(np.exp(-np.multiply(beta,t))-np.exp(-beta*t2))
        normalized_sfr_full = np.add(self.t[:int(t2/self.dt)],t1)**gamma*\
                             (np.exp(-np.multiply(beta,self.t[:int(t2/self.dt)]))-np.exp(-beta*t2))
        SFR_mean = sigma/np.sum(g*normalized_sfr_full*self.dt)
        SFR = SFR_mean*normalized_sfr
        
        return (SFR, normalized_sfr)
    
    
    def sfrt_sj21_default(self,t):
        """
        Thick-disk SFR, as defined in Sysoliatina and Just (2021), 
        calculated with best parameters (model MCMC1). 
        
        Parameters
        ----------
        t : scalar or array_like
            Galactic time, Gyr.           
            
        Returns
        -------
        SFR : float or ndarray
            Star formation rate at time t, in Msun/pc^2/Gyr. 
        NSFR : float or ndarray
            Star formation rate at time t normalized on the SFR 
            averaged over the whole time range of the thick disk 
            evolution [t1,t2] Gyr. 
        """
        
        t1, t2 = 0.1, 4 
        gamma, beta = 3.5, 2 
        sigma = 4.9
        return self.sfrt_sj21(t,gamma,beta,t1,t2,sigma)
    
    
    def sfrr(self,p,a,gd,gd0,gt):
        """
        Thin- and thick-disk SFR as a function of Galactocentric 
        distance R. 
        
        Parameters
        ----------
        p : namedtuple
            Set of the model parameters from the parameter file. 
        a : namedtuple
            Collection of the fixed model parameters, useful 
            quantities and arrays.
        gd : array_like
            Thin-disk mass loss function (for the whole thin-disk 
            formation duration time range, [p.td1,p.tp] Gyr), at 
            different distances a.R.  
        gd0 : array_like
            The local thin-disk mass loss function.
        gt : array_like
            Thick-disk mass loss (for the whole thick-disk 
            formation duration time range, [p.tt1,p.tt2] Gyr). 
            Assumed to be the same for all distances a.R. 

        Returns
        -------
        TYPE
            DESCRIPTION.
        """
        
        SFRd, NSFRd = np.zeros((a.Rbins,a.jd)), np.zeros((a.Rbins,a.jd))        
        SFRt, NSFRt = np.zeros((a.Rbins,a.jt)), np.zeros((a.Rbins,a.jt))        
        SFRtot, NSFRtot = np.zeros((a.Rbins,a.jd)), np.zeros((a.Rbins,a.jd))        
        
        # SFR parameters are power-law functions of R: 
        dzeta_array = [p.dzeta*(i/p.Rsun)**p.k_dzeta for i in a.R]
        eta_array = [p.eta*(i/p.Rsun)**p.k_eta for i in a.R]
        td2_array = [p.td2*(i/p.Rsun)**p.k_td2 for i in a.R]
        
        # Exponential disks: 
        sigmad_array = [p.sigmad*np.exp(-(i-p.Rsun)/p.Rd) for i in a.R]
        sigmat_array = [p.sigmat*np.exp(-(i-p.Rsun)/p.Rt) for i in a.R]
        
        if p.pkey==0:
            for i in range(a.Rbins):
                SFRd[i], NSFRd[i] = self.sfrd_sj21(a.t,dzeta_array[i],eta_array[i],p.td1,
                                                   td2_array[i],sigmad_array[i],g=gd[i]
                                                   )             
            SFRd0, NSFRd0 = self.sfrd_sj21(a.t,p.dzeta,p.eta,p.td1,p.td2,p.sigmad,g=gd0) 
        else:    
            Fp = []
            sigmap_array = [p.sigmap*np.exp(-((i - p.Rp)**2/2/p.dRp**2))/\
                            np.exp(-((p.Rsun - p.Rp)**2/2/p.dRp**2)) for i in a.R]                       
                       
            for i in range(a.Rbins):
                SFRd[i], NSFRd[i], fpeak = self.sfrd_sj21_multipeak(tp,tr,a.t,dzeta_array[i],eta_array[i],
                                                             p.td1,td2_array[i],sigmad_array[i],
                                                             sigmap_array[i],p.tpk,p.dtp,g=gd[i]
                                                             )   
                Fp.append(fpeak)
            SFRd0, NSFRd0, Fp0 = self.sfrd_sj21_multipeak(tp,tr,a.t,p.dzeta,p.eta,p.td1,p.td2,p.sigmad,
                                                     p.sigmap,p.tpk,p.dtp,g=gd0
                                                     )
        for i in range(a.Rbins):
            SFRt[i], NSFRt[i] = self.sfrt_sj21(a.t[:a.jt],p.gamma,p.beta,p.tt1,p.tt2,sigmat_array[i],g=gt)
            
            SFRtot[i] = np.concatenate((np.add(SFRd[i][:a.jt],SFRt[i]),SFRd[i][a.jt:]),axis=None)
            NSFRtot[i] = SFRtot[i]/np.mean(SFRtot[i]) 
        
        SFRt0, NSFRt0 = self.sfrt_sj21(a.t[:a.jt],p.gamma,p.beta,p.tt1,p.tt2,p.sigmat,g=gt)
        SFRtot0 = np.concatenate((np.add(SFRd0[:a.jt],SFRt0),SFRd0[a.jt:]),axis=None)
        NSFRtot0 = SFRtot0/np.mean(SFRtot0)
        
        if p.pkey==0:
            return ((SFRd,NSFRd,SFRd0,NSFRd0),
                    (SFRt,NSFRt,SFRt0,NSFRt0),
                    (SFRtot,NSFRtot,SFRtot0,NSFRtot0))
        else:
            return ((SFRd,NSFRd,Fp,SFRd0,NSFRd0,Fp0),
                    (SFRt,NSFRt,SFRt0,NSFRt0),
                    (SFRtot,NSFRtot,SFRtot0,NSFRtot0))



class IMF():
    """
    Class for calling the initial mass function (IMF).
    Available IMFs (class methods):
        - BPL_4slopes: 4-slope broken power law (custom parameters)
        - BPL_4slopes_rj15_default: BPL from Rybizki and Just (2015)
        - BPL_4slopes_sj21_default: BPL from Sysoliatina and Just (2021)
        - Chabrier03: lognormal + power-law IMF from Chabrier (2003).
        - KTG93: 3-slope IMF from Kroupa et al. (1993).
    """
    
    def __init__(self,mlow,mup):
        self.mlow, self.mup = mlow, mup
        self.mres = 0.005               # Msun
        self.m_lin = np.linspace(self.mlow,self.mup,int((self.mup-self.mlow)//self.mres+2))

    
    def _bpl_4slopes_call_(self,mass,ka0,ka1,ka2,ka3,a0,a1,a2,a3,m1,m2,m3):   
        rez=0                
        if self.mlow <= mass < m1:
            rez = ka0*mass**(-a0)
        else:
            if m1 <= mass < m2:
                rez = ka1*mass**(-a1)
            if m2 <= mass < m3:
                rez = ka2*mass**(-a2)
            if m3 <= mass <= self.mup:
                rez = ka3*mass**(-a3)
                
        return rez
    
    
    def _chabrier03_call_(self,mass,ka1,a1,m1):   
        rez=0                
        if self.mlow <= mass < m1:
            rez = 0.158/np.log(10)/mass*np.exp(-(np.log10(mass)-np.log10(0.079))**2/2/0.69**2)
        else:
            rez = ka1*mass**(-a1)
        return rez
    
    
    def _ktg93_3slopes_call_(self,mass,ka1,ka2,ka3,a1,a2,a3,m1,m2):   
        rez=0                
        if self.mlow <= mass < m1:
            rez = ka1*mass**(-a1)
        else:
            if m1 <= mass < m2:
                rez = ka2*mass**(-a2)
            if m2 <= mass <= self.mup:
                rez = ka3*mass**(-a3)
        return rez
        
    '''
    def dndm_probability(self,mass1,mass2,Nmdm):
        """
        Calculates the probability of a star to be born with the 
        mass in the given interval. 
        
        Parameters
        ----------
        mass1, mass2 : scalar
            Mass interval of interest, Msun.  
        dndm_probability : array_like
            Full IMF in the form of an array.

        Returns
        -------
        dndm_probability_m12 : scalar
            Probability of a star to be born with mass within the 
            given interval [mass1,mass2]. Normalized to unity. 
        """
        if mass1 < self.mlow:
            mass1 = self.mlow
        if mass2 > self.mup:
            mass2 = self.mup
        m1_ind = int((mass1-self.mlow)//self.mres)
        m2_ind = int((mass2-self.mlow)//self.mres)
        if (m1_ind==m2_ind) or (m2_ind - m1_ind==1):
            m2_ind = m1_ind + 2 
            weight = (mass2-mass1)/(self.m_lin[m2_ind]-self.m_lin[m1_ind])
        else:   
            weight = 1 
        dndm_probability_m12 = weight*np.trapz(dndm_probability_func[m1_ind:m2_ind],
                                               self.m_lin[m1_ind:m2_ind])
        return dndm_probability_m12
    '''
    
    def number_stars(self,mass1,mass2):
        
        if mass1 < self.mlow:
            mass1 = self.mlow
        if mass2 > self.mup:
            mass2 = self.mup
            
        m1_ind = int((mass1-self.mlow)//self.mres)
        m2_ind = int((mass2-self.mlow)//self.mres)
        
        if (m1_ind==m2_ind) or (m2_ind - m1_ind==1):
            m2_ind = m1_ind + 2 
            weight = (mass2-mass1)/(self.m_lin[m2_ind]-self.m_lin[m1_ind])
        else:   
            weight = 1 
        #cut = np.where(np.logical_and(self.m_lin>=mass1,self.m_lin<mass2))
        number = np.sum(self.Nmdm[m1_ind:m2_ind])*weight
        return(number)
    

    def BPL_4slopes(self,a0,a1,a2,a3,m1,m2,m3):
        """
        A four-slope broken power law (BPL) IMF.
        
        Parameters
        ----------     
        a0, a1, a2, a3 : scalar
            IMF slopes. 
        m1, m2, m3 : scalar
            Masses, corresponding to the slope's breaks, Msun.
                   
        Returns
        -------            
        (m_lin, dndm_probability) : 2d-ndarray
            Full IMF - full mass interval [mlow,mup] (in Msun) 
            binned in a linear space, and probabilities 
            corresponding to these mass intervals. 
        """        
        # if mass2 - mass1 > 3*0.005:
        #     mres = 0.005
        # else:
        #     mres = (mass2-mass1)/3
        #mres = 0.005
        
        term1 = (m1**(-a0+2)-self.mlow**(-a0+2))/(-a0+2)
        term2 = (m2**(-a1+2)-m1**(-a1+2))/(-a1+2)
        term3 = (m3**(-a2+2)-m2**(-a2+2))/(-a2+2)
        term4 = (self.mup**(-a3+2)-m3**(-a3+2))/(-a3+2)
        ka0 = 1/(term1 + m1**(-a0+a1)*term2 + 
                 m1**(-a0+a1)*m2**(-a1+a2)*term3 + m1**(-a0+a1)*m2**(-a1+a2)*m3**(-a2+a3)*term4) 
        ka1 = ka0*m1**(-a0+a1)
        ka2 = ka1*m2**(-a1+a2)
        ka3 = ka2*m3**(-a2+a3)
        
        # frequencies corresponding to masses m_lin
        f = [self._bpl_4slopes_call_(i,ka0,ka1,ka2,ka3,a0,a1,a2,a3,m1,m2,m3) for i in self.m_lin]
        u = np.multiply(f,self.m_lin)
        u = np.divide(u,np.sum(u))
        self.Nmdm = np.divide(u,self.m_lin)
        
        return (self.m_lin, self.Nmdm)
    
    
    def BPL_4slopes_rj15_default(self):
        """
        A four-slope broken power law (BPL) IMF from Rybizki and 
        Just (2015).
        
        Parameters
        ----------
        mass1, mass2 : scalar
            Mass interval of interest, Msun.  

        Returns
        -------
        (m_lin, dndm_probability) : 2d-ndarray
            Full IMF : full mass interval [mlow,mup] (in Msun) 
            binned in a linear space, and probabilities 
            corresponding to these mass intervals. 
        """  
        
        a0, a1, a2, a3 = 1.26, 1.49, 3.02, 2.28
        m1, m2, m3 = 0.5, 1.39, 6
        return self.BPL_4slopes(a0,a1,a2,a3,m1,m2,m3)


    def BPL_4slopes_sj21_default(self):
        """
        A four-slope broken power law (BPL) IMF from Sysoliatina 
        and Just (2021).
        
        Parameters
        ----------
        mass1, mass2 : scalar
            Mass interval of interest, Msun.      
                   
        Returns
        -------
        (m_lin, dndm_probability) : 2d-ndarray
            Full IMF : full mass interval [mlow,mup] (in Msun) 
            binned in a linear space, and probabilities 
            corresponding to these mass intervals. 
        """  
        
        a0, a1, a2, a3 = 1.31, 1.5, 2.88, 2.28
        m1, m2, m3 = 0.49, 1.43, 6
        return self.BPL_4slopes(a0,a1,a2,a3,m1,m2,m3)


    def Chabrier03(self):
        """
        A lognormal + power-law IMF from Chabrier (2003).
        
        Parameters
        ----------
        mass1, mass2 : scalar
            Mass interval of interest, Msun.     
                   
        Returns
        -------
        (m_lin, dndm_probability) : 2d-ndarray
            Full IMF : full mass interval [mlow,mup] (in Msun) 
            binned in a linear space, and probabilities 
            corresponding to these mass intervals. 
        """  
        # if mass2 - mass1 > 3*0.005:
        #     mres = 0.005
        # else:
        #     mres = (mass2-mass1)/5
            
        a1 = 2.3
        m1 = 1
        ka1 = 0.158/np.log(10)/m1*np.exp(-(np.log10(m1)-np.log10(0.079))**2/2/0.69**2)/m1**(-a1)

        m_lin = np.linspace(self.mlow,self.mup,int((self.mup-self.mlow)//self.mres+2))
        f = [self._chabrier03_call_(i,ka1,a1,m1) for i in m_lin]
        
        u = f*self.m_lin
        u = np.divide(u,np.sum(u))
        self.Nmdm = np.divide(u,self.m_lin)
               
        return (m_lin, self.Nmdm)
    
    
    def KTG93(self):
        """
        A three-slope broken power law (BPL) IMF from Kroupa et al. 
        (1993).
        
        Parameters
        ----------
        mass1, mass2 : scalar
            Mass interval of interest, Msun.  
                   
        Returns
        -------
        (m_lin, dndm_probability) : 2d-ndarray
            Full IMF : full mass interval [mlow,mup] (in Msun) 
            binned in a linear space, and probabilities 
            corresponding to these mass intervals. 
        """  
            
        a1, a2, a3 = 1.3, 2.2, 2.7
        m1, m2 = 0.5, 1
        
        term1 = (m1**(-a1+2)-self.mlow**(-a1+2))/(-a1+2)
        term2 = (m2**(-a2+2)-m1**(-a2+2))/(-a2+2)
        term3 = (self.mup**(-a3+2)-m2**(-a3+2))/(-a3+2)
        ka1 = 1/(term1 + m1**(-a1+a2)*term2 + m1**(-a1+a2)*m2**(-a2+a3)*term3) 
        ka2 = ka1*m1**(-a1+a2)
        ka3 = ka2*m2**(-a2+a3) 
        
        m_lin = np.linspace(self.mlow,self.mup,int((self.mup-self.mlow)//self.mres+2))
        f = [self._ktg93_3slopes_call_(i,ka1,ka2,ka3,a1,a2,a3,m1,m2) for i in m_lin]
        
        u = f*self.m_lin
        u = np.divide(u,np.sum(u))
        self.Nmdm = np.divide(u,self.m_lin)
        
        return (m_lin, self.Nmdm)
    


def log_surface_gravity(Mf,L,Teff):
    """
    Function for calculation of surface gravity. 
    
    Parameters
    ----------
    Mf : scalar
        Stellar mass (present-day mass in isochrones), M_sun.
    L : scalar
        Stellar luminosity, L_sun.
    Teff : scalar
        Effective temperature, K.

    Returns
    -------
    logg : scalar
        Surface gravity, log(m/s^2).
    """
    
    G = 6.67*10**(-11)      # Gravitational constant, m3*kg-1*s-2
    sigma = 5.67*10**(-8)   # Stefan-Boltzmann constant, W*m−2*K−4
    L_sun = 3.828*10**26    # Solar luminosity, W
    M_sun = 1.988*10**30    # Solar mass, kg
    Teff_sun = 5778         # Solar temperature, K

    L_W = L*L_sun
    M_kg = Mf*M_sun
    
    # g_sun = 4*np.pi*sigma*Teff_sun**4*G*M_sun/L_sun
    g = 4*np.pi*sigma*Teff**4*G*M_kg/L_W  # Surface gravity, m*s-2
    g_sgs = g*1e2                         # In cm*s-2
        
    logg = np.log10(g_sgs)
    return logg
    




