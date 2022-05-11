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
from .tools import _transition_2curves_, ConvertAxes
from .constants import tp, tr, GA, G, GYR, PC, SIGMA_E, KM, M_SUN
from . import localpath


# =============================================================================
# Functions
# =============================================================================

def hgr(p,a):
    """
    Scale heights of the atomic and molecular gas components as 
    functions of Galactocentric distance. Data are taken from Nakanishi and Sofue (2016). 
    
    :param p: Set of model parameters from the parameter file. 
    :type p: namedtuple
    :param a: Collection of the fixed model parameters, useful quantities, and arrays.
    :type a: namedtuple
        
    :return: *((hg1,hg10),(hg2,hg20))*, indices 1 and 2 correspond to molecular and atomic gas, 
        respectively. Gas scale heights (pc) at Galactocentric distances ``a.R`` and at the Solar 
        radius ``p.Rsun``. 
    :rtype: ((1d-array,float),(1d-array,float))
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


def heffr(p,a,heffd0):
    """
    Thin-disk half-thickness as a function of Galactocentric distance. 

    :param p: Set of model parameters from the parameter file. 
    :type p: namedtuple
    :param a: Collection of the fixed model parameters, useful quantities, and arrays.
    :type a: namedtuple
    :param heffd0: Thin-disk half-thickness at the Solar radius ``p.Rsun``.
    :type heffd0: scalar
    
    :return: Thin-disk half-thickness calculated at Galactocentric distances ``a.R``, pc. 
    :rtype: 1d-array        
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
        hdeff = _transition_2curves_(epsilon_R,p.Rf,a.R,hdeff)
    else:
        hdeff = [heffd0 for i in a.R]
    
    return np.array(hdeff)


def log_surface_gravity(Mf,L,Teff):
    """
    Function for calculation of surface gravity. 
    
    :param Mf: Stellar mass (present-day mass in isochrones), :math:`\mathrm{M}_\odot`.
    :type Mf: scalar or array-like
    :param L: Stellar luminosity, :math:`\mathrm{L}_\odot`.
    :type L: scalar or array-like
    :param Teff: Effective temperature, K.
    :type Teff: scalar or array-like
    
    :return: Log surface gravity, :math:`\mathrm{log(cm \ s^{-2})}`.
    :rtype: scalar or array-like
    """
    
    G = 6.67*10**(-11)      # Gravitational constant, m3*kg-1*s-2
    sigma = 5.67*10**(-8)   # Stefan-Boltzmann constant, W*m−2*K−4
    L_sun = 3.828*10**26    # Solar luminosity, W
    M_sun = 1.988*10**30    # Solar mass, kg
    Teff_sun = 5778         # Solar temperature, K

    L_W = np.multiply(L,L_sun)
    M_kg = np.multiply(Mf,M_sun)
    
    # g_sun = 4*np.pi*sigma*Teff_sun**4*G*M_sun/L_sun
    g = 4*np.pi*sigma*Teff**4*G*M_kg/L_W  # Surface gravity, m*s-2
    g_sgs = g*1e2                         # In cm*s-2
        
    logg = np.log10(g_sgs)
    return logg
    

# =============================================================================
# Classes
# =============================================================================

class RadialPotential():
    """
    Galactic potential in the midplane. 
    """
    
    def __init__(self,Rsun,R):
        """
        Class instance is initialized by two parameters.
        
        :param Rsun: Solar Galactocentric distance, kpc.
        :type Rsun: float
        :param R: Galactocentric distance(s) where potential has to be calculated, kpc.  
        :type R: float or array-like          
        """
        self.R, self.Rsun = R, Rsun
        
        
    def exp_disk(self,sigma0,Rd):
        """
        Potential of a razor-thin exponential disk (via Bessel functions). 
        
        :param sigma0: Local surface density, :math:`\mathrm{M_\odot \ pc^{-2}}`.    
        :type sigma0: scalar
        :param Rd: Radial scale length of the disk, kpc. 
        :type Rd: scalar
               
        :return: Potential of the disk at the given Galactocentric distance(s) **R**, 
            :math:`\mathrm{m^2 \ s^{-2}}`.             
        :rtype: float or array-like
        """
    
        y = self.R/2/Rd
        Fi = -np.pi*G*sigma0*M_SUN/PC**2*self.R*1e3*PC*\
            (iv(0,y)*kv(1,y) - iv(1,y)*kv(0,y)) 
        return Fi
    
    
    def pow_law(self,rho0,alpha):
        """
        Potential of a MW component with a power-law radial density profile, 
        :math:`{\\rho}(R) \propto (R_\odot/R)^{\\alpha}`. 
        
        :param rho0: Local mass density, :math:`\mathrm{M_\odot \ pc^{-3}}`. 
        :type rho0: scalar
        :param alpha: Power-law slope.
        :type alpha: scalar
               
        :return: Potential at the given Galactocentric distance(s) **R**, :math:`\mathrm{m^2 \ s^{-2}}`.             
        :rtype: float or array-like
        """
    
        Fi = 4*np.pi*G*rho0*M_SUN/PC**3/(3-alpha)/(2-alpha)*\
            (self.Rsun/self.R)**alpha*(self.R*1e3*PC)**2 
        return Fi
        
    
    def cored_iso_sphere(self,rho0,ah):
        """
        Potential of a cored isothermal sphere. 

        :param rho0: Local mass density, :math:`\mathrm{M_\odot \ pc^{-3}}`. 
        :type rho0: scalar
        :param ah: Scaling parameter, kpc. 
        :type ah: scalar
        
        :return: Potential the given Galactocentric distance(s) **R**, :math:`\mathrm{m^2 \ s^{-2}}`. 
        :rtype: float or array-like
        """
    
        C = (ah**2 + self.Rsun**2)/ah**2
        Fi = 4*np.pi*G*rho0*C*ah**2*(ah/self.R*np.arctan(self.R/ah) +\
                                     0.5*np.log(1 + (self.R/ah)**2))*M_SUN/PC
        return Fi



class RadialDensity():
    """
    Radial density profiles of the different Galactic components.     
    """
    
    def __init__(self,Rsun,zsun,R):
        """
        Class instance is initialized by three parameters.
        
        :param Rsun: Solar Galactocentric distance, kpc.
        :type Rsun: float
        :param zsun: Height of the Sun above the Galactic plane, pc.
        :type zsun: float
        :param R: Galactocentric distance(s) where density has to be calculated, kpc.  
        :type R: float or array-like       
        """
        self.Rsun = Rsun
        self.zsun = zsun*1e-3
        self.rsun = np.sqrt(Rsun**2 + (zsun*1e-3)**2)
        self.R = R
        
        
    def rho_disk(self,rho0,Rd):
        """
        Midplane mass density of an exponential disk.
    
        :param rho0: Local mass density, :math:`\mathrm{M_\odot \ pc^{-3}}`. 
        :type rho0: scalar
        :param Rd: Radial scale length of the disk, kpc. 
        :type Rd: scalar
        
        :return: Mass density of the disk calculated at the given Galactocentric distance(s) **R**, 
            :math:`\mathrm{M_\odot \ pc^{-3}}`.             
        :rtype: float or array-like  
        """
        rho = rho0*np.exp(-(np.subtract(self.R,self.Rsun))/Rd)
        return rho
    
    
    def sigma_disk(self,sigma0,Rd):
        """
        Surface density of an exponential disk.
    
        :param sigma0: Local surface density, :math:`\mathrm{M_\odot \ pc^{-2}}`. 
        :type sigma0: scalar
        :param Rd: Radial scale length of the disk, kpc. 
        :type Rd: scalar
        
        :return: Surface density of the disk calculated at the given Galactocentric distance(s) **R**, 
            :math:`\mathrm{M_\odot \ pc^{-2}}`. 
        :rtype: float or array-like 
        """
        
        sigma = sigma0*np.exp(-(np.subtract(self.R,self.Rsun))/Rd)
        return sigma
        
    
    def rho_dm_halo(self,z,rho0,ah):
        """
        3d mass density of an isothermal dark matter (DM) sphere.
    
        :param z: Height above the Galactic plane, kpc.
        :type z: scalar
        :param rho0: Local mass density of the DM halo, :math:`\mathrm{M_\odot \ pc^{-3}}`.
        :type rho0: scalar
        :param ah: DM scaling parameter, kpc. 
        :type ah: scalar
        
        :return: DM halo mass density at the given **z** and **R**, :math:`\mathrm{M_\odot \ pc^{-3}}`.
        :rtype: float or array-like
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
        Surface density of an isothermal dark matter (DM) sphere.
        
        :param zmax: Maximal height above the Galactic plane, kpc. Up to this height DM mass 
            density law will be integrated.
        :type zmax: scalar
        :param sigma0: Local surface density, :math:`\mathrm{M_\odot \ pc^{-2}}`. 
        :type sigma0: scalar 
        :param ah: DM scaling parameter, kpc. 
        :type ah: scalar 
      
        :return: DM halo surface density at Galactocentric distance(s) **R** 
                 (up to the height **zmax**), :math:`\mathrm{M_\odot \ pc^{-2}}`.   
        :rtype: float or array-like
        """
        tan_term = np.arctan(zmax/np.sqrt(ah**2 + self.R**2))/np.arctan(zmax/np.sqrt(ah**2 + self.Rsun**2))
        sigma = sigma0*np.sqrt(ah**2 + self.Rsun**2)/np.sqrt(ah**2 + self.R**2)*tan_term    
        return sigma

    
    def rho_stellar_halo(self,z,rho0,a_sh):
        """
        3d mass density of a spherical stellar halo.
        Flattening is ignored, profile is a power law, :math:`{\\rho}(R) \propto (R_\odot/R)^{-\\alpha}`. 
        
        :param z: Height above the Galactic plane, kpc. 
        :type z: scalar
        :param rho0: Local mass density of the halo, :math:`\mathrm{M_\odot \ pc^{-3}}`. 
        :type rho0: scalar
        :param a_sh: Slope of the halo profile (about -2.5, see Bland-Hawthorn 2016). 
        :type a_sh: scalar
                     
        :return: Halo mass density at the given **z** and **R**, 
            :math:`\mathrm{M_\odot \ pc^{-3}}`.    
        :rtype: float or array-like
        """
    
        #rho = rho0*(np.divide(self.Rsun,self.R))**(-a_sh)
        r = np.sqrt(self.R**2 + z**2)
        rho = rho0*(self.rsun/r)**(-a_sh)
        return rho
    
    
    def sigma_stellar_halo(self,zmax,sigma0,a_sh):
        """
        Surface density of a spherical stellar halo.
        Flattening is ignored, profile is a power law (see :meth:`jjmodel.RadialDensity.rho_stellar_halo`). 
        
        :param zmax: Maximal height above the Galactic plane, kpc. 
            Up to this height halo mass density law will be integrated.
        :type zmax: scalar
        :param sigma0: Local surface density, :math:`\mathrm{M_\odot \ pc^{-2}}`.   
        :type sigma0: scalar
        :param a_sh: Slope of the halo profile (about -2.5, see Bland-Hawthorn 2016). 
        :type a_sh: scalar
             
        :return:  Halo surface density at the given Galactocentric distance(s) **R** 
            (up to the height **zmax**), :math:`\mathrm{M_\odot \ pc^{-2}}`.       
        :rtype: float or array-like
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
    Circular velocity as a function of Galactocentric distance 
    as follows from the assumed density laws for the MW components.
    """
    
    def __init__(self,Rsun,R):
        """
        Class instance is initialized by two parameters. 
        
        :param Rsun: Solar Galactocentric distance, kpc.
        :type Rsun: float
        :param R: Galactocentric distance(s) where circular velocity has to be calculated, kpc.  
        :type R: float or array-like       
        """
        self.Rsun = Rsun
        self.R = R
        self.m_r = M_SUN/PC/1e3
        self.rho_r2 = M_SUN/PC*1e6
        self.sigma_r = M_SUN/PC*1e3
        
    def vc_bulge(self,Mb):
        """
        Rotation curve of a point-mass bulge. 
        
        :param Mb: Mass of the bulge, :math:`\mathrm{M_\odot}`. 
        :type Mb: scalar

        :return: Circular velocity corresponding to **R**, :math:`\mathrm{km \ s^{-1}}`. 
        :rtype: scalar or array-like        
        """
        
        vc = np.sqrt(G*Mb/self.R*self.m_r)/1e3
        return vc

    def vc_disk(self,sigma0,Rd,R0):
        """
        Rotation curve of an infinitely thin exponential disk 
        (Eq. 2-169 in Binney and Tremaine).
        
        :param sigma0: Local surface density, :math:`\mathrm{M_\odot \ pc^{-2}}`.   
        :type sigma0: scalar
        :param Rd: Radial scale length of the disk, kpc.
        :type Rd: scalar
        :param R0: Radius of the inner hole in the disk density profile, kpc. 
        :type R0: scalar  
        
        :return: Circular velocity corresponding to **R**, :math:`\mathrm{km \ s^{-1}}`. 
        :rtype: scalar or array-like  
        """
        
        sigma = sigma0*np.exp((self.Rsun - R0)/Rd)
        R = self.R - R0
        vc = np.sqrt(np.pi*G*sigma*R**2/Rd*\
                     (iv(0,R/2/Rd)*kv(0,R/2/Rd)-iv(1,R/2/Rd)*kv(1,R/2/Rd))*self.sigma_r)/1e3
        vc[vc*0!=0]=0
        return vc

    def vc_halo_nfw(self,rho0,ah):
        """
        Rotation curve of dark matter (DM) halo with NWF profile.
        
        :param rho0: Local DM mass density, :math:`\mathrm{M_\odot \ pc^{-3}}`. 
        :type rho0: scalar
        :param ah: Scaling parameter, kpc.
        :type ah: scalar
        
        :return: Circular velocity corresponding to **R**, :math:`\mathrm{km \ s^{-1}}`. 
        :rtype: scalar or array-like  
        """
        
        C = (self.Rsun/ah)*(1 + self.Rsun/ah)**2
        #vc = np.sqrt(4*np.pi*G*rho0*C*ah**3/self.R*\
        #             (np.log(1+self.R/ah)-self.R/ah/(1+self.R/ah))*self.rho_r2)/1e3
        vc = np.sqrt(4*np.pi*G*rho0*C*ah**2*\
                     (np.log(1+self.R/ah)/self.R*ah - 1/(1 + self.R/ah))*self.rho_r2)/1e3
        return vc

    def vc_halo_cored_iso_sphere(self,rho0,ah):
        """
        Rotation curve of dark matter (DM) halo, which is a cored isothermal sphere.
        
        :param rho0: Local DM mass density, :math:`\mathrm{M_\odot \ pc^{-3}}`. 
        :type rho0: scalar
        :param ah: Scaling parameter, kpc.
        :type ah: scalar 
            
        :return: Circular velocity corresponding to **R**, :math:`\mathrm{km \ s^{-1}}`. 
        :rtype: scalar or array-like  
        """
        
        C = (ah**2 + self.Rsun**2)/ah**2
        vc = np.sqrt(4*np.pi*G*rho0*C*ah**2*(1 - ah/self.R*np.arctan(self.R/ah))*self.rho_r2)/1e3
        
        return vc
    
    def _vc_halo_power_law_(self,rho0,alpha):
        """
        Rotation velocity for stellar halo with a power-law profile.
        Not physical, gives too large enclosed halo mass. Must be 
        truncated at some R near GC, or core has to be added. 
        
        :param rho0: Halo density at Rsun, :math:`\mathrm{M_\odot \ pc^{-3}}`. 
        :type rho0: scalar
        :param alpha: Power-law slope (rho ~ 1/R^alpha, make sure that you give alpha > 0).
        :type alpha: scalar
            
        :return: Circular velocity corresponding to **R**, :math:`\mathrm{km \ s^{-1}}`. 
        :rtype: scalar or array-like  
        """
    
        vc = np.sqrt(4*np.pi*G*rho0*self.Rsun**alpha/(3 - alpha)*self.R**(2 - alpha)*self.rho_r2)/1e3
        return vc
            
    
    def vc_tot(self,vc_array):
        """
        Total rotation curve as quadratic sum of all velocity components.         
        
        :param vc_array: :math:`{\\upsilon}_\mathrm{c}` components at distance(s) **R**. 
            Output units are the same as for the input velocities. 
        :type vc_array: array-like (list or list[list])
            
        :return: Circular velocity corresponding to **R**, :math:`\mathrm{km \ s^{-1}}`. 
        :rtype: scalar or array-like 
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
        Calculates total circular velocity :math:`{\\upsilon}_\mathrm{c}` at the Solar radius, ``p.Rsun``.  
        
        :param vc_tot: Total circular velocity at the given Galactocentric distance(s) **R**. 
            Note that this method is only useful when **R** is an array and covers Solar neighbourhood 
            with a decent resolution. 
        :type vc_tot: array-like
            
        :return: Local circular velocity corresponding to **R**, :math:`\mathrm{km \ s^{-1}}`.
        :rtype: scalar
        """
        
        ind_r0 = np.where(np.abs(self.R-self.Rsun)==np.amin(np.abs(self.R-self.Rsun)))[0]
        vc0 = vc_tot[ind_r0]
        return vc0



class AMR():
    """
    Collection of methods which are used to define age-metallicity relation (AMR) 
    and work with metallicities in the model and data.  
    """
        
    def amrd_jj10(self,t,tp,q,r,FeH_0,FeH_p):
        """
        Thin-disk AMR from Just and Jahreiss (2010), Eq.(31).
        
        :param t: Galactic time, Gyr. 
        :type t: scalar or array-like
        :param tp: Present-day age of the MW disk (present-day Galactic time), Gyr.
        :type tp: scalar
        :param q: AMR parameter.
        :type q: scalar
        :param r: AMR power index.
        :type r: scalar
        :param FeH_0: Initial metallicity of the thin disk in the Solar neighbourhood. 
        :type FeH_0: scalar
        :param FeH_p: Present-day metallicity of the thin disk in the Solar neighbourhood. 
        :type FeH_p: scalar

        :return: Metallicities of the local thin-disk populations born at Galactic times **t**.
        :rtype: float or array-like             
        """
        
        a = 2.67
        OH_0, OH_p = FeH_0/a, FeH_p/a
        ZO_0, ZO_p=10**OH_0,10**OH_p
        ZO = ZO_0 + (ZO_p-ZO_0)*np.log(1+q*(np.divide(t,tp)**r))/np.log(1+q)
        FeH = np.log10(ZO)*a
        
        return FeH
    

    def amrd_jj10_default(self,t,**kwargs):
        """
        Thin-disk AMR from Just and Jahreiss (2010), Eq.(31), calculated with 
        the best parameters (model A). 
        
        :param t: Galactic time, Gyr. 
        :type t: scalar or array-like
        :param stretch: Optional. If True, an extended (relative to model A) 
                time scale is used - a present-day MW disk age is set to **tp** = 13 Gyr. 
                By default is False, **tp** = 12 Gyr, as in Just and Jahreiss (2010).
        :type stretch: boolean
                                
        :return: Metallicities of the local thin-disk populations born at Galactic times **t**.
        :rtype: float or array-like      
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
        Thin-disk AMR from Sysoliatina and Just (2021), Eq. (21) and (22).
        
        :param t: Galactic time, Gyr. 
        :type t: scalar or array-like
        :param tp: Present-day age of the MW disk (present-day Galactic time), Gyr.
        :type tp: scalar
        :param q: AMR parameter (impacts the function shape).
        :type q: scalar
        :param r: AMR power index.
        :type r: scalar
        :param FeH_0: Initial metallicity of the thin disk in the Solar neighbourhood. 
        :type FeH_0: scalar
        :param FeH_p: Present-day metallicity of the thin disk in the Solar neighbourhood. 
        :type FeH_p: scalar
            
        :return: Metallicities of the local thin-disk populations born at Galactic times **t**.
        :rtype: float or array-like  
        """
        
        FeH = FeH_0 + (FeH_p - FeH_0)*np.log(1+q*(np.divide(t,tp)**r))/np.log(1+q)
        return FeH
    
    
    def amrd_sj21_default(self,t):
        """
        Thin-disk AMR from Sysoliatina and Just (2021), Eq. (21) and (22), 
        calculated with the best parameters (Table 3).
        
        :param t: Galactic time, Gyr. 
        :type t: scalar or array-like
           
        :return: Metallicities of the local thin-disk populations born at Galactic times **t**.
        :rtype: float or array-like 
        """
        
        tp_ = tp
        q, r = -0.72, 0.34
        FeH_0, FeH_p = -0.7, 0.29
        return self.amrd_sj21(t,tp_,q,r,FeH_0,FeH_p)
    
    
    def amrd_global_sj22(self,t,t01,t02,r1,r2,alpha_w,FeH_0,FeH_p):
        """
        Thin-disk AMR from Sysoliatina and Just (2022), Eq. (22) and (23). 
        All parameters must correspond to the same Galactocentric distance (but not necessarily to 
        the Solar radius ``p.Rsun``). 
        
        :param t: Galactic time, Gyr. 
        :type t: scalar or array-like
        :param t01: Time-scale parameter of the first *tanh* term, Gyr.
        :type t01: scalar
        :param t02: Time-scale parameter of the second *tanh* term, Gyr.
        :type t02: scalar
        :param r1: Power index of the first *tanh* term. 
        :type r1: scalar
        :param r2: Power index of the second *tanh* term. 
        :type r2: scalar
        :param FeH_0: Initial metallicity of the thin disk. 
        :type FeH_0: scalar
        :param FeH_p: Present-day metallicity of the thin disk. 
        :type FeH_p: scalar
           
        :return: Metallicities of the thin-disk populations born at Galactic times **t**.
        :rtype: float or array-like  
        """
        
        f_t = (alpha_w*np.tanh(np.divide(t,t01))**r1/np.tanh(np.divide(tp,t01))**r1 +
              (1 - alpha_w)*np.tanh(np.divide(t,t02))**r2/np.tanh(np.divide(tp,t02))**r2)
        FeH = FeH_0 + (FeH_p - FeH_0)*f_t
        return FeH
    
    
    def amrd_global_sj22_custom(self,t,R,p):
        """
        Thin-disk AMR from Sysoliatina and Just (2022), Eq. (22) and (23), 
        calculated for the specified Galactocentric distance. Extension of the local parameters 
        to an arbitrary radius is done as explained 
        in Table 2 in Sysoliatina and Just (2022). 
        
        :param t: Galactic time, Gyr. 
        :type t: scalar or array-like
        :param R: Galactocentric distance for which AMR has to be calculated, kpc. 
            Note that the recommended range is 4-14 kpc (see the paper). 
        :type R: scalar
        :param p: Set of model parameters from the parameter file. 
        :type p: namedtuple
        
        :return: Metallicities of the thin-disk populations born at Galactic times **t**.
        :rtype: float or ndarray  
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
        Thin-disk AMR from Sysoliatina and Just (2022), Eq. (22) and (23), 
        calculated for the specified Galactocentric distance with the best parameters (Table 2).  
        
        :param t: Galactic time, Gyr. 
        :type t: scalar or array-like
        :param R: Galactocentric distance for which AMR has to be calculated, kpc. 
            Note that the recommended range is 4-14 kpc (see the paper). 
        :type R: scalar
        :param p: Set of model parameters from the parameter file. 
        :type p: namedtuple
        
        :return: Metallicities of the local thin-disk populations born at Galactic times **t**.
        :rtype: float or ndarray  
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
        Thick-disk AMR from Sysoliatina and Just (2021), Eq. (21) and (22). 
        Applicable to the Solar neighbourhood and other Galactocentric distances 
        (AMR of the thick disk is assumed to be inpedended of radius). 
        
        :param t: Galactic time, Gyr. 
        :type t: scalar or array-like
        :param t0: Time-scale parameter, Gyr. 
        :type t0: scalar
        :param r: Power index.
        :type r: scalar
        :param FeH_0: Initial metallicity of the thick disk in the Solar neighbourhood. 
        :type FeH_0: scalar
        :param FeH_p: Present-day metallicity of the thick disk in the Solar neighbourhood. 
        :type FeH_p: scalar
           
        :return: Metallicities of the thick-disk populations born at Galactic times **t**.
        :rtype: float or array-like  
        """
        
        FeH = FeH_0 + (FeH_p - FeH_0)*np.tanh(np.divide(t,t0))**r
        return FeH
    
    
    def amrt_sj21_default(self,t):
        """
        Thick-disk AMR from Sysoliatina and Just (2021), Eq. (21) and (22), 
        calculated with the best parameters (Table 3). 
                
        :param t: Galactic time, Gyr. 
        :type t: scalar or array-like
           
        :return: Metallicities of the thick-disk populations born at Galactic times **t**.
        :rtype: float or array-like
        """
        
        FeH_0, FeH_p = -0.94, 0.04
        r = 0.77
        t0 = 0.97
        return self.amrt_sj21(t,t0,r,FeH_0,FeH_p)
    
    
    def amrt_sj22(self,t,t0,r,FeH_0,FeH_p):
        """
        Thick-disk AMR from Sysoliatina and Just (2021), Eq. (22) and (23).  
        Difference from SJ21 is normalization (impact negligible). 
        
        :param t: Galactic time, Gyr. 
        :type t: scalar or array-like
        :param t0: Time-scale parameter, Gyr. 
        :type t0: scalar
        :param r: Power index.
        :type r: scalar
        :param FeH_0: Initial metallicity of the thick disk in the Solar neighbourhood. 
        :type FeH_0: scalar
        :param FeH_p: Present-day metallicity of the thick disk in the Solar neighbourhood. 
        :type FeH_p: scalar
           
        :return: Metallicities of the thick-disk populations born at Galactic times **t**.
        :rtype: float or array-like  
        """
        
        FeH = FeH_0 + (FeH_p - FeH_0)*np.tanh(np.divide(t,t0))**r/np.tanh(np.divide(tp,t0))**r
        return FeH


    def amrt_sj22_default(self,t):
        """
        Thick-disk AMR from Sysoliatina and Just (2021), Eq. (22) and (23), calculated with 
        the best parameters (Table 2). Independent of radius, applicable for Galactocentric distances 
        R = 4-14 kpc. 
        
        :param t: Galactic time, Gyr. 
        :type t: scalar or array-like
           
        :return: Metallicities of the local thick-disk populations born at Galactic times **t**.
        :rtype: float or array-like  
        """
        
        FeH_0, FeH_p = -0.89, 0.04
        r = 1.04
        t0 = 0.96
        return self.amrt_sj22(t,t0,r,FeH_0,FeH_p)

    
    def amrr(self,p,a):
        """
        Thin-disk AMR across the disk. 
        If ``p.fehkey=0``, AMR is from Sysoliatina and Just (2021), and AMR parameters are assumed 
        to be linear functions of Galactocentric distance (old version of the AMR generalization). 
        If ``p.fehkey=1`` or ``p.fehkey=2``, AMR and its extension approach are as in 
        Sysoliatina and Just (2022). In case of ``p.fehkey=1``, parameters are taken from 
        the parameter file (i.e., custom), and when ``p.fehkey=2``, 
        parameters are default (best from Sysoliatina and Just 2022). 
        
        :param p: Set of model parameters from the parameter file. 
        :type p: namedtuple
        :param a: Collection of the fixed model parameters, useful quantities, and arrays.
        :type a: namedtuple  

        :return: Thin-disk AMR calculated for the Galactocentric distances ``a.R``, 
            array size is ``(a.Rbins,a.jd)``.
        :rtype: 2d-array 
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
        Reconstructs AMR from a normalized cumulative metallicity distribution function (CMDF) 
        and a normalized cumulative age distribution function (CADF). The first can be taken from 
        some observational data, the latter is modeled. Both CMDF and CADF correspond to the same 
        stellar population. See Sysoliatina and Just (2021) and Sysoliatina and Just (2022) for 
        the approach description. 
        
        :param fe_ax: Bin centers of the CMDF metallicity grid, dex.
        :type fe_ax: array-like
        :param nfe_cum: CMDF y-values corresponding to fe_ax. 
        :type nfe_cum: array-like
        :param t_ax: Bin centers of CADF (Galactic time, not age!), Gyr. Length of t_ax does not need 
            to be the same as of fe_ax. 
        :type t_ax: array-like
        :param nt_cum: CADF y-values corresponding to t_ax.
        :type nt_cum: array-like
        :param a: Collection of the fixed model parameters, useful quantities, and arrays.
        :type a: namedtuple  
        
        :return: Metallicity as a function of Galactic time, array corresponds 
            to the time array ``a.t``.
        :rtype: 1d-array
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
        Method used to separate two populations in :math:`\mathrm{[Fe/H]}\\text{-}\mathrm{[{\\alpha}/Fe]}` plane. 
        Shape of the separating border was chosed based on the APOGEE Red Clump data 
        (may be not optimal for other data samples). Equation defining the border 
        between high-:math:`\\alpha` and low-:math:`\\alpha` populations has the form:
        :math:`\mathrm{[\\alpha/Fe]} = \mathrm{[\\alpha/Fe]}_1, \  \\text{if} \ \mathrm{[Fe/H]} < \mathrm{[Fe/H]}_1`, 
        :math:`\mathrm{[\\alpha/Fe]} = k \mathrm{[Fe/H]} + b,  \ \\text{if} \ \mathrm{[Fe/H]}_1 < \mathrm{[Fe/H]} < \mathrm{[Fe/H]}_2`, 
        :math:`\mathrm{[\\alpha/Fe]} = \mathrm{[\\alpha/Fe]}_2, \ \\text{if} \ \mathrm{[Fe/H]} > \mathrm{[Fe/H]}_2`
        
        :param tab: :math:`\mathrm{[Fe/H]}` and :math:`\mathrm{[\\alpha/Fe]}` data columns.
        :type tab: list[array-likes]
        :param feh_br: Break points :math:`\mathrm{[Fe/H]}_1` and :math:`\mathrm{[Fe/H]}_2`.
        :type feh_br: list
        :param alpha_br: Parameters :math:`\mathrm{[\\alpha/Fe]}_1` and :math:`\mathrm{[\\alpha/Fe]}_2`. 
        :type alpha_br: list
            
        :return: Arrays with indices of **tab** rows corresponding to the low- and 
            high-:math:`\\alpha` populations.
        :rtype: list[1d-array]
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
        Same as :meth:`jjmodel.funcs.AMR.chemical_disks`, but with the specified separating border 
        location (for the APOGEE RC DR14). Can be not optimal for other data samples 
        (also for other releases of the RC catalogue). 
        
        :param tab: :math:`\mathrm{[Fe/H]}` and :math:`\mathrm{[\\alpha/Fe]}` data columns.
        :type tab: list[array-likes]

        :return: Arrays with indices of **tab** rows corresponding to the low- and 
            high-:math:`\\alpha` populations.
        :rtype: list[1d-array]
        """
        
        low_alpha, high_alpha = self.chemical_disks(tab,[-0.69,0.0],[0.18,0.07])
        return (low_alpha, high_alpha)
    
    
    def chemical_disks_mg(self,tab):
        """
        Same as :meth:`jjmodel.funcs.AMR.chemical_disks`, but for :math:`\mathrm{[Fe/H]}\\text{-}\mathrm{[Mg/Fe]}`
        plane. Separating border is adapted for the RAVE DR5, may be not optimal for other data samples. 
        
        :param tab: :math:`\mathrm{[Fe/H]}` and :math:`\mathrm{[Mg/Fe]}` data columns.
        :type tab: list[array-likes]

        :return: Arrays with indices of **tab** rows corresponding to the low- and 
            high-:math:`\\alpha` populations.
        :rtype: list[1d-array]
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
        Calculates normalized cumulative metallicity distribution function (CMDF) from the data. 
        
        :param Rlim: Range of Galactocentric distances, same units as for Galactocentric distance 
            column *R* in the input data **tab**.
        :type Rlim: list[scalar]
        :param zlim: Range of heights above the Galactic plane, same units as for column *z* 
            in the input data **tab**.
        :type zlim: list[scalar]
        :param tab: Array of the shape *(n,4)*, where *n* is an arbitrary length of all columns. 
                Columns order is *(R,D,z,Fe)*. *R* and *D* are Galactocentric and heliocentric distances 
                (both in the same units, pc or kpc). *z* is distance from the Galactic plane 
                (can be also just absolute value). *Fe* is metallicity. 
        :type tab: ndarray
        :param a: Collection of the fixed model parameters, useful quantities, and arrays.
        :type a: namedtuple
        :param Dmax: Optional, maximal heliocentric distance, same units as in column *D* in **tab**. 
        :type Dmax: scalar         
        
        :return: The output list consists of *(FeH_bins, Ncum, Ncum_err, FeH_mean)*. 
                *FeH_bins* is the metallicity grid with bin centers. *Ncum* contains CMDF y-values
                corresponding to *FeH_bins*. *Ncum_err* is a column of y-errors (Poisson noise in bins). 
                *FeH_mean* is a mean metallicity of the selected sample.
        :rtype: list[1d-array,1d-array,1d-array,float]            
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
        Analytical convolution of the linear function :math:`(k*x + b)` with a Gaussian kernel. 
        See Eq.(20) in Sysoliatina and Just (2022). 
        
        :param x: x-coordinates.
        :type x: array-like
        :param k: Slope of the linear function.
        :type k: scalar
        :param b: Intercept of the linear function.
        :type b: scalar
        :param sigma: Standard deviation of the kernel, same units as for **x**.
        :type sigma: scalar
        
        :return: y-coordinates of the convolution corresponding to **x** grid. 
        :rtype: 1d-array  
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
        Convolution of the normalized cumulative metallicity distribution function (CMDF) 
        with a Gaussian kernel based on :meth:`jjmodel.funcs.AMR.conv`. 
        Only the upper part of CMDF (**ycum** > 0.5) is convolved. 
        
        :param x: x-coordinates of CMDF (centers of metallicity bins).
        :type x: array-like
        :param ycum: y-coordinates of CMDF corresponding to **x**. 
        :type ycum: array-like
        :param sigma: Standard deviation of the kernel, same units as for **x**.
        :type sigma: scalar    
        
        :return: y-coordinates of the convolved CMDF corresponding to **x** grid. 
        :rtype: 1d-array
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
        Reconstructs 'true' normalized cumulative metallicity distribution function (CMDF) 
        assuming that the observed CMDF is a convolution of the 'true' distribution with a Gaussian 
        kernel (e.g. related to observational errors). See Sysoliatina and Just (2021) and 
        Sysoliatina and Just (2022) for details. 
        
        :param x: x-coordinates of CMDF (centers of metallicity bins).
        :type x: array-like
        :param ycum: y-coordinates of CMDF corresponding to **x**. 
        :type ycum: array-like
        :param y_linpart: y-values of the input CMDF giving the range 
                where the distribution is approximately linear.  
        :type y_linpart: list[scalar]    

        :return: Output is the list *(ycum_deconvolved,ycum_convolved,sigma)*. 
            *ycum_deconvolved* contains y-coordinates of the reconstructed 'true' CMDF.
            *ycum_convolved* is a convolution of *ycum_deconvolved* with a kernel with 
            the standard deviation *sigma*. Should reasonably reproduce the input CMDF. 
        :rtype: list[1d-array,1d-array,float]        
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

    
    def mass_loss_jj10_default(self):
        """
        Thin-disk mass loss function calculated with the **Chempy** code consistent with the three-slope 
        broken power-law IMF from Rybizki and Just (2015) and AMR from Just and Jahreiss (2010). 
        This mass loss function is close to the one used in Just and Jahreiss (2010). 
        
        :return: Mass loss as a function of Galactic time ``a.t``.
        :rtype: 1d-array            
        """
        
        t, gd_jj10_default = np.loadtxt(os.path.join(localpath,'input','mass_loss','gd_jj10_default.txt')).T
        return gd_jj10_default


    def mass_loss_sj21_default(self):
        """
        Thin-disk mass loss function calculated with the **Chempy** code consistent with the four-slope 
        broken power-law IMF and AMR from Sysoliatina and Just (2021).
        
        :return: Thin-disk mass loss as a function of Galactic time ``a.t`` and 
            thick-disk mass loss as a function of Galactic time ``a.t[:a.jt]``. 
        :rtype: list[1d-array]
        """
        
        t, gd_sj21_default = np.loadtxt(os.path.join(localpath,'input','mass_loss','gd_sj20_default.txt')).T
        gt_sj21_default = np.loadtxt(os.path.join(localpath,'input','mass_loss','gt_sj20_default.txt')).T[1]
        
        return (gd_sj21_default, gt_sj21_default)
        
    
    def mass_loss(self,t,FeH):
        """
        Mass loss for an arbitrary AMR (IMF 
        is a four-slope broken power law from Sysoliatina and Just 2021).         
        
        :param t: Galactic time, Gyr.
        :type t: array-like
        :param FeH: Metallicity of stellar populations born at Galactic time **t**.
        :type FeH: array-like
        
        :return: Mass loss function (fraction in stars and remnants) as a function of time **t**. 
        :rtype: 1d-array
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
        Function to convert mass fraction of metals *Z* into abundance :math:`\mathrm{[Fe/H]}`. 
        Formulae are taken 
        from Choi et al. (2016). The approach adopts a primordial helium abundance *Yp* = 0.249
        (Planck Collaboration et al. 2015) determined by combining the Planck power spectra, 
        Planck lensing, and a number of 'external data' such as baryonic acoustic oscillations. 
        In the equations below, a linear enrichment law to the protosolar helium abundance, 
        *Y_protosolar* = 0.2703 (Asplund et al.2009), is assumed. Once *Y* is computed for a desired 
        value of *Z*, *X* and :math:`\mathrm{[Fe/H]}` is trivial to compute.
        
        :param Z: Mass fraction of metals in a star.
        :type Z: scalar or array-like
            
        :return: Chemical abundance of metals, dex. 
        :rtype: float or array-like
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
        Function to convert abundance :math:`\mathrm{[Fe/H]}` into mass fraction of metals *Z*. 
        Formulae are taken 
        from Choi et al. (2016). The approach adopts a primordial helium abundance *Yp* = 0.249
        (Planck Collaboration et al. 2015) determined by combining the Planck power spectra, 
        Planck lensing, and a number of 'external data' such as baryonic acoustic oscillations. 
        In the equations below, a linear enrichment law to the protosolar helium abundance, 
        *Y_protosolar* = 0.2703 (Asplund et al.2009), is assumed. 
        
        :param FeH: Chemical abundance of metals, dex. 
        :type FeH: scalar or array-like

        :return: Mass fraction of metals in a star.
        :rtype: float or array-like
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
    Collection of methods to work with the age-velocity dispersion relation (AVR). 
    Here velocity dispersion is :math:`\\sigma_\mathrm{W}`, the vertical component. 
    """
    
    def avr_jj10(self,t,tp,sigma_e,tau0,alpha):
        """
        Thin-disk AVR in the Solar neighbourhood as defined in Just and Jahreiss (2010).
        
        :param t: Galactic time, Gyr. 
        :type t: float or array-like
        :param tp: Present-day age of the MW disk (i.e., present-day Galactic time), Gyr.
        :type tp: scalar 
        :param sigma_e: AVR scaling factor, W-velocity dispersion of the oldest thin-disk 
            stellar population in the Solar neighbourhood, :math:`\mathrm{km \ s^{-1}}`.    
        :type sigma_e: float
        :param tau0: AVR parameter, Gyr.
        :type tau0: scalar
        :param alpha: AVR power index.  
        :type alpha: scalar 
        
        :return: Present-day W-velocity dispersion of the thin-disk populations at time **t**, 
            :math:`\mathrm{km \ s^{-1}}`.  
        :rtype: float or array-like
        """
        
        sigma_W = sigma_e*(np.add(np.subtract(tp,t),tau0)/(tp + tau0))**alpha
        return sigma_W

    
    def avr_jj10_default(self,t,**kwargs):
        """
        Thin-disk AVR in the Solar neighbourhood as defined in Just and Jahreiss (2010) 
        calculated with the best parameters (model A).
    
        :param t: Galactic time, Gyr. 
        :type t: float or array-like
        :param stretch: Optional. If True, uses an extended (relative to model A) 
                time scale by setting a present-day age of the MW disk to **tp** = 13 Gyr. 
                By default is False, **tp** = 12 Gyr, as in Just and Jahreiss (2010).
        :type stretch: boolean 
        
        :return: Present-day W-velocity dispersion of the thin-disk populations at time **t**, 
            :math:`\mathrm{km \ s^{-1}}`.    
        :rtype: float or array-like
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
    Class with definitions of the MW disk star formation rate (SFR) function. 
    """
    
    def __init__(self):
        """
        Initialization includes reading several tables with the default mass loss functions, 
        no parameters have to be specified. 
        """
        g_jj10_table = np.loadtxt(os.path.join(localpath,'input','mass_loss','gd_jj10_default.txt')).T
        self.t, self.gd_jj10_default = g_jj10_table
        self.dt = np.diff(self.t)[0]
        self.gd_jj10_default_stretch = np.loadtxt(os.path.join(localpath,'input','mass_loss',
                                                               'gd_jj10_default_stretch.txt')).T[1]
        self.gd_sj21_default = np.loadtxt(os.path.join(localpath,'input','mass_loss','gd_sj21_default.txt')).T[1]
        self.gt_sj21_default = np.loadtxt(os.path.join(localpath,'input','mass_loss','gt_sj21_default.txt')).T[1]
        
    def sfrd_jj10(self,t,t0,t1,sigma,**kwargs):
        """
        Thin-disk SFR equation as defined in Just and Jahreiss (2010) (model A). 
        
        :param t: Galactic time, Gyr. 
        :type t: scalar or array-like
        :param tp: Present-day age of the MW disk (i.e., present-day Galactic time), Gyr.
        :type tp: scalar 
        :param t0: SFR parameter, Gyr. 
        :type t0: scalar
        :param t1: SFR parameter, Gyr. 
        :type t1: scalar 
        :param sigma: Midplane local present-day surface density of the thin disk, 
            :math:`\mathrm{M_\odot \ pc^{-2}}`.      
        :type sigma: scalar
        :param g: Optional, mass loss function (same length as **t**). 
        :type g: scalar or array-like
        
        :return: Absolute and normalized SFR as functions of **t**, 
            :math:`\mathrm{M_\odot \ pc^{-2} \ Gyr^{-1}}`. 
            Normalization factor is the mean SFR 
            (averaged over the whole time range of the disk evolution 0-**tp** Gyr). 
        :rtype: list[1d-array] or list[float]             
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
        SFR equation from Just and Jahreiss (2010) (model A)  
        (:meth:`jjmodel.funcs.SFR.sfrd_jj10`) calculated with the best parameters.
        
        :param t: Galactic time, Gyr. 
        :type t: scalar or array-like
        :param stretch: Optional. If True, uses an extended (relative to model A) 
                time scale by setting a present-day age of the MW disk to **tp** = 13 Gyr. 
                By default is False, **tp** = 12 Gyr, as in Just and Jahreiss (2010).
        :type stretch: boolean 
                   
        :return: Absolute and normalized SFR as a function of **t**, 
            :math:`\mathrm{M_\odot \ pc^{-2} \ Gyr^{-1}}`. 
            Normalization factor is the mean SFR 
            (averaged over the whole time range of the disk evolution 0-**tp** Gyr). 
        :rtype: list[1d-array] or list[float]     
        """
        t0, t1 = 5.6, 8.2    
        sigma0 = 29.4 
            
        return self.sfrd_jj10(t,t0,t1,sigma0)
        

    def sfrd_sj21(self,t,dzeta,eta,t1,t2,sigma,**kwargs):
        """
        SFR of the thin disk as defined in Sysoliatina and Just (2021).
        
        :param t: Galactic time, Gyr. 
        :type t: scalar or array-like  
        :param dzeta: SFR power index. 
        :type dzeta: scalar 
        :param eta: SFR power index. 
        :type eta: scalar 
        :param t1: Parameter defining the SFR initial time-point (by default, **t1** = 0 Gyr, i.e., 
                the thin disk starts to form without a delay). 
        :type t1: scalar 
        :param t2: Parameter controlling the SFR shape. 
        :type t2: scalar 
        :param sigma: Midplane present-day thin-disk surface density, :math:`\mathrm{M_\odot \ pc^{-2}}`.        
        :type sigma: scalar
        :param g: Optional, mass loss function (same length as **t**). 
        :type g: scalar or array-like
            
        :return: Absolute and normalized SFR as a function of **t**, 
            :math:`\mathrm{M_\odot \ pc^{-2} \ Gyr^{-1}}`. Normalization factor 
            is the mean SFR (averaged over the whole time range of the disk evolution 0-**tp** Gyr). 
        :rtype: list[1d-array] or list[float]    
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
        SFR of the thin disk, as defined in Sysoliatina and Just (2021) 
        (:meth:`jjmodel.funcs.SFR.sfrd_sj21`) calculated with 
        the best-fit parameters (model MCMC1).
        
        :param t: Galactic time, Gyr. 
        :type t: scalar or array-like 
        
        :return: Absolute and normalized SFR as a function of **t**, 
            :math:`\mathrm{M_\odot \ pc^{-2} \ Gyr^{-1}}`. Normalization factor 
            is the mean SFR (averaged over the whole time range of the disk evolution 0-**tp** Gyr). 
        :rtype: list[1d-array] or list[float]    
        """
        
        dzeta, eta = 0.8, 5.6
        t1, t2 = 0, 7.8
        sigma = 29.4 
        return self.sfrd_sj21(t,dzeta,eta,t1,t2,sigma)


    def sfrd_sj21_multipeak(self,tp,tr,t,dzeta,eta,t1,t2,sigma,sigmap,tpk,dtp,**kwargs):
        """
        SFR of the thin disk as defined in Sysoliatina and Just (2021) with any number of extra Gaussian 
        peaks. Also, see Eqs. (7)-(10) in Sysoliatina and Just (2022). 
        
        :param tp: Present-day age of the MW disk (i.e., present-day Galactic time), Gyr.
        :type tp: scalar 
        :param tr: Time resolution of the model, Gyr.
        :type tr: scalar 
        :param t: Galactic time, Gyr. 
        :type t: scalar or array-like 
        :param dzeta: SFR power index. 
        :type dzeta: scalar 
        :param eta: SFR power index. 
        :type eta: scalar 
        :param t1: Parameter defining the SFR initial time-point (by default, **t1** = 0 Gyr, i.e., 
                the thin disk starts to form without a delay). 
        :type t1: scalar 
        :param t2: Parameter controlling the SFR shape. 
        :type t2: scalar            
        :param sigma: Midplane present-day thin-disk surface density, :math:`\mathrm{M_\odot \ pc^{-2}}`.        
        :type sigma: scalar
        :param sigmap: Amplitude-related parameter(s) of the additional Gaussian peak(s), 
            :math:`\mathrm{M_\odot \ pc^{-2}}`. 
        :type sigmap: scalar or array-like 
        :param tpk: Mean Galactic time(s) of the Gaussian peak(s), Gyr. 
        :type tpk: scalar or array-like 
        :param dtp: Dispersion(s) of the Gaussian peak(s), Gyr
        :type dtp: scalar or array-like
        :param g: Optional, mass loss function (same length as **t**). 
        :type g: scalar or array-like
            
        :return: Absolute and normalized SFR as a function of **t**, 
            :math:`\mathrm{M_\odot \ pc^{-2} \ Gyr^{-1}}`, and ratio of the peaks' contributions
            to the total SFR. Normalization factor is the mean SFR 
            (averaged over the whole time range of the disk evolution 0-**tp** Gyr). 
        :rtype: [1d-array,1d-array,list[1d-array]] or [float,float,list[float]]    
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
        SFR of the thin disk as defined in Sysoliatina and Just (2021) with two extra Gaussian peaks 
        (see :meth:`jjmodel.funcs.SFR.sfrd_sj21_multipeak`). 
        Calculated with the best-fit parameters (model MCMC1). 
        
        :param t: Galactic time, Gyr. 
        :type t: scalar or array-like               
             
        :return: Absolute and normalized SFR as a function of **t**, 
            :math:`\mathrm{M_\odot \ pc^{-2} \ Gyr^{-1}}`, and ratio of the peaks' contributions
            to the total SFR. Normalization factor is the mean SFR 
            (averaged over the whole time range of the disk evolution 0-**tp** Gyr). 
        :rtype: [1d-array,1d-array,list[1d-array]] or [float,float,list[float]]  
        """
        
        tp_, tr_ = 13, 0.025 
        dzeta, eta = 0.83, 5.59
        t1, t2 = 0, 7.8
        sigma = 29.3
        sigmap, tpk, dtp = [3.5,1.4], [10,12.5], [0.7,0.25]
        return self.sfrd_sj21_multipeak(tp_,tr_,t,gamma,beta,t1,t2,sigma,sigmap,tpk,dtp)
    
    
    def sfrt_sj21(self,t,gamma,beta,t1,t2,sigma,**kwargs):
        """
        SFR of the thick disk as defined in Sysoliatina and Just (2021), Eq. (7)-(8). 
        
        :param t: Galactic time, Gyr. 
        :type t: scalar or array-like  
        :param gamma: SFR power index. 
        :type gamma: scalar
        :param beta: SFR exponential index. 
        :type beta: scalar
        :param t1: Parameter defining the star formation rate at the initial time-point, **t** = 0 Gyr.  
        :type t1: scalar 
        :param t2: Parameter defining the final time-point of the thick-disk formation, Gyr. 
        :type t2: scalar 
        :param sigma: Midplane present-day thick-disk surface density, :math:`\mathrm{M_\odot \ pc^{-2}}`. 
        :type sigma: scalar 
        :param g: Optional, mass loss function (for the time range 0-**t2** Gyr or 0-**tp** Gyr). 
        :type g: scalar or array-like
        
        :return: Absolute and normalized SFR as a function of **t**, 
            :math:`\mathrm{M_\odot \ pc^{-2} \ Gyr^{-1}}`. 
            Normalization factor  is the mean SFR 
            (averaged over the whole time range of the thick-disk evolution). 
        :rtype: list[1d-array] or list[float]  
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
        Thick-disk SFR as defined in Sysoliatina and Just (2021) (:meth:`jjmodel.funcs.SFR.sfrt_sj21`)
        calculated with best parameters (model MCMC1). 
        
        :param t: Galactic time, Gyr. 
        :type t: scalar or array-like            
            
        :return: Absolute and normalized SFR as a function of **t**, 
            :math:`\mathrm{M_\odot \ pc^{-2} \ Gyr^{-1}}`. Normalization factor 
            is the mean SFR (averaged over the whole time range of the thick-disk evolution). 
        :rtype: list[1d-array] or list[float]  
        """
        t1, t2 = 0.1, 4 
        gamma, beta = 3.5, 2 
        sigma = 4.9
        return self.sfrt_sj21(t,gamma,beta,t1,t2,sigma)
    
    
    def sfrr(self,p,a,gd,gd0,gt):
        """
        Thin- and thick-disk SFR as functions of Galactocentric distance. 
        
        :param p: Set of model parameters from the parameter file. 
        :type p: namedtuple
        :param a: Collection of the fixed model parameters, useful quantities, and arrays.
        :type a: namedtuple
        :param gd: Thin-disk mass loss function at a function of ``a.t``, at the different 
            Galactocentric distances ``a.R``.  
        :type gd: array-like
        :param gd0: Local thin-disk mass loss function. 
        :param gt: Thick-disk mass loss function corresponding to ``a.t`` or ``a.t[:a.jt]``. 
                Assumed to be the same for all distances ``a.R``. 
        
        :return: When ``p.pkey=0`` (no extra peaks in the thin-disk SFR), the output has the following structure: 
            *((SFRd,NSFRd,SFRd0,NSFRd0),(SFRt,NSFRt,SFRt0,NSFRt0),(SFRtot,NSFRtot,SFRtot0,NSFRtot0))*. 
            For the thin-disk, thick-disk, and total SFR calculated quantities are absolute and 
            normalized star formation rate at ``a.R`` (array shapes ``(a.Rbins,a.jd)`` or ``(a.Rbins,a.jt)``) 
            and in the Solar neighbourhood. If ``p.pkey=1``, the thin-disk part is 
            *(SFRd,NSFRd,Fp,SFRd0,NSFRd0,Fp0)*, 
            where *Fp* and *Fp0* are the relative contributions of the peaks to the total SFR at each 
            radius and also locally.     
        :rtype: list[list[1d-array]]
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
    Class for defining the initial mass function (IMF).
    """
    
    def __init__(self,mlow,mup):
        """
        Class instance is initialized by two parameters. 
        
        :param mlow: Lower limit of the mass range, :math:`\mathrm{M_\odot}`. 
        :type mlow: scalar 
        :param mup: Upper limit of the mass range, :math:`\mathrm{M_\odot}`. 
        :type mup: scalar 
        """
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
        
    
    def _dndm_probability_(self,mass1,mass2,Nmdm):
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
    
    
    def _number_stars_(self,mass1,mass2):
        
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
        A four-slope broken power-law (BPL) IMF.
        
        :param a0: First IMF slope. 
        :type a0: scalar 
        :param a1: Second IMF slope.  
        :type a1: scalar 
        :param a2: Third IMF slope.   
        :type a2: scalar 
        :param a3: Fourth IMF slope. 
        :type a3: scalar 
        :param m1: First break point (slopes **a0**-**a1**), :math:`\mathrm{M_\odot}`.  
        :type m1: scalar 
        :param m2: Second break point (slopes **a1**-**a2**), :math:`\mathrm{M_\odot}`. 
        :type m2: scalar 
        :param m3: Third break point (slopes **a2**-**a3**), :math:`\mathrm{M_\odot}`. 
        :type m3: scalar         
        
        :return: Linear mass grid from **mlow** to **mup** (in :math:`\mathrm{M_\odot}`) and 
                probabilities corresponding to these mass intervals.
        :rtype: 2d-array
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
        A four-slope broken power-law (BPL) IMF from Rybizki and Just (2015) and Rybizki (2018) 
        (:meth:`jjmodel.funcs.IMF.BPL_4slopes`)
        calculated with the best parameters. 
        
        :return: Linear mass grid from **mlow** to **mup** (in :math:`\mathrm{M_\odot}`) and 
                probabilities corresponding to these mass intervals.
        :rtype: 2d-array
        """  
        
        a0, a1, a2, a3 = 1.26, 1.49, 3.02, 2.28
        m1, m2, m3 = 0.5, 1.39, 6
        return self.BPL_4slopes(a0,a1,a2,a3,m1,m2,m3)


    def BPL_4slopes_sj21_default(self):
        """
        A four-slope broken power law (BPL) IMF from Sysoliatina and Just (2021) 
        (:meth:`jjmodel.funcs.IMF.BPL_4slopes`) 
        calculated with the best parameters (MCMC1 run). 
        
        :return: Linear mass grid from **mlow** to **mup** (in :math:`\mathrm{M_\odot}`) and 
                probabilities corresponding to these mass intervals.
        :rtype: 2d-array
        """  
        
        a0, a1, a2, a3 = 1.31, 1.5, 2.88, 2.28
        m1, m2, m3 = 0.49, 1.43, 6
        return self.BPL_4slopes(a0,a1,a2,a3,m1,m2,m3)


    def Chabrier03(self):
        """
        A lognormal + power-law IMF from Chabrier (2003). 
        
        :return: Linear mass grid from **mlow** to **mup** (in :math:`\mathrm{M_\odot}`) and 
                probabilities corresponding to these mass intervals.
        :rtype: 2d-array
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
        A three-slope broken power law (BPL) IMF from Kroupa et al. (1993).
        
        :return: Linear mass grid from **mlow** to **mup** (in :math:`\mathrm{M_\odot}`) and 
                probabilities corresponding to these mass intervals.
        :rtype: 2d-array
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
    

