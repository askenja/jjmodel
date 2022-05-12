"""
Created on Fri Jul 19 16:41:07 2019
@author: Skevja

Routine to solve Poisson-Boltzmann eq. 
"""

import os
import numpy as np
from scipy.integrate import cumtrapz
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt 
from scipy.ndimage import zoom
from scipy.signal import savgol_filter
from .constants import G, M_SUN, PC, KM, tp, tr, ZN, SIGMA_E, RHO_D0 
from .funcs import AVR
from .iof import tab_sorter, tab_reader
from .tools import ConvertAxes


def z_potential(x,a,b,c,d,e,f):
    """
    6-order polynom to fit the vertical gravitational potential. 
    There is no free coefficient to ensure that :math:`\\phi = 0` in the Galactic plane. 
    
    :param x: Height above the Galactic plane, pc or kpc. 
    :type x: scalar or array-like 
    :param a,b,c,d,e,f: Polynom coefficients.
    :type a,b,c,d,e,f: scalar
        
    :return: Potential at height **x**.
    :rtype: float or array-like
    """
    return a*x + b*x**2 + c*x**3 + d*x**4 + e*x**5 +f*x**6 #+ g*x**7


def vertical_force(a,fimax,Sigma,sigW,h):
    """
    Calculates the vertical force produced by some Galactic component. 
    
    :param a: Collection of the fixed model parameters, useful quantities, and arrays.
    :type a: namedtuple
    :param fimax: The optimal maximum value of the normalized gravitational potential 
        up to which the Poisson-Boltzmann eq. is solved (approximately corresponds 
        to the maximum height ``p.zmax`` prescribed in the parameter file). 
    :type fimax: scalar 
    :param Sigma: Surface density, :math:`\mathrm{M_\odot \ pc^{-2}}` 
        (for the thin and thick disk can be a function of time, *SFRd*gd* and *SFRt*gt*, 
        where *gd* and *gt* are mass loss functions).
    :type Sigma: scalar or array-like
    :param sigW: W-velocity dispersion, :math:`\mathrm{km \ s^{-1}}` (for the thin disk this is AVR).
    :type sigW: scalar or array-like
    :param h: Scale height, pc (for the thin disk this is a function of time).
    :type h: scalar or array-like
    
    :return: Vertical force (up to ``p.zmax``, corresponds to the grid ``a.z``), 
        :math:`\mathrm{km^2 \ s^{-2} \ kpc^{-1}}`. 
    :rtype: 1d-array        
    """
    
    # Grid along the potential axis. 
    fi1 = 1e-6
    dfi = 0.05 
    fist = int(fimax/dfi) 
    fieq = np.logspace(np.log10(fi1),np.log10(fimax),fist,base=10)
    fieq1 = np.concatenate(([fi1],fieq),axis=0)
    
    # Intergration of the Poisson-Boltzman eq. 
    rho0 = Sigma/2/h
    Ar1 = rho0*(sigW/SIGMA_E)**2/RHO_D0
    Ar2 = (SIGMA_E/sigW)**2
    dz_offset = np.sqrt(SIGMA_E**2*fi1*KM**2*PC/(2*np.pi*G*np.sum(rho0)*M_SUN))/ZN
    integrator = lambda x: 1/np.sqrt(np.sum(Ar1*(1-np.exp(-Ar2*x))))
    dzi = np.add(dz_offset,cumtrapz([integrator(k) for k in fieq1],fieq1))  
    
    # i-th component of the potential and the corresponding vertical force. 
    dzeq1 = np.concatenate((a.dzeq,[a.ddz]),axis=0)
    #popt,pcov = curve_fit(z_potential,dzi,fieq) 
    #fii = z_potential(dzeq1,*popt)
    interpolation_tool = ConvertAxes()
    fii = interpolation_tool.interpolate(dzeq1,zoom(dzi,4),zoom(fieq,4))
    Kzi = np.diff(fii)*SIGMA_E**2/(np.diff(dzeq1*ZN)/KM)   
    
    return Kzi



def _fimax_optimal_(a,SFRd,SFRt,gd,gt,Sigma,sigW,hg,**kwargs):
    """
    Estimates reasonable maximum value of the normalized potential,
    which is needed to optimize the computation time of the 
    Poisson-Boltzmann eq. solver. 

    Parameters
    ----------
    a : namedtuple
        Collection of the fixed model parameters, useful 
        quantities and arrays.
    SFRd : array_like
        Thin-disk star formation rate function, in Msun/pc^2/Gyr. 
        Array length is equal to the number of thin-disk 
        subpopulations: a.jd = int((tp-p.td1)/tr).
    SFRt : array_like
        Thick-disk star formation rate function in Msun/pc^2/Gyr. 
        Length of the array is a.jt = int(p.tt2/tr). 
    gd : array_like
        Thin-disk mass loss function, len(gd)==len(SFRd).
    gt : array_like
        Thin-disk mass loss function, len(gd)==len(SFRd).
    Sigma : array_like
        Present-day surface densities of the (molecular gas, 
        atomic gas, DM, stellar halo). All values are in Msun/pc^2. 
    sigW : array_like
        Set of parameters to define W-velocity dispersions of the 
        Galactic components: (sige, alpha, sigt, sigdh, sigsh).
        sige and alpha are the scaling parameter (in km/s) and 
        power index (dimensionless) of the power-law age-velocity 
        relation of the thin disk. sigt, sigdh, sigsh (all in km/s)
        are W-velocity dispersions of the thick disk, DM and 
        stellar halo, respectively.  
    hg : array_like
        Scale heights of the molecular and atomic gas (hg1, hg2), 
        in pc.
    **kwargs : dict, optional keyword arguments 
        log : text file
            When given, the details of the iteration procedure 
            to calculate the optimal potential value are written 
            to the file. 

    Returns
    -------
    fimax : float
        The optimal maximum value of the normalized gravitational 
        potential, up to which the Poisson-Boltzmann eq. will be 
        solved (approximately corresponds to the prescribed in 
        parameterfile maximum height zmax).
    dfi : float
        Reasonable step in normalized potential to be used when 
        solving the Poisson-Boltzmann eq. 
    """
    
    # ----------------------------------------------------------------------------------------------
    # Initialization of fimax optimization procedure. fist_min is the minimum number of grid points 
    # to be used along the potential axis. Initial values of fimax and dfi are assumed. 
    # Then fist is number of the potential grid points as follows from the assumed fimax and dfi. 
    # zdif (in pc) is the difference between the maximum height z as follows from the current 
    # and previous iterations of the PB eq. solution. epsz (in pc) is the desired accuracy of zdif, 
    # i.e. it defines the accuracy of the fimax optimization. Initially, zdif > epsz.  
    # hd, ht, hdh, hsh (all in pc) are guesses for the scale heights of the thin and thick disk, 
    # DM and stellar halo. sigg1 and sigg2 (in km/s) are the assumed W-velocity dispersions of the 
    # molecular and atomic gas, respectively. These quantities are, of course, dependent 
    # on the Galactocentric distance R, but for the rough estimation of fimax these fixed values
    # are good enough. 
    # ----------------------------------------------------------------------------------------------

    sigmag1, sigmag2, sigmadh, sigmash = Sigma
    sige, alpha, sigt, sigdh, sigsh = sigW
    hg1, hg2 = hg  
    
    fi1 = 1e-6
    fist_min = 30 
    fimax = 8 
    dfi = 0.25     
    fist = int(round(fimax/dfi,0))
    zdif = 1 
    epsz = 0.2 
    hd = np.linspace(700,50,a.jd)
    ht, hdh, hsh = 900, 1800, 2000
    sigg1, sigg2 = 4, 10 
    
    if 'log' in kwargs:
        kwargs['log'].append(''.join(('{:<8}'.format('fimax'),
                                      '{:<8}'.format('len(fi)'),'\n',
                                      '{:<8}'.format(round(fimax,2)),
                                      '{:<8}'.format(fist))))
                                      
    while (zdif > epsz) or (zdif < 0):
             
        fist = int(round(fimax/dfi,0)) 
        if fist < fist_min:
            dfi = fimax/fist_min
            fist = int(round(fimax/dfi,0))
        
        # Note: the log space is used for the potential grid in order to trace the ~quadratic 
        # potential shape near the plane.
        fieq = np.logspace(np.log10(fi1),np.log10(fimax),fist,base=10) 
        
        # Create AVR of the thin disk.
        tau0 = tp/((sigg1/sige)**(-1/alpha)-1)
        age_velocity = AVR()
        avr = age_velocity.avr_jj10(a.t,tp,sige,tau0,alpha)
        
        # Calculate dz = function(fi), i.e., normalized height = function(normalized potential). 
        
        Ard1 = SFRd*gd*tr*np.divide(avr,SIGMA_E)**2/(2*hd*RHO_D0)           # Thin disk
        Ard2 = (SIGMA_E/avr)**2
                  
        Art1 = SFRt*gt*tr*(sigt/SIGMA_E)**2/(2*ht*RHO_D0)                   # Thick disk  
        Art2 = (SIGMA_E/sigt)**2
        
        Ardh1 = sigmadh/2/hdh*(sigdh/SIGMA_E)**2/RHO_D0                     # DM halo
        Ardh2 = (SIGMA_E/sigdh)**2
        
        Arsh1 = sigmash/2/hsh*(sigsh/SIGMA_E)**2/RHO_D0                     # Stellar halo
        Arsh2 = (SIGMA_E/sigsh)**2
        
        Arg11 = sigmag1/2/hg1*(sigg1/SIGMA_E)**2/RHO_D0                     # Molecular gas
        Arg12 = (SIGMA_E/sigg1)**2
        
        Arg21 = sigmag2/2/hg2*(sigg2/SIGMA_E)**2/RHO_D0                     # Atomic gas 
        Arg22 = (SIGMA_E/sigg2)**2
        
        rho0 = np.sum(SFRd*gd*tr/2/hd) + np.sum(SFRt*gt*tr/2/ht) + \
                    sigmash/2/hsh + sigmadh/2/hdh + sigmag1/2/hg1 + sigmag2/2/hg2
        dz_offset = np.sqrt(SIGMA_E**2*fi1*KM**2*PC/(2*np.pi*G*rho0*M_SUN))/ZN
        
        sArd1,sArt1 = np.sum(Ard1),np.sum(Art1)
        sum_term = sArd1 + sArt1 + Ardh1 + Arsh1 + Arg11 + Arg21
        
        def integrator(x): 
            denominator = (sum_term - np.sum(Ard1*np.exp(-np.multiply(Ard2,x)))
                                    - sArt1*np.exp(-np.multiply(Art2,x))
                                    - Ardh1*np.exp(-Ardh2*x) -Arsh1*np.exp(-Arsh2*x)
                                    - Arg11*np.exp(-Arg12*x) -Arg21*np.exp(-Arg22*x)
                           )
            return 1/np.sqrt(denominator)
            
        fieq1 = np.concatenate(([fi1],fieq),axis=0)
        dz = np.add(dz_offset,cumtrapz([integrator(k) for k in fieq1],fieq1))            
        
        # We are interested in max(dz) = dz[-1].
        # The new value of fimax is be chosen according to relative difference 
        # (dz[-1] - a.dzmax)/a.dzmax. After iteration, fimax is multiplied by 1.2, 
        # which is an empirical factor added to ensure that fimax is not underestimated.   
        
        dzmaxcalc = dz[-1]
        zdif = (dzmaxcalc-a.dzmax)/a.dzmax
        fimax_old = fimax
        fimax = (1.1+0.1*np.random.rand())*fimax*a.dzmax/dzmaxcalc 
        fimax = np.mean([fimax_old,fimax])
        
        if 'log' in kwargs:
            kwargs['log'].append(''.join(('\n','{:<8}'.format(round(fimax,2)),'{:<8}'.format(fist))))
    
    return (1.2*fimax, dfi)



def poisson_solver(a,fimax,dfi,SFRd,SFRt,gd,gt,Sigma,sigW,hg,**kwargs):
    """
    Solver of the Poisson-Boltzmann equation. 
    
    :param a: Collection of the fixed model parameters, useful quantities, and arrays.
    :type a: namedtuple
    :param fimax: The optimal maximum value of the normalized gravitational potential 
        up to which the Poisson-Boltzmann eq. is solved (approximately corresponds 
        to the maximum height ``p.zmax`` prescribed in the parameter file). 
    :type fimax: scalar 
    :param dfi: Step in normalized potential. 
    :type dfi: float
    :param SFRd: Thin-disk SFR function, :math:`\mathrm{M_\odot \ pc^{-2} \ Gyr^{-1}}`. 
        Array length is equal to the number of thin-disk subpopulations: ``a.jd = int((tp-p.td1)/tr)``, 
        where ``tp`` is a present-day MW disk age and ``tr`` is the model age resolution. 
    :type SFRd: array-like
    :param SFRt: Thick-disk SFR function, :math:`\mathrm{M_\odot \ pc^{-2} \ Gyr^{-1}}`. 
        Length of the array is ``a.jt = int(p.tt2/tr)``. 
    :type SFRt: array-like
    :param gd: Thin-disk mass loss function, ``len(gd)==len(SFRd)``.
    :type gd: array-like
    :param gt: Thick-disk mass loss function, ``len(gt)==len(SFRt)``.
    :type gt: array-like
    :param Sigma: Present-day surface densities of non-disk components 
        (molecular gas, atomic gas, DM, stellar halo), :math:`\mathrm{M_\odot \ pc^{-2}}`.
    :type Sigma: array-like
    :param sigW: Set of parameters defining W-velocity dispersions of the Galactic components: 
        (*sige, alpha, sigt, sigdh, sigsh*). *sige* and *alpha* are the AVR scaling parameter (:math:`\mathrm{km \ s^{-1}}`) 
        and power index (dim). *sigt*, *sigdh*, *sigsh* (:math:`\mathrm{km \ s^{-1}}`) are W-velocity dispersions 
        of the thick disk, DM, and stellar halo, respectively.
    :type sigW: array-like
    :param hg: Scale heights of the molecular and atomic gas (*hg1*, *hg2*), pc.
    :type hg: array-like
    :param fp: Optional. Relative contributions of the additional thin-disk SFR peaks to the total thin-disk SFR, 
        :math:`\mathrm{M_\odot \ pc^{-2} \ Gyr^{-1}}` (output *Fp* in :meth:`jjmodel.funcs.SFR.sfrr`). 
        Must be given if ``p.pkey=1`` or ``p.pkey=2``. 
    :type fp: array-like
    :param sigp: Optional. W-velocity dispersions (:math:`\mathrm{km \ s^{-1}}`) of the thin-disk populations associated with 
        the stellar density excess in the additional peaks. Must be given when ``p.pkey=1``. 
    :type sigp: array-like
    :param heffd: Optional. Thin-disk half-thickness (effective scale height), pc. If fixed by this parameter, 
        additional iterations will be performed to adapt AVR to fulfill this requirement. 
    :type heffd: scalar 
    :param hefft: Optional. Thick-disk half-thickness (effective scale height), pc. If fixed by this parameter, 
        additional iterations will be performed to adapt thick-disk W-velocity dispersion *sigt* 
        to fulfill this requirement. 
    :type hefft: scalar
    :param status_equation: Optional. If True, the iteration details are printed to console.
    :type status_equation: boolean
    :param log: Optional. If given, the details of the iteration are written to the file.
    :type log: file
    :param plot: Optional. If True, the derived potential is plotted for each iteration.  
    :type plot: boolean
        
    :return: Dictionary with all sorts of output. 
    
        Keys of the standard output:
            
        - ``'hd'``: 1d-array of length ``a.jd``, scale heights of the thin-disk subpopulations (pc).             
        - ``'ht'``, ``'hdh'``, ``'hsh'`` : float, thick-disk, DM, and halo scale heights (pc).             
        - ``'heffd'``, ``'hefft'`` : float, half-thickness of the thin and thick disk (pc).             
        - ``'sigg1'``, ``'sigg2'``, ``'sigt'`` : molecular and atomic gas and thick-disk W-velocity dispersions (:math:`\mathrm{km \ s^{-1}}`). 
        - ``'sige'`` : float, scaling parameter of the thin-disk AVR (:math:`\mathrm{km \ s^{-1}}`).             
        - ``'avr'`` : 1d-array of length ``a.jd``, thin-disk AVR (:math:`\mathrm{km \ s^{-1}}`). 
        - ``'fie'``, ``'phi'`` : total vertical gravitational potential (corresponds to ``a.z`` grid). ``fie`` is the normalized potential multiplied by the constant ``SIGMA_E^2`` (:math:`\mathrm{km^2 \ s^{-2}}`), useful for further calculations of potential-dependend quantities. ``phi`` is the potential in physical units, :math:`\mathrm{m^2 \ s^{-2}}`. 
        - ``'rhodtot'``, ``'rhot'``, ``'rhog1'``, ``'rhog2'``, ``'rhodh'``, ``'rhosh'`` : Mass density vertical profiles of the Galactic components (correspond to ``a.z`` grid), :math:`\mathrm{M_\odot \ pc^{-3}}`. ``'rhodtot'``' is the total thin-disk density, that includes subpopulations characterized by W-velocity dispersion prescribed by the AVR and SFR-peaks' subpopulations with special kinematics, if any.
            
        Keys of the optional output (depending on *kwargs*):
            
        - ``'hdp'`` : Scale height(s) of the SFR-peak(s)' subpopulations, pc. 
        - ``'rhodp'``, ``'rhod0'`` : Mass density vertical profiles of the SFR-peak(s)' subpopulations, and of the thin-disk subpopulations with the vertical kinematics described by the AVR, :math:`\mathrm{M_\odot \ pc^{-3}}`. In this case total density profile is ``rhodtot = rhod0 + sum(rhodp,axis=0)``. 
        - ``'log'`` : file, log file with the iteration details.  
        
    :rtype: dict 
        
    """
    # ----------------------------------------------------------------------------------------------
    # Initialization of the procedure. sigg1 and sigg2 (in km/s) are the assumed W-velocity 
    # dispersions of the molecular and atomic gas components, respectively. All other Galactic 
    # components are initialized with their scale heights: hd, ht, hdh, hsh (pc). 
    # fist is a number of grid points along the normalized potential axis, and fieq is the log-grid
    # in potential. epsh (pc) is the desired accuracy in the model components' scale heights. 
    # ----------------------------------------------------------------------------------------------

    sigmag1, sigmag2, sigmadh, sigmash = Sigma
    sige, alpha, sigt, sigdh, sigsh = sigW
    hg1, hg2 = hg  
    
    sigg1, sigg2 = 3, 10    
    hd = np.linspace(700,50,a.jd)
    ht, hdh, hsh = 1000, 1800, 2000
    epsh = 1 
    
    fi1 = 1e-6
    fist = int(fimax/dfi) 
    fieq = np.logspace(np.log10(fi1),np.log10(fimax),fist,base=10)
    
    if 'sigp' in kwargs:
        from .input_ import p, inp
        npeak = len(kwargs['sigp'])
        hdpdif = epsh+1
        hdp = np.linspace(200,200,npeak)
        fp0 = 1 - np.sum(kwargs['fp'],axis=0)
        indt_sigp = [np.where(np.abs(time-a.t)==np.amin(np.abs(time-a.t)))[0][0] for time in p.tpk]

    
    if 'status_equation' in kwargs:
        print(''.join(('\n','{:<8}'.format('heffd'),'{:<8}'.format('hefft'),
                           '{:<8}'.format('hg1'),'{:<8}'.format('hg2'),
                           '{:<8}'.format('hdh'),'{:<8}'.format('hsh'))))
    if 'log' in kwargs:
        kwargs['log'].append('\n\nheffd, hefft = thin- and thick-disk half-thickness\
                              \nhg1, hg2 = molecular and atomic gas scale height\
                              \nhdh, hsh = DM and stellar halo scale heights\
                              \nunits = [pc]')
        kwargs['log'].append(''.join(('\n\n','{:<8}'.format('heffd'),
                           '{:<8}'.format('hefft'),
                           '{:<8}'.format('hg1'),'{:<8}'.format('hg2'),
                           '{:<8}'.format('hdh'),'{:<8}'.format('hsh'))))

    if 'plot' in kwargs:
        plt.figure()
                    
    count = 0           
    hddif, hg1dif, hg2dif, htdif, hdhdif, hshdif = epsh+1, epsh+1, epsh+1, epsh+1, epsh+1, epsh+1
    heffddif, hefftdif = epsh-1, epsh-1
    if 'heffd' in kwargs:
        heffddif = epsh+1
    if 'hefft' in kwargs:
        hefftdif = epsh+1
        
    while ((hddif > epsh) or (hg1dif > epsh) or (hg2dif > epsh)
           or (htdif > epsh) or (hdhdif > epsh) or (hshdif > epsh)
           or (heffddif > epsh) or (hefftdif > epsh)):
        
        tau0 = tp/((sigg1/sige)**(-1/alpha)-1)
        age_velocity = AVR()
        avr = age_velocity.avr_jj10(a.t,tp,sige,tau0,alpha)
        if 'sigp' in kwargs:
            sigp = np.sqrt(inp['sigwpeak_excess']**2 + avr[indt_sigp]**2)
        
        if 'sigp' not in kwargs:
            Ard1 = SFRd*gd*tr*(avr/SIGMA_E)**2/(2*hd*RHO_D0)                  # Thin disk
            Ard2 = (SIGMA_E/avr)**2
        else:
            Ard1 = SFRd*fp0*gd*tr*(avr/SIGMA_E)**2/(2*hd*RHO_D0)              # Thin disk
            Ard2 = (SIGMA_E/avr)**2
            
            Ardp1 = [SFRd*i*gd*tr*(k/SIGMA_E)**2/(2*l*RHO_D0)                 # Thin-disk peaks 
                     for i,k,l in zip(kwargs['fp'],sigp,hdp)]                 # with special 
            Ardp2 = (np.divide(SIGMA_E,sigp))**2                              # kinematics
            
        Art1 = SFRt*gt*tr*(sigt/SIGMA_E)**2/(2*ht*RHO_D0)                     # Thick disk  
        Art2 = (SIGMA_E/sigt)**2
        
        Ardh1 = sigmadh/2/hdh*(sigdh/SIGMA_E)**2/RHO_D0                       # DM halo
        Ardh2 = (SIGMA_E/sigdh)**2
        
        Arsh1 = sigmash/2/hsh*(sigsh/SIGMA_E)**2/RHO_D0                       # Stellar halo
        Arsh2 = (SIGMA_E/sigsh)**2
        
        Arg11 = sigmag1/2/hg1*(sigg1/SIGMA_E)**2/RHO_D0                       # Molecular gas
        Arg12 = (SIGMA_E/sigg1)**2
        
        Arg21 = sigmag2/2/hg2*(sigg2/SIGMA_E)**2/RHO_D0                       # Atomic gas 
        Arg22 = (SIGMA_E/sigg2)**2
        
        if 'sigp' not in kwargs:
            rho0 = (np.sum(SFRd*gd*tr/2/hd) + np.sum(SFRt*gt*tr/2/ht)
                    + sigmash/2/hsh + sigmadh/2/hdh + sigmag1/2/hg1 + sigmag2/2/hg2
                    )
            sArd1, sArt1 = np.sum(Ard1), np.sum(Art1)
            sum_term = sArd1 + sArt1 + Ardh1 + Arsh1 + Arg11 + Arg21
            
            def integrator(x): 
                denominator = (sum_term -np.sum(Ard1*np.exp(-np.multiply(Ard2,x)))
                                        -sArt1*np.exp(-np.multiply(Art2,x))
                                        -Ardh1*np.exp(-Ardh2*x) -Arsh1*np.exp(-Arsh2*x)
                                        -Arg11*np.exp(-Arg12*x) -Arg21*np.exp(-Arg22*x)
                               )
                return 1/np.sqrt(denominator)
        
        else:
            rho0 = (np.sum(SFRd*fp0*gd*tr/2/hd)
                    + np.sum([np.sum(SFRd*i*gd*tr/2/k) for i,k in zip(kwargs['fp'],hdp)])
                    + np.sum(SFRt*gt*tr/2/ht) + sigmash/2/hsh + sigmadh/2/hdh + sigmag1/2/hg1 
                    + sigmag2/2/hg2
                    )
            sArd1, sArdp1, sArt1 = np.sum(Ard1),np.sum([np.sum(i) for i in Ardp1]),np.sum(Art1)
            sum_term = sArd1 + sArdp1 + sArt1 + Ardh1 + Arsh1 + Arg11 + Arg21
            
            def integrator(x): 
                denominator = (sum_term 
                               -np.sum(Ard1*np.exp(-np.multiply(Ard2,x))) 
                               -np.sum([np.sum(i*np.exp(-np.multiply(k,x))) for i,k in zip(Ardp1,Ardp2)])
                               -sArt1*np.exp(-np.multiply(Art2,x))
                               -Ardh1*np.exp(-Ardh2*x) -Arsh1*np.exp(-Arsh2*x)
                               -Arg11*np.exp(-Arg12*x) -Arg21*np.exp(-Arg22*x)
                               )
                return 1/np.sqrt(denominator)
            
        dz_offset = np.sqrt(SIGMA_E**2*fi1*KM**2*PC/(2*np.pi*G*rho0*M_SUN))/ZN
        fieq1 = np.concatenate(([fi1],fieq),axis=0)
        dz = np.add(dz_offset,cumtrapz([integrator(k) for k in fieq1],fieq1))  
        interpolation_tool = ConvertAxes()
        fi = interpolation_tool.interpolate(a.dzeq,dz,fieq)
        fie = fi*SIGMA_E**2
        
        if 'plot' in kwargs:
            plt.plot(dz*ZN,fieq,marker='o',markersize=4,
                     label='$\mathrm{initial \ grid, \ d \phi=const}$')
            plt.plot(a.z,fi,ls='--',c='k',
                     label='$\mathrm{secondary \ grid, \ d z=const, \ iter=}$'+str(count))
            plt.axis([0,round(a.dzmax*ZN,0),0,np.round(fimax,1)])
            plt.xlabel('$\mathrm{|z|, \ pc}$')
            plt.ylabel(r'$\mathrm{\phi / \sigma^2}$')
            plt.legend(loc=2)              
        
        popt,pcov = curve_fit(z_potential,a.z,fie,p0=(1e-2,1e-3,1e-6,1e-9,1e-12,1e-16)) #,1e-19
        fie_smooth = z_potential(a.z,*popt)
        phi_smooth = fie_smooth*KM**2
    
        # New scale heights and differences between their old and new values.
        hdnew = np.array([np.trapz(np.exp(-fie_smooth/avr[i]**2),x=a.z)     
                          for i in a.jd_array]                          # Thin disk
                         )
        difd = np.abs(hdnew - hd)
        hd = hdnew
        hddif = np.amax(difd)   
        if 'sigp' in kwargs:
            hdpnew = np.array([np.trapz(np.exp(-fie_smooth/i**2),x=a.z) 
                               for i in sigp]                           # Thin-disk SFR peaks
                              )                                         # with special kinematics
            difdp = np.abs(hdpnew - hdp)
            hdp = hdpnew
            hdpdif = np.amax(difdp)     
            hddif = np.amax([hddif,hdpdif])  
                  
        htnew = np.trapz(np.exp(-fie_smooth/sigt**2),x=a.z)             # Thick disk
        htdif = np.abs(htnew - ht)
        ht = htnew        
        
        hg1new = np.trapz(np.exp(-fie_smooth/sigg1**2),x=a.z)           # Molecular gas
        hg1dif = np.abs(hg1new - hg1)
        sigg1_old = sigg1
        sigg1 = sigg1_old*hg1/hg1new
        sigg1 = np.mean([sigg1_old,sigg1])        
        
        hg2new = np.trapz(np.exp(-fie_smooth/sigg2**2),x=a.z)           # Atomic gas
        hg2dif = np.abs(hg2new - hg2)
        sigg2_old = sigg2
        sigg2 = sigg2_old*hg2/hg2new
        sigg2 = np.mean([sigg2_old,sigg2])        
            
        hdhnew = np.trapz(np.exp(-fie_smooth/sigdh**2),x=a.z)           # DM halo
        hdhdif = np.abs(hdhnew - hdh)
        hdh = hdhnew        
        
        hshnew = np.trapz(np.exp(-fie_smooth/sigsh**2),x=a.z)           # Stellar halo
        hshdif = np.abs(hshnew - hsh)
        hsh = hshnew
                  
        if 'heffd' in kwargs:
            if 'sigp' not in kwargs:
                rhodtot0 = np.sum(SFRd*gd*tr/2/hd)
            else:
                rhod0 = np.sum(SFRd*fp0*gd*tr/2/hd)
                rhodp = [np.sum(SFRd*i*gd*tr/2/k) for i,k in zip(kwargs['fp'],hdp)]
                rhodtot0 = rhod0 + np.sum(rhodp)
            heffd_new = np.sum(SFRd*gd*tr)/2/rhodtot0 
            heffddif = abs(kwargs['heffd'] - heffd_new) 
            #hddif = np.mean([hddif,heffddif])
            
            sige_old = sige                             
            sige = kwargs['heffd']/heffd_new*sige_old
            sige = np.mean([sige,sige_old])
            
        if 'hefft' in kwargs:
            rhot0 = np.sum(SFRt*gt*tr/2/ht)
            hefft_new = np.sum(SFRt*gt*tr)/2/rhot0 
            
            hefftdif = abs(kwargs['hefft'] - hefft_new) 
            #htdif = np.mean([htdif,hefftdif])
            
            sigt_old = sigt
            sigt = kwargs['hefft']/hefft_new*sigt     
            sigt = np.mean([sigt_old,sigt])
        
        # The updated thin- and thick-disk half-thickness values (pc) 
        # and in-plane densities (Msun/pc^3). 
        if 'sigp' not in kwargs:
            rhodtot0 = np.sum(SFRd*gd*tr/2/hd)
        else:
            rhod0 = np.sum(SFRd*fp0*gd*tr/2/hd)
            rhodp = [np.sum(SFRd*i*gd*tr/2/k) for i,k in zip(kwargs['fp'],hdp)]
            rhodtot0 = rhod0 + np.sum(rhodp)
        heffd = np.sum(SFRd*gd*tr)/2/rhodtot0  
        rhot0 = np.sum(SFRt*gt*tr/2/ht)
        hefft = np.sum(SFRt*gt*tr)/2/rhot0 
        
        if 'status_equation' in kwargs:
            print(''.join(('{:<8}'.format(round(heffd,1)),'{:<8}'.format(round(hefft,1)),
                       '{:<8}'.format(round(hg1new,1)),'{:<8}'.format(round(hg2new,1)),
                       '{:<8}'.format(round(hdhnew,1)),'{:<8}'.format(round(hshnew,1)))))
        if 'log' in kwargs:
            kwargs['log'].append(''.join(('\n','{:<8}'.format(round(heffd,1)),
                   '{:<8}'.format(round(hefft,1)),
                   '{:<8}'.format(round(hg1new,1)),'{:<8}'.format(round(hg2new,1)),
                   '{:<8}'.format(round(hdhnew,1)),'{:<8}'.format(round(hshnew,1)))))
    
        count = count + 1  
           
    # Output
    # ---------------------------------------------------------------------------------------------
    if 'sigp' in kwargs:   
        rhodp = [[np.sum(SFRd*i*gd*tr/2/k*np.exp(-pot/l**2)) for pot in fie_smooth] 
                 for i,k,l in zip(kwargs['fp'],hdp,sigp)
                 ]
        rhod0 = [np.sum(SFRd*fp0*gd*tr/2/hd*np.exp(-i/avr**2)) for i in fie_smooth]
        rhodtot = rhod0 + np.sum(rhodp,axis=0)
    else:
        rhodtot = [np.sum(SFRd*gd*tr/2/hd*np.exp(-i/avr**2)) for i in fie_smooth]
    rhot = [np.sum(SFRt*gt*tr/2/ht*np.exp(-i/sigt**2)) for i in fie_smooth]
    rhog1 = [sigmag1/2/hg1*np.exp(-i/sigg1**2) for i in fie_smooth]
    rhog2 = [sigmag2/2/hg2*np.exp(-i/sigg2**2) for i in fie_smooth]
    rhodh = [sigmadh/2/hdh*np.exp(-i/sigdh**2) for i in fie_smooth]
    rhosh = [sigmash/2/hsh*np.exp(-i/sigsh**2) for i in fie_smooth]
    
    if 'plot' in kwargs:
        plt.savefig(os.path.join(a.dir,'fi_iteration.png'))
        plt.close()

    out = {'hd':hd,'ht':ht,'hdh':hdh,'hsh':hsh,'heffd':heffd,'hefft':hefft,
           'sigg1':sigg1,'sigg2':sigg2,'sigt':sigt,'sige':sige,'avr':avr,
           'fie':fie_smooth,'phi':phi_smooth,
           'rhodtot':rhodtot,'rhot':rhot,'rhog1':rhog1,'rhog2':rhog2,'rhodh':rhodh,'rhosh':rhosh,
           }
    if 'sigp' in kwargs:  
        out['hdp'], out['rhodp'], out['rhod0'], out['sigp'] = hdp, rhodp, rhod0, sigp
    if 'log' in kwargs:
        out['log'] = kwargs['log']
    
    return out



