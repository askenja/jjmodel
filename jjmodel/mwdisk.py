"""
Created on Wed May  9 18:21:36 2018
@author: Skevja
"""

import os
import sys
import numpy as np 
import matplotlib.pyplot as plt
from itertools import repeat
from multiprocessing import Pool
from .funcs import heffr
from .iof import dir_tree, TabSaver
from .tools import Timer, LogFiles
from .poisson import _fimax_optimal_, poisson_solver, vertical_force
from .constants import tr, SIGMA_E


def rbin_builder(R,a,SFRd,SFRt,gd,gt,Sigma,sigW,hg,**kwargs):
    """
    Predicts the vertical structure of the MW disk at a given Galactocentric distance.  
    
    :param R: Galactocentric distance, kpc. 
    :type R: scalar
    :param a: Collection of the fixed model parameters, useful quantities, and arrays.
    :type a: namedtuple
    :param SFRd: Thin-disk star formation rate function, :math:`\mathrm{M_\odot \ pc^{-2} \ Gyr^{-1}}`. 
        Array length is equal to the number of thin-disk subpopulations: ``a.jd = int((tp-p.td1)/tr)``,
        where ``tp`` is a present-day MW disk age and ``tr`` is the model age resolution. 
    :type SFRd: array-like
    :param SFRt: Thick-disk star formation rate function, :math:`\mathrm{M_\odot \ pc^{-2} \ Gyr^{-1}}`. 
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
    :param status_progress: Optional. If True, the overall progress details are printed to console.
    :type status_progress: boolean
    :param log: Optional. If given, the details of the iteration are written to the file.
    :type log: file
    :param plot: Optional. If True, the derived potential is plotted for each iteration, plots are saved.  
    :type plot: boolean
    :param save: Optional. If True, the output tables, are saved to the specified directory, ``a.dir``. 
    :type save: boolean
            
    :return: Dictionary with all sorts of output (mainly the output of :func:`jjmodel.poisson.poisson_solver`).  
    
        Keys of the standard output:
            
        - ``'hd'``: 1d-array of length ``a.jd``, scale heights of the thin-disk subpopulations (pc).                   
        - ``'ht'``, ``'hdh'``, ``'hsh'`` : float, thick-disk, DM, and halo scale heights (pc).          
        - ``'heffd'``, ``'hefft'`` : float, half-thickness of the thin and thick disk (pc).             
        - ``'sigg1'``, ``'sigg2'``, ``'sigt'`` : molecular and atomic gas and thick-disk W-velocity dispersions (:math:`\mathrm{km \ s^{-1}}`). 
        - ``'sige'`` : float, scaling parameter of the thin-disk AVR (:math:`\mathrm{km \ s^{-1}}`).               
        - ``'avr'`` : 1d-array of length ``a.jd``, thin-disk AVR (:math:`\mathrm{km \ s^{-1}}`). 
        - ``'fie'``, ``'phi'`` : total vertical gravitational potential (corresponds to ``a.z`` grid). ``fie`` is the normalized potential multiplied by the constant ``SIGMA_E^2`` (:math:`\mathrm{km^2 \ s^{-2}}`), useful for further calculations of potential-dependend quantities. ``phi`` is the potential in physical units, :math:`\mathrm{m^2 \ s^{-2}}`. 
        - ``'rhodtot'``, ``'rhot'``, ``'rhog1'``, ``'rhog2'``, ``'rhodh'``, ``'rhosh'`` : Mass density vertical profiles of the Galactic components (correspond to ``a.z`` grid), :math:`\mathrm{M_\odot \ pc^{-3}}`. ``'rhodtot'``' is the total thin-disk density, that includes subpopulations characterized by W-velocity dispersion prescribed by the AVR and SFR-peaks' subpopulations with special kinematics, if any.
        - ``'Kzdtot'``, ``'Kzt'``, ``'Kzg1'``, ``'Kzg2'``, ``'Kzdh'``, ``'Kzsh'`` : Vertical force from *i*-th model component, :math:`\mathrm{m^2 \ s^{-2} \ pc^{-1}}`. ``Kzdtot`` corresponds to the total thin disk, as ``rhodtot``. 
        
        Keys of the optional output (depending on *kwargs*):
            
        - ``'hdp'`` : Scale height(s) of the SFR-peak(s)' subpopulations, pc. 
        - ``'rhodp'``, ``'rhod0'`` : Mass density vertical profiles of the SFR-peak(s)' subpopulations, and of the thin-disk subpopulations with the vertical kinematics described by the AVR, :math:`\mathrm{M_\odot \ pc^{-3}}`. In this case total density profile is ``rhodtot = rhod0 + sum(rhodp,axis=0)``. 
        - ``'Kzd0'``, ``'Kzdp'`` : Analogically to ``rhodp`` and ``rhod0``, thin-disk vertical graditational force components, :math:`\mathrm{m^2 \ s^{-2} \ pc^{-1}}`.   
        - ``'plot'`` : matplotlib figure and axis for the plot of normalized potential. 
    :rtype: dict 
    """

    timer = Timer()
    t_start = timer.start()
    
    if 'plot' in kwargs and kwargs['plot']==True:
        fig, ax = plt.subplots(figsize=(16,10))
        figname = os.path.join(a.T['fiplt'],'fieq_iter_live_R'+str(R)+'.png')
        del kwargs['plot']
        kwargs['plot'] = [(fig,ax),figname]
    
    if 'status_progress' in kwargs and kwargs['status_progress']==True:
        sys.stdout.write(''.join(('\n','{:<2}'.format(''),
                                       '{:<14}'.format('Process for R = %s kpc' %(R)), 
                                       '{:<14}'.format(': start')))
                         )        
    
    # Firstly, fimax value is optimized.  
    if 'log' in kwargs:
        fimax, dfi = _fimax_optimal_(a,SFRd,SFRt,gd,gt,Sigma,sigW,hg,log=kwargs['log'])
    else:
        fimax,dfi = _fimax_optimal_(a,SFRd,SFRt,gd,gt,Sigma,sigW,hg)
    
    if 'status_progress' in kwargs and kwargs['status_progress']==True:
        sys.stdout.write(''.join(('\n','{:<4}'.format(''),
                                       '{:<14}'.format('Process for R = %s kpc' %(R)),
                                       '{:<14}'.format(': fimax optimized')))
                         )
    
    # Secondly, the Poisson-Boltzmann eq. is solved.     
    S = poisson_solver(a,fimax,dfi,SFRd,SFRt,gd,gt,Sigma,sigW,hg,**kwargs)
    
    if 'status_progress' in kwargs and kwargs['status_progress']==True:                                 
        sys.stdout.write(''.join(('\n','{:<6}'.format(''),
                                       '{:<14}'.format('Process for R = %s kpc' %(R)),
                                       '{:<14}'.format(': PE solved')))
                         )
    
    # Finally, we prepare the output. 
    out = S
    if 'sigp' not in kwargs:
        out['Kzdtot'] = vertical_force(a,fimax,SFRd*gd*tr,S['avr'],S['hd'])
    else:
        fp0 = 1 - np.sum(kwargs['fp'],axis=0)
        out['Kzd0'] = vertical_force(a,fimax,SFRd*fp0*gd*tr,S['avr'],S['hd'])
        out['Kzdp'] = [vertical_force(a,fimax,SFRd*i*gd*tr,k,l)
                       for i,k,l in zip(kwargs['fp'],kwargs['sigp'],S['hdp'])]
        out['Kzdtot'] = out['Kzd0'] + np.sum(out['Kzdp'],axis=0)
    out['Kzt'] = vertical_force(a,fimax,SFRt*gt*tr,S['sigt'],S['ht'])
    out['Kzg1'] = vertical_force(a,fimax,Sigma[0],S['sigg1'],hg[0])
    out['Kzg2'] = vertical_force(a,fimax,Sigma[1],S['sigg2'],hg[1])
    out['Kzdh'] = vertical_force(a,fimax,Sigma[2],sigW[3],S['hdh'])
    out['Kzsh'] = vertical_force(a,fimax,Sigma[3],sigW[4],S['hsh'])
               
    t_exit = timer.stop(t_start)
    sys.stdout.write(''.join(('\n','{:<8}'.format(''),
                                   '{:<14}'.format('Process %s' %(R)),
                                   '{:<14}'.format(''.join((': exit, time: ', t_exit)))))
                     )
    
    if 'log' in kwargs:
        kwargs['log'].append('\n\nDensity in the Galactic plane [Msun/pc^3]:\n')
        kwargs['log'].append(''.join(('{:<11}'.format('Thin.d'),
                                      '{:<11}'.format('Thick.d'),
                                      '{:<11}'.format('Mol.gas'),
                                      '{:<11}'.format('At.gas'),
                                      '{:<11}'.format('DM.halo'),
                                      '{:<11}'.format('St.halo'),
                                      '{:<11}'.format('Total'),'\n'))
                             )
        kwargs['log'].append(''.join(('{:<11}'.format(round(S['rhodtot'][0],4)),
                                      '{:<11}'.format(round(S['rhot'][0],4)),
                                      '{:<11}'.format(round(S['rhog1'][0],4)),
                                      '{:<11}'.format(round(S['rhog2'][0],4)),
                                      '{:<11}'.format(round(S['rhodh'][0],4)),
                                      '{:<11}'.format(round(S['rhosh'][0],5)),
                                      '{:<11}'.format(round(S['rhodtot'][0] + S['rhot'][0]
                                                          + S['rhog1'][0] + S['rhog2'][0]
                                                          + S['rhodh'][0] + S['rhosh'][0],3)),'\n'))
                             )
        kwargs['log'].append(''.join(('\nTime: ',t_exit)))
    
    return out



def local_run(p,a,inp,**kwargs):
    """
    Builds the local JJ model based on the given parameters and input functions.
    Accepts optional parameters of :func:`jjmodel.mwdisk.rbin_builder`. 
    
    :param p: Set of model parameters from the parameter file. 
    :type p: namedtuple
    :param a: Collection of the fixed model parameters, useful quantities, and arrays.
    :type a: namedtuple
    :param inp: Collection of the input functions including SFR, AVR, AMR, and IMF.
    :type inp: dict      

    :return: Output of the function :func:`jjmodel.mwdisk.rbin_builder` for ``p.Rsun``.
    :rtype: dict
    """
    
    f = LogFiles(os.path.join(a.T['stat'],''.join(('log_R',str(p.Rsun),'kpc.txt'))))
    f.write(''.join(('Galactocentric distance = ', str(p.Rsun),' kpc\n\n --- fimax optimization ---\n')))
    
    if p.pkey==1:
        kwargs['sigp'] = p.sigp
        kwargs['fp'] = inp['Fp0']
    out = rbin_builder(p.Rsun,a,inp['SFRd0'],inp['SFRt0'],inp['gd0'],inp['gt'],
                       [p.sigmag1,p.sigmag2,p.sigmadh,p.sigmash],
                       [p.sige,p.alpha,p.sigt,p.sigdh,p.sigsh],[inp['hg10'],inp['hg20']],
                       log=f,**kwargs)                      
    return out


def extended_run(inp,i,out_local,**kwargs):
    """
    Calculates the JJ-model predictions at a given Galactocentric distance. 
    Accepts optional parameters of :func:`jjmodel.mwdisk.rbin_builder`. 
        
    :param inp: Collection of the input functions including SFR, AVR, AMR, and IMF.
    :type inp: dict   
    :param i: Index of the current radial bin in ``a.R`` array.
    :type i: int
    :param out_local: Output of the function :func:`jjmodel.mwdisk.local_run`.
    :type out_local: dict
    :param status_progress: Optional. If True, the overall progress details are printed to console.
    :type status_progress: boolean
            
    :return: Output of the function Accepts optional parameters of :func:`jjmodel.mwdisk.rbin_builder` 
        for the *i*-th radial bin.
    :rtype: dict        
    """
    
    from .input_ import p, a
        
    f = LogFiles(os.path.join(a.T['stat'],''.join(('log_R',str(a.R[i]),'kpc.txt'))))
    f.write(''.join(('Galactocentric distance = ', str(a.R[i]),' kpc\n\n --- fimax optimization ---\n')))
    
    # Initial guess for sige and sigt at this Galactocentric distance. 
    # Locally, sigt is by construction the value from parameter file. 
    sige = SIGMA_E + 3.125*(p.Rsun - a.R[i])
    sigt = p.sigt*(p.Rsun/a.R[i])**0.4 
    
    # From the local model we extract the thin-disk half-thickness and use it to define it as 
    # a function of R, depending on the model parameters (without or with thin-disk flaring). 
    hdeff = heffr(p,a,out_local['heffd'])
    
    if p.pkey==1:
        kwargs['sigp'] = p.sigp
        kwargs['fp'] = inp['Fp'][i]
    
    kwargs['hefft'] = out_local['hefft']

    out = rbin_builder(a.R[i],a,inp['SFRd'][i],inp['SFRt'][i],inp['gd'][i],inp['gt'],
                       [inp['SigmaR'][1,i],inp['SigmaR'][2,i],inp['SigmaR'][4,i],inp['SigmaR'][5,i]],
                       [sige,p.alpha,sigt,p.sigdh,p.sigsh],
                       [inp['hg1'][i],inp['hg2'][i]],
                       heffd=hdeff[i],log=f,**kwargs)
    return out


def _starmap_with_kwargs_(pool, fn, args_iter, kwargs_iter):
    args_for_starmap = zip(repeat(fn), args_iter, kwargs_iter)
    return pool.starmap(_apply_args_and_kwargs_, args_for_starmap)

def _apply_args_and_kwargs_(fn, args, kwargs):
    return fn(*args, **kwargs)


def disk_builder(p,a,inp,**kwargs):
    """
    Constructs the disk (vertical structure at some Galactocentric distance). 
    Can work in two modes (depending on the parameter ``p.run_mode``):
        
        1. models the Solar neighbourhood only;
        2. starts with the Solar neighbourhood and then extends the local JJ model to other radii. 
        
    Input data (e.g. SFR and AMR) and results (e.g. potential, densities, scale heights, AVR) 
    are saved as txt files to the output directory ``a.dir``. 
    Accepts optional parameters of :func:`jjmodel.mwdisk.rbin_builder`. 
    
    :param p: Set of model parameters from the parameter file. 
    :type p: namedtuple
    :param a: Collection of the fixed model parameters, useful quantities, and arrays.
    :type a: namedtuple
    :param inp: Collection of the input functions including SFR, AVR, AMR, and IMF.
    :type inp: dict     
    :param status_progress: Optional. If True, the overall progress details are printed to console.
    :type status_progress: boolean

    :return: None. 
    """
    
    timer = Timer()
    t_run = timer.start()
    
    dir_tree(p,make=True) 
    
    # First, the code must be run for R == Rsun in order to find the half-thickness 
    # of the thin and thick disk. 
    print('\n---Local run---')
    out_local = local_run(p,a,inp,**kwargs)
    print('\n')
    tabsaver = TabSaver(p,a)
    tabsaver.input_local_save(inp)
    tabsaver.output_local_save(out_local)
    print('\n---Local run ended sucessfully---\n')
    
    # Then all other distances are processed. 
    if p.run_mode==1:
        print('\n---Global run---')
        
        args_iter = [(inp,i,out_local) for i in a.R_array]
        kwargs_iter = repeat(kwargs)
        
        pool = Pool(processes=p.nprocess)
        result = _starmap_with_kwargs_(pool,extended_run,args_iter,kwargs_iter)
        pool.close()
        pool.join()
        print('\n')
        tabsaver.input_extended_save(inp)
        tabsaver.output_extended_save(result)
    
        print('\n---Global run ended sucessfully---\n\n',timer.stop(t_run))
    
