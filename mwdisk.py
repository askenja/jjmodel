"""
Created on Wed May  9 18:21:36 2018
@author: Skevja
"""

import os
import sys
import numpy as np 
from itertools import repeat
from multiprocessing import Pool
from .funcs import heffr
from .iof import dir_tree, TabSaver
from .tools import Timer, LogFiles
from .poisson import fimax_optimal, poisson_solver, vertical_force
from .constants import tr, SIGMA_E


def rbin_builder(R,a,SFRd,SFRt,gd,gt,Sigma,sigW,hg,**kwargs):
    """
    Predicts the vertical structure of the MW disk at the given 
    Galactocentric distance R.  
    
    Parameters
    ----------
    R : scalar
        Galactocentric distance in kpc. 
    p : namedtuple
        Set of the model parameters from the parameter file. 
    a : namedtuple
        Collection of the fixed model parameters, useful quantities 
        and arrays.
    SFRd : array_like
        Thin-disk star formation rate function, Msun/pc^2/Gyr. 
        Array length is equal to the number of thin-disk sub-
        populations: a.jd = int((tp-p.td1)/tr).
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
        are  W-velocity dispersions of the thick disk, DM and 
        stellar halo, respectively.  
    hg : array_like
        Scale heights of the molecular and atomic gas (hg1, hg2), 
        in pc.
    **kwargs : dict, optional keyword arguments
        status_equation : boolean
            If True, the details of solving the Poisson-Boltzmann 
            eq. are printed to the console.
        status_progress : boolean
            If True, the overall progress details are printed to 
            console.
        log : file
            When given, the details of the iteration are written 
            to the file. 
        plot: boolean
            If True, the derived potential is plotted for each 
            iteration.  
        fp : array_like
            Amplitude-related parameters for all additional thin-
            disk SFR peaks. len(fp)==npeak. Must be specified 
            together with npeak and sigp.
        sigp : array_like
            W-velocity dispersions (km/s) of the thin-disk 
            populations associated with the stellar density excess 
            forming additional peaks. len(sigp)==npeak. Must be 
            given together with npeak and fp. 
        heffd : scalar
            Thin-disk half-thickness (effective scale height), pc. 
            If fixed by this parameter, additional iterations will 
            be performed to adapt AVR to fulfill this requirement. 
        hefft : scalar
            Thick-disk half-thickness (effective scale height), pc. 
            If fixed by this parameter, additional iterations will 
            be performed to adapt sigt to fulfill this requirement. 
        save: boolean
            If True, the output tables (and plot with converging 
            potential in plot==True), is saved to the specified 
            directory, a.dir. 
            
    Returns
    -------
    out : dict
        Keys of the standard output:
        'hd': 1d-array of length a.jd, scale heights of the thin-
            disk subpopulations (pc). 
        'ht', 'hdh', 'hsh' : float, thick-disk DM and halo scale 
            heights (pc). 
        'heffd', 'hefft' : float, half-thickness of the thin and 
            thick disk (pc). 
        'sigg1','sigg2','sigt' : molecular and atomic gas and 
            thick-disk W-velocity dispersions (km/s). 
        'sige' : float, scaling parameter of the thin-disk AVR 
            (km/s). 
        'avr' : 1d-array of length a.jd, thin-disk AVR (km/s). 
        'fie', 'phi' : gravitational potential as a function of z 
            (corresponds to a.z grid). fie is the normalized  
            potential multiplied by SIGMA_E^2 (in km/s), useful 
            for the further calculations of the potential-dependend 
            quantites. phi is the potential in physical units, 
            m^2/s^2. 
        'rhodtot','rhot','rhog1','rhog2','rhodh','rhosh' : Matter 
            density vertical profiles of the Galactic components 
            (correspond to a.z grid), in Msun/pc^3. rhodtot is 
            the total thin-disk density, that includes sub-
            populations characterized by W-velocity dispersion 
            prescribed by the AVR and SFR-peaks' subpopulations 
            with special kinematics, if any.
        'Kzdtot','Kzt','Kzg1','Kzg2','Kzdh','Kzsh' : Vertical 
            gravitational force from i-th model component, in 
            m^2/s^2/pc. Kzdtot corresponds to the total thin disk, 
            as rhodtot. 
        Keys of the optional output (depends on kwargs):
        'hdp' : Scale height(s) of the SFR-peak(s)' subpopulations, 
            in pc. 
        'rhodp','rhod0' : Matter density vertical profiles of the 
            SFR-peak(s)' subpopulations, and of the thin-disk sub-
            populations with the vertical kinematics described by 
            the AVR, in Msun/pc^3. In this case rhodtot == rhod0 + 
            sum(rhodp,axis=0). 
        'Kzd0', 'Kzdp' : Analogically to rhodp and rhod0, thin-
            disk vertical graditational force components, in 
            m^2/s^2/pc.     
    """

    timer = Timer()
    t_start = timer.start()
    
    if 'status_progress' in kwargs and kwargs['status_progress']==True:
        sys.stdout.write(''.join(('\n','{:<2}'.format(''),
                                       '{:<14}'.format('Process for R = %s kpc' %(R)), 
                                       '{:<14}'.format(': start')))
                         )        
    
    # Firstly, fimax value is optimized.  
    if 'log' in kwargs:
        fimax, dfi = fimax_optimal(a,SFRd,SFRt,gd,gt,Sigma,sigW,hg,log=kwargs['log'])
    else:
        fimax,dfi = fimax_optimal(a,SFRd,SFRt,gd,gt,Sigma,sigW,hg)
    
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
    Builds the local JJ model based on the given parameters and 
    input functions.
    
    Parameters
    ----------
    p : namedtuple
        Set of the model parameters from the parameter file. 
    a : namedtuple
        Collection of the fixed model parameters, useful quantities 
        and arrays.
    inp : dict
        Collection of the input functions including SFR, AVR, AMR 
        and IMF.

    Returns
    -------
    out : dict
        Output of the function rbin_builder (for R == R_SUN).
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
    Calculates the predictions of JJ-model at a given Galactocentric 
    distance. 
        
    Parameters
    ----------
    inp : dict
        Collection of the input functions including SFR, AVR, AMR 
        and IMF.
    i : int
        Index of the R-bin.
    out_local : dict
        Output of the function local_run.
    **kwargs : dict, optional keyword arguments
        status_progress : boolean
            If True, the overall progress details are printed 
            to the console.
            
    Returns
    -------
    out : dict
        Output of the function rbin_builder (for this i-th R-bin).
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


def starmap_with_kwargs(pool, fn, args_iter, kwargs_iter):
    args_for_starmap = zip(repeat(fn), args_iter, kwargs_iter)
    return pool.starmap(apply_args_and_kwargs, args_for_starmap)

def apply_args_and_kwargs(fn, args, kwargs):
    return fn(*args, **kwargs)


def disk_builder(p,a,inp,**kwargs):
    """
    Constructs the disk (vertical structure at some 
    Galactocentric distance R). Can work in two modes (depending 
    on the parameter run_mode):
        (1) models the Solar neighbourhood only 
        (2) starts with the Solar neighbourhood and then extends
        the local JJ model to other R. 
    Input data (e.g. SFR and AMR) and results (e.g. potential, 
    densities, scale heights, AVR) are saved as txt files 
    to the output directory. 
    
    Parameters
    ----------
    p : namedtuple
        Set of the model parameters from the parameter file. 
    a : namedtuple
        Collection of the fixed model parameters, useful quantities 
        and arrays.
    inp : dict
        Collection of the input functions including SFR, AVR, AMR 
        and IMF.
    **kwargs : dict, optional keyword arguments
        status_progress : boolean
            If True, the overall progress details are printed to 
            console during the main run.

    Returns
    -------
    None.
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
        result = starmap_with_kwargs(pool,extended_run,args_iter,kwargs_iter)
        pool.close()
        pool.join()
        print('\n')
        tabsaver.input_extended_save(inp)
        tabsaver.output_extended_save(result)
    
        print('\n---Global run ended sucessfully---\n\n',timer.stop(t_run))
    
