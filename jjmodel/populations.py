# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 18:17:18 2017

@author: Skevja
"""
import os 
import inspect
import numpy as np 
from multiprocessing import Pool
from astropy.table import Table
from .funcs import AMR, log_surface_gravity
from .constants import tp, tr
from .control import CheckIsoInput
from .tools import gauss_weights
from . import localpath


class ColumnsIso():
    """
    Collection of methods to work with the columns of Padova, MIST, and BaSTI isochrones.
    """
    
    def column_namespace(self,photometric_system):
        """
        Names of the useful isochrone columns to be extracted from 
        the stellar library (or calculated from them).
        
        :param photometric_system: Name of the photometric system to be used. 
            Valid names are: ``'UBVRIplus'``, ``'GaiaDR2_MAW'``, ``'GaiaEDR3'``, ``'UBVRIplus+GaiaDR2_MAW'``,
            ``'UBVRIplus+GaiaEDR3'``, ``'GaiaDR2_MAW+GaiaEDR3'``, ``'UBVRIplus+GaiaDR2_MAW+GaiaEDR3'``. 
            For MIST isochrones ``'UBVRIplus'`` is UBV(RI)c + 2MASS, for 
            Padova ``'UBVRIplus'`` is UBVRIJHK. For BaSTI, only ``'GaiaEDR3'`` system is available. 
        :type photometric_system: str
         
        :return: Dictionary with column names. 
        
            Keys are:    
                - ``'all'``: list of all columns that will be eventually saved.
                - ``'basic'``: list of columns independent from the chosen photometric system.
                - ``'basic_short'``: same as ``'basic'``, but only columns initially present in the isochrones.
                - ``'phot'``: columns with photometry corresponding to the chosen photometric system). 
                
        :rtype: dict 
        """  
        
        this_function = inspect.stack()[0][3]
        ch = CheckIsoInput()
        photometric_system = ch.check_photometric_system(photometric_system,this_function)
 
        basic_columns = ['Mini','Mf','logL','logT','logg']
        photo_columns = {'GaiaDR2_MAW':['G_DR2','GBPbr_DR2','GBPft_DR2','GRP_DR2'],
                         'GaiaEDR3':['G_EDR3','GBP_EDR3','GRP_EDR3'],
                         'UBVRIplus':['U','B','V','R','I','J','H','K']
                         }          
        photo_names = photometric_system.split('+') 
        
        if len(photo_names)>1:
            all_columns = basic_columns
            for i in range(len(photo_names)):
                all_columns.extend(photo_columns[photo_names[i]])        
        else:
            all_columns = basic_columns + photo_columns[photometric_system]   
       #{'all':all_columns,'basic':basic_columns,'phot':photo_columns[photometric_system]}
        return all_columns
                
        
    def column_positions(self,mode,get,**kwargs):
        """
        Gets position of columns in the isochrone tables.
            
        :param mode: Defines which set of isochrones is used, can be ``'Padova'``, ``'MIST'``, or ``'BaSTI'``. 
        :type mode: str
        :param get: List names of the columns to be extracted from the isochrone tables.
        :type get: list[str]
        :param printnames: Optional. If True, prints all useful columns available in the isochrones. 
        :type printnames: boolean
         
        :return: List of positions of the columns given in parameter **get**. 
        :rtype: list             
        """  

        # After my pre-processing (only potentially useful columns left)
        namespace_padova = {'Mini':0,'Mf':1,'logL':2,'logT':3,'logg':4,
                            'U':5,'B':6,'V':7,'R':8,'I':9,'J':10,'H':11,'K':12,
                            'G_DR2':13,'GBPbr_DR2':14,'GBPft_DR2':15,'GRP_DR2':16,
                            'G_EDR3':17,'GBP_EDR3':18,'GRP_EDR3':19
                            }
        namespace_mist = {'Mini':0,'Mf':1,'logT':2,'logg':3,'logL':4,
                          'U':5,'B':6,'V':7,'R':8,'I':9,'J':10,'H':11,'K':12,
                            'G_DR2':13,'GBPbr_DR2':14,'GBPft_DR2':15,'GRP_DR2':16,
                            'G_EDR3':17,'GBP_EDR3':18,'GRP_EDR3':19
                            }
        namespace_basti = {'Mini':0,'Mf':1,'logL':2,'logT':3,
                           'G_EDR3':4,'GBP_EDR3':5,'GRP_EDR3':6
                            }
        
        if mode=='Padova':
            out = [namespace_padova[i] for i in get]
        if mode=='MIST':
            out = [namespace_mist[i] for i in get]
        if mode=='BaSTI':
            get.remove('logg')
            out = [namespace_basti[i] for i in get]
            
        if 'printnames' in kwargs and kwargs['printnames']==True:
            if mode=='Padova':
                print(namespace_padova.keys())
            if mode=='MIST':
                print(namespace_mist.keys())
                
        return out
        
    
    def read_columns(self,mode,isochrone,columns,indices):
        """
        Extracts columns from the isochrone tables.
            
        :param mode: Defines which set of isochrones is used, can be ``'Padova'``, ``'MIST'``, or ``'BaSTI'``. 
        :type mode: str
        :param isochrone: Isochrone table which has been read from the input directory. 
        :type isochrone: array 
        :param columns: Names of the columns to be extracted from the isochrone.
        :type columns: array-like 
        :param indices: Positions of the columns. 
        :type indices: 1d-array 
            
        :return: Isochrone columns with an age column in units of Gyr. 
        :rtype: dict
        """ 
        
        iso = {}
        for i in range(len(indices)):
            iso[columns[i]] = isochrone[indices[i]]
                        
        return iso
    
    
    def sort_mass_column(self,iso):
        """
        Reshuffles isochrone rows to order the mass column.
            
        :param iso: Isochrone (output of read_columns).
        :type iso: dict           
        
        :return: Isochrone with the reshuffled rows, such that the column ``'Mini'`` (initial mass) is ordered. 
        :rtype: dict             
        """ 
        
        index_sorted = np.array([i[0] for i in sorted(enumerate(np.array(iso['Mini'])),
                                                      key=lambda x:x[1])])
        keys = list(iso.keys())
        for i in range(len(keys)):
            iso[keys[i]] = np.array(iso[keys[i]])[index_sorted]
            
        return iso


    def apply_IMF(self,imf,iso_masses,mass):   
        """
        Applies IMF to the isochrone mass column.
        
        :param imf: IMF PDF function returning the probability to form a 
            star with a mass between *mass1* and *mass2*. 
        :type imf: *function(mass1,mass2)* 
        :param iso_masses: Mass column from the isochrone table. 
        :type iso_masses: array-like
        :param mass: Total mass that was converted into stars of the chosen 
            metallicity and age (this isochrone), :math:`\mathrm{M}_\odot`.
        :type mass: scalar  
        
        :return: New column with the present-day surface number densities of the semi-(metallicity-age-mass)
             'stellar assemblies',  :math:`\mathrm{number \ pc}^{-2}`. 
        :rtype: 1d-array            
        """ 
        
        lenm = len(iso_masses)
        m_centers = np.zeros((lenm+1))
        m_centers[0], m_centers[-1] = iso_masses[0], iso_masses[-1]
        m_centers[1:-1] = [np.mean([iso_masses[i],iso_masses[i+1]]) for i in np.arange(lenm-1)]
    
        num_dens = np.array([imf(m_centers[k],m_centers[k+1])*mass for k in np.arange(lenm)])
        
        return num_dens
    
    

def stellar_assemblies_iso(mode,photometric_system,met,age,mass,imf):
    """
    Creates a table with the semi-(metallicity,age,mass) 'stellar assemblies'.
    
    :param mode: Defines which set of isochrones is used, can be ``'Padova'``, ``'MIST'``, or ``'BaSTI'``. 
    :type mode: str
    :param photometric_system: Photometric system. 
        Name of the photometric system to use, can be: 
            
            - 1 = ``'UBVRIplus'`` (UBVRIJHK - for Padova; UBV(RI)c+2MASS - for MIST), 
            - 2 = ``'GaiaDR2_MAW'``
            - 3 = ``'GaiaEDR3'`` 
            - 4 = ``'UBVRIplus + GaiaDR2_MAW'``
            - 5 = ``'UBVRIplus + GaiaEDR3'``
            - 6 = ``'GaiaDR2_MAW + GaiaEDR3'``
            - 7 = ``'UBVRIplus + GaiaDR2_MAW + GaiaEDR3'``
            
        For BaSTI the only option at the moment is 3.     
    :type photometric_system: int 
    :param met: Metallicity [Fe/H]. 
    :type met: scalar 
    :param age: Age (in case of the disk, age is linked to the metallicity via the AMR).                      
    :type age: scalar
    :param mass: Total mass that was converted into stars of the chosen metallicity and age (isochrone). 
    :type mass: scalar 
    :param imf: IMF PDF function returning the probability to form a star 
        with a mass between *mass1* and *mass2*. 
    :type imf: *function(mass1,mass2)*
    
    :return: Isochrone table for the given metallicity and age, with several additional columns.
    :rtype: dict            
    """ 
    
    if mode=='BaSTI':
        folder_name = 'gaiaedr3'
    else:
        folder_name = 'multiband'
        
    grid_mask = np.loadtxt(os.path.join(localpath,'input','isochrones',mode,folder_name,
                                        ''.join(('grid_mask_',mode,'.txt')))).T
    grid_mask = np.array(grid_mask,dtype=bool)
    # File grid_mask is a boolean mask indicating which isochrone agex are available 
    # for a given metallicity. In fact, only needed with BaSTI isochrones, as for 
    # Padova and MIST all ages in the range of 0-13 Gyr are available for the adopted 
    # metallicity grid. 
    
    met_available_table = np.loadtxt(os.path.join(localpath,'input','isochrones',
                                                  'Metallicity_grid.txt')).T
    met_available = met_available_table[1][grid_mask[0]]
    age_available = np.arange(0.05,13.05,0.05)
    
    index_best_met = np.where(np.abs(np.subtract(met_available,met))==\
                              np.amin(np.abs(np.subtract(met_available,met))))[0][0]
    index_best_met2 = np.where(met_available_table[1]==met_available[index_best_met])[0][0]
    age4met_available = age_available[grid_mask[:,index_best_met2]]
    index_best_age = np.where(np.abs(np.subtract(age4met_available,age))==\
                              np.amin(np.abs(np.subtract(age4met_available,age))))[0][0]                                                         
    
    name = os.path.join(localpath,'input','isochrones',mode,folder_name,
                        ''.join(('iso_fe',str(round(met_available[index_best_met],2)))),
                        ''.join(('iso_age',str(round(age4met_available[index_best_age],2)),'.txt'))) 
    
    isochrone = np.genfromtxt(name).T
    
    cols = ColumnsIso()
    all_columns = cols.column_namespace(photometric_system)
    
    indices = cols.column_positions(mode,all_columns)
    iso = cols.read_columns(mode,isochrone,all_columns,indices)
    
    iso = cols.sort_mass_column(iso)
    iso['N'] = cols.apply_IMF(imf,iso['Mini'],mass)
    iso['age'], iso['FeH'] = [age for i in iso['logT']],[met for i in iso['logT']]
    if mode=='BaSTI':
        iso['logg'] = [log_surface_gravity(mass,10**logL,10**logT) 
                       for (mass,logL,logT) in zip(iso['Mf'],iso['logL'],iso['LogT'])]
    return iso
    
  
def stellar_assemblies_r(R,p,a,amrd,amrt,sfrd,sfrt,sigmash,imf,mode,photometric_system,**kwargs):
    """
    Constructs a list of the semi-(metallicity,age,mass) 
    'stellar assemblies' at some given Galactocentric distance. 
        
    :param R: Galactocentric distance, kpc. 
    :type R: scalar
    :param p: Set of model parameters from the parameter file. 
    :type p: namedtuple
    :param a: Collection of the fixed model parameters, useful quantities, and arrays.
    :type a: namedtuple
    :param amrd: Thin-disk AMR at this **R** 
        (only metallicity column, without the corresponding time ``a.t``).
    :type amrd: array-like 
    :param amrt: Thick-disk AMR (again, only metallicity column).
    :type amrt: array-like 
    :param sfrd: Thin-disk SFR at this **R**, :math:`\mathrm{M_\odot \ pc^{-2} \ Gyr^{-1}}`.
    :type sfrd: array-like 
    :param sfrt: Thick-disk SFR at this **R**, :math:`\mathrm{M_\odot \ pc^{-2} \ Gyr^{-1}}`.  
    :type sfrt: array-like
    :param sigmash: Present-day surface density of the stellar halo at this **R**, :math:`\mathrm{M_\odot \ pc^{-2}}`. 
    :type sigmash: scalar 
    :param imf: IMF PDF function returning the probability to form a star 
        with a mass between *mass1* and *mass2*. 
    :type imf: *function(mass1,mass2)*
    :param mode: Defines which set of isochrones is used, can be ``'Padova'``, ``'MIST'``, or ``'BaSTI'``. 
    :type mode: str
    :param photometric_system: Photometric system to use, can be an integer value from 1 to 7. 
        For the list of available systems see :func:`jjmodel.populations.stellar_assemblies_iso`. 
    :type photometric_system: int
    :param FeH_mean_sh: Optional, mean metallicity of the halo.
    :type FeH_mean_sh: scalar 
    :param FeH_sigma_sh: Optional. Standard deviation of the Gaussian metallicity distribution of the halo. 
    :type FeH_sigma_sh: scalar 
    :param Nmet_sh: Optional. Number of metallicity populations used to represent halo metallicity distribution. 
    :type Nmet_sh: int 
    :param FeH_scatter: Optional. Physical scatter in the thin- and thick-disk AMR, by default there is no scatter.  
    :type FeH_scatter: scalar 
    :param Nmet_dt: Number of metallicity populations used to represent the Gaussian distribution 
        (**FeH_scatter**) around mean metallicities (the thin- and thick-disk AMR). 
    :type Nmet_dt: int 
    
    :return: None. Saves the calculated tables to the output directory defined in the directory tree ``a.T``.         
    """ 
    
    this_function = inspect.stack()[0][3]
    ch = CheckIsoInput()
    ch.check_mode_isochrone(mode,this_function)
    
    print(''.join(('\nStellar population synthesis for R = ', str(R),' kpc:')))
    # By default, the halo metalicity distribution is a Gaussian 
    # with mean at -1.5 and std=0.4 (An, Beers+2013). 
        
    amrsh_spread = np.linspace(p.FeHsh-3*p.dFeHsh,p.FeHsh+3*p.dFeHsh,p.n_FeHsh)
    wsh = gauss_weights(amrsh_spread,p.FeHsh,p.dFeHsh)
    popsh = [np.linspace(0,0,p.n_FeHsh),amrsh_spread,wsh]
    
    if mode=='Padova':
        metmin, metmax = -2.2, 0.5
        # from http://stev.oapd.inaf.it/cgi-bin/cmd
    if mode=='MIST':
        metmin, metmax = -4.0, 0.5
        # from http://waps.cfa.harvard.edu/MIST/interp_isos.html
    if mode=='BaSTI':
        metmin, metmax = -3.2, 0.45
        # from https://iopscience.iop.org/article/10.3847/1538-4357/aab158/pdf
        
    metcheckd = [ch.check_metallicity(met,metmin,metmax,this_function,print_warning=False) 
                 for met in amrd]                                        
    popd = [a.t,amrd,np.linspace(1,1,a.jd)]
    popt = [a.t[:a.jt],amrt[:a.jt],np.linspace(1,1,a.jt)]
    
    metcheckt = [ch.check_metallicity(met,metmin,metmax,this_function,print_warning=False) 
                 for met in amrt[:a.jt]]
    metchecksh = [ch.check_metallicity(met,metmin,metmax,this_function,print_warning=False) 
                  for met in amrsh_spread]
    badd = np.where(np.array(metcheckd)==False)[0]
    badt = np.where(np.array(metcheckt)==False)[0]
    badsh = np.where(np.array(metchecksh)==False)[0]
    if len(badd)!=0 or len(badt)!=0 or len(badsh)!=0:
        key = 0
        components_outside_metrange = ''
        if len(badd)!=0:
            components_outside_metrange += 'thin disk'
            key = 1
        if len(badt)!=0 and R==p.Rsun:
            if components_outside_metrange!='':
                if p.run_mode==0:
                    components_outside_metrange += ', thick disk'
                else:
                    components_outside_metrange += ', thick disk -- and this is valid for all R'
            else:
                if p.run_mode==0:
                    components_outside_metrange += 'thick disk'
                else:    
                    components_outside_metrange += 'thick disk -- and this is valid for all R'
                key = 1 
        if len(badsh)!=0 and R==p.Rsun:
            if components_outside_metrange!='':
                if p.run_mode==0:
                    components_outside_metrange += ', halo'
                else:
                    components_outside_metrange += ', halo -- and this is valid for all R'
            else:
                if p.run_mode==0:
                    components_outside_metrange += 'halo'
                else:
                    components_outside_metrange += 'halo -- and this is valid for all R'
                key = 1 
        if key==1: 
            warning_message = 'Warning. Some of modeled metallicities ('+\
                               components_outside_metrange+') are outside of '+\
                               mode+' metallicity range ['+str(round(metmin,2))+\
                               ','+str(round(metmax,2))+'], \ni.e., the adopted best isochrones'+\
                               ' may be not representative.'
            print(warning_message)
    
    if p.n_FeHdt > 1:
        td, metd, wd = [], [], [] 
        for i in range(a.jd):
            amrd_spread = np.linspace(amrd[i]-3*p.dFeHdt,amrd[i]+3*p.dFeHdt,p.n_FeHdt)
            weights = gauss_weights(amrd_spread,amrd[i],p.dFeHdt)
            metd.extend(amrd_spread)
            wd.extend(weights)
            td.extend([a.t[i] for k in weights])
        popd = [td,metd,wd]
        
        tt, mett, wt = [], [], [] 
        for i in range(a.jt):
            amrt_spread = np.linspace(amrt[i]-3*p.dFeHdt,
                                      amrt[i]+3*p.dFeHdt,p.n_FeHdt
                                      )
            weights = gauss_weights(amrt_spread,amrt[i],p.dFeHdt)
            mett.extend(amrt_spread)
            wt.extend(weights)
            tt.extend([a.t[i] for k in weights])
        popt = [tt,mett,wt]
        
    pop = [popd,popt,popsh]
    labels=['d','t','sh']
        
    for i in range(len(pop)):
        
        age, met = np.subtract(tp,pop[i][0]), pop[i][1] 
        indt = np.array(np.array(pop[i][0])//tr,dtype=np.int)
        
        if i==0:
            print('\tthin disk',end='')
            mass = sfrd[indt]*tr*pop[i][2]
        if i==1:
            print('\tthick disk',end='')
            mass = sfrt[indt]*tr*pop[i][2]
        if i==2:
            print('\thalo')
            g_grid = np.load(os.path.join(localpath,'input','mass_loss','g_grid.npy'))
            fe0,dfe = -2.0, 0.02
            indmet = p.FeHsh//dfe - fe0//dfe + 1
            mass = sigmash/g_grid[int(indmet)][0]*pop[i][2]
        
        jm = len(age)
        argument_list = [(mode,photometric_system,met[k],age[k],mass[k],imf) for k in np.arange(jm)]
        
        # Create output lists
        cols = ColumnsIso()
        columns = cols.column_namespace(photometric_system)
        all_columns = ['N','age','FeH'] + columns + ['disk_label']
        
        ncols = len(all_columns)
        output = [[] for i in range(ncols)]
                
        pool = Pool(processes=p.nprocess)
        result = pool.starmap(stellar_assemblies_iso,argument_list)
        pool.close()
        pool.join()
        
        for k in range(len(result)):
            for m in range(ncols-1):
                output[m].extend(result[k][all_columns[m]])
            output[-1].extend(np.repeat(i,len(result[k]['age'])))
                
        out_tab = Table()
        for k in range(ncols):
            out_tab[all_columns[k]] = output[k]
              
        # Save the table        
        out_tab.write(os.path.join(a.T['poptab'],''.join(('SSP_R',
                                   str(R),'_',labels[i],'_',mode,'.csv'))),overwrite=True) 
                         
                                   
        

