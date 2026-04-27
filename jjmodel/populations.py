# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 18:17:18 2017

@author: Skevja
"""
import os 
import inspect
import numpy as np 
from itertools import repeat
from multiprocessing import Pool
from astropy.table import Table
from .funcs import AMR, log_surface_gravity
from .constants import tp, tr
from .control import CheckIsoInput
from .tools import gauss_weights
from . import localpath
from functools import lru_cache

# Helper functions to speed up the code for WDs

ISOCHRONE_ROOT = os.path.join(localpath, 'input', 'isochrones')
WD_ISOCHRONE_ROOT = os.path.join(ISOCHRONE_ROOT, 'WD_isochrones')
METALLICITY_GRID = np.loadtxt(os.path.join(ISOCHRONE_ROOT, 'Metallicity_grid.txt')).T
AGE_AVAILABLE = np.arange(0.05, 13.05, 0.05)

@lru_cache(maxsize=32)
def _load_isochrone_table(path):
    """
    Reads one numeric isochrone table from disk.

    The isochrone files contain a commented header followed by numeric columns.
    ``np.loadtxt`` is used instead of ``np.genfromtxt`` because the format is simple
    and ``loadtxt`` is faster. The transposed output matches the old code layout.
    """
    return np.loadtxt(path).T

WD_GRID_ROOTS = {
    'basti': 'multiband_basti',
    'lpcode': 'multiband_lpcode',
    'montreal': 'multiband_montreal',
}

WD_GRID_ALIASES = {
    'basti': 'basti',
    'lpcode': 'lpcode',
    'montreal': 'montreal',
}


def _normalize_wd_mode(mode_wd):
    """
    Normalizes the requested white-dwarf isochrone grid name.

    Accepted values are ``'BaSTI'``, ``'LPCODE'``, and ``'Montreal'``.
    The comparison is case-insensitive.
    """
    mode_key = str(mode_wd).strip().lower()
    if mode_key not in WD_GRID_ALIASES:
        raise ValueError(
            "mode_wd must be one of: 'BaSTI', 'LPCODE', or 'Montreal'. "
            f"Got {mode_wd!r}."
        )
    return WD_GRID_ALIASES[mode_key]


def _wd_grid_root(mode_wd):
    """
    Returns the root directory of the selected white-dwarf isochrone grid.

    The WD grids are stored under ``input/isochrones/WD_isochrones``.
    Expected folder names are ``multiband_basti``, ``multiband_lpcode``,
    and ``multiband_montreal``.
    """
    mode_key = _normalize_wd_mode(mode_wd)
    root = os.path.join(WD_ISOCHRONE_ROOT, WD_GRID_ROOTS[mode_key])

    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"WD isochrone grid not found: {root}. "
            f"Extract or rename the {mode_key} WD grid to "
            f"{WD_GRID_ROOTS[mode_key]} under {WD_ISOCHRONE_ROOT}."
        )

    return root



@lru_cache(maxsize=16)
def _metallicity_folders(root):
    """
    Lists available metallicity folders for one WD atmosphere grid.

    The expected folder names are like ``iso_fe-0.2`` or ``iso_fe0.0``.
    The returned tuple contains ``(metallicity, folder_path)`` pairs.
    """
    folders = []

    for name in os.listdir(root):
        if not name.startswith('iso_fe'):
            continue
        try:
            metallicity = float(name.replace('iso_fe', ''))
        except ValueError:
            continue
        folders.append((metallicity, os.path.join(root, name)))

    if not folders:
        raise FileNotFoundError(f"No metallicity folders found in: {root}")

    return tuple(sorted(folders))


def _nearest_metallicity_folder(root, met):
    """
    Finds the closest available WD metallicity folder to the requested metallicity.
    """
    folders = _metallicity_folders(root)
    metallicities = np.array([item[0] for item in folders])
    index_best_met = np.argmin(np.abs(metallicities - met))
    return folders[index_best_met]

def _fast_imf_weights(imf, mass_edges):
    """
    Calculates IMF bin weights using the precomputed IMF arrays.

    This is a faster replacement for repeatedly calling ``imf(m1, m2)`` in a
    Python loop. If the input IMF does not expose the expected internal arrays,
    the function returns ``None`` and the caller should fall back to the old method.
    """
    imf_obj = getattr(imf, "__self__", None)

    required = ("mlow", "mup", "mres", "m_lin", "Nmdm")
    if imf_obj is None or not all(hasattr(imf_obj, attr) for attr in required):
        return None

    mass1 = np.asarray(mass_edges[:-1], dtype=float)
    mass2 = np.asarray(mass_edges[1:], dtype=float)

    mass1 = np.maximum(mass1, imf_obj.mlow)
    mass2 = np.minimum(mass2, imf_obj.mup)

    m1_ind = ((mass1 - imf_obj.mlow) // imf_obj.mres).astype(int)
    m2_ind = ((mass2 - imf_obj.mlow) // imf_obj.mres).astype(int)

    close = (m1_ind == m2_ind) | ((m2_ind - m1_ind) == 1)
    m2_eff = m2_ind.copy()
    m2_eff[close] = m1_ind[close] + 2

    m1_ind = np.clip(m1_ind, 0, len(imf_obj.Nmdm) - 1)
    m2_eff = np.clip(m2_eff, 0, len(imf_obj.Nmdm))

    csum = np.concatenate(([0.0], np.cumsum(imf_obj.Nmdm)))
    weights = csum[m2_eff] - csum[m1_ind]

    interval_weight = np.ones_like(weights)
    interval_weight[close] = (
        (mass2[close] - mass1[close]) /
        (imf_obj.m_lin[m2_eff[close]] - imf_obj.m_lin[m1_ind[close]])
    )

    return weights * interval_weight


class ColumnsIso():
    """
    Collection of methods to work with the columns of Padova, MIST, and BaSTI isochrones.
    """
    
    def column_namespace(self,mode,photometric_system,wd=False):
        r"""
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
        photometric_system = ch.check_photometric_system(mode,photometric_system,this_function)
 
        basic_columns = ['Mini','Mf','logL','logT','logg']
        if wd:
            basic_columns.append('age_WD')

        if mode!='BaSTI':
            basic_columns += ['phase']
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
        r"""
        Gets position of columns in the isochrone tables.
            
        :param mode: Defines which set of isochrones is used, can be ``'Padova'``, ``'MIST'``, or ``'BaSTI'``. 
        :type mode: str
        :param get: List names of the columns to be extracted from the isochrone tables.
        :type get: list[str]
        :param printnames: Optional. If True, prints all useful columns available in the isochrones. 
        :type printnames: boolean
        :param wd: Optional. Indicates whether we work with main sequence + giants isochrones or white dwarfs. 
        :type wd: bool
         
        :return: List of positions of the columns given in parameter **get**. 
        :rtype: list             
        """  

        # After my pre-processing (only potentially useful columns left)
        '''
        # old metallicity grd
        namespace_padova = {'Mini':0,'Mf':1,'logL':2,'logT':3,'logg':4,
                            'U':6,'B':7,'V':8,'R':9,'I':10,'J':11,'H':12,'K':13,
                            'G_DR2':14,'GBPbr_DR2':15,'GBPft_DR2':16,'GRP_DR2':17,
                            'G_EDR3':18,'GBP_EDR3':19,'GRP_EDR3':20,'phase':5
                            }
        '''
        # updated grid
        namespace_padova = {'Mini':0,'Mf':1,'logL':2,'logT':3,'logg':4,
                            'phase':5,'G_EDR3':6,'GBP_EDR3':7,'GRP_EDR3':8
                            }
        '''
        # old metallicity grd
        namespace_mist = {'Mini':0,'Mf':1,'logT':2,'logg':3,'logL':4,
                          'U':5,'B':6,'V':7,'R':8,'I':9,'J':10,'H':11,'K':12,
                          'G_DR2':13,'GBPbr_DR2':14,'GBPft_DR2':15,'GRP_DR2':16,
                          'G_EDR3':17,'GBP_EDR3':18,'GRP_EDR3':19, 'phase':20
                          }
        '''
        namespace_mist = {'Mini':0,'Mf':1,'logT':3,'logg':4,'logL':2,
                          'G_EDR3':7,'GBP_EDR3':8,'GRP_EDR3':9, 'phase':6
                          }
        if 'wd' in kwargs and kwargs['wd']==True:
            '''
            # old grid
            namespace_basti = {'Mini':0,'Mf':1,'logL':3,'logT':2,'logg':4,
                               'U':5,'B':6,'V':7,'R':8,'I':9,'J':10,'H':11,'K':12,
                               'G_EDR3':13,'GBP_EDR3':14,'GRP_EDR3':15,
                               'age_WD':16
                               }
            '''
            # new grid
            namespace_basti = {'Mini':0,'Mf':1,'logL':2,'logT':3,'logg':4,
                               'age_WD':5,'G_EDR3':6,'GBP_EDR3':7,'GRP_EDR3':8
                               }
            
        else:
            namespace_basti = {'Mini':0,'Mf':1,'logL':2,'logT':3,'logg':4,
                               'G_EDR3':5,'GBP_EDR3':6,'GRP_EDR3':7
                               }
        
        if mode=='Padova':
            out = [namespace_padova[i] for i in get]
        if mode=='MIST':
            out = [namespace_mist[i] for i in get]
        if mode=='BaSTI':
            out = [namespace_basti[i] for i in get]
            
        if 'printnames' in kwargs and kwargs['printnames']==True:
            if mode=='Padova':
                print(namespace_padova.keys())
            if mode=='MIST':
                print(namespace_mist.keys())
            if mode=='BaSTI':
                print(namespace_basti.keys())
                
        return out
        
    
    def read_columns(self,mode,isochrone,columns,indices):
        r"""
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
        
        if len(columns) != len(indices):
            raise ValueError('Length of column and index lists must be the same!')

        iso = {}
        for col,idx in zip(columns,indices):
            iso[col] = isochrone[idx]
                        
        return iso
    
    
    def sort_mass_column(self,iso):
        r"""
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
    
    
    def append_iso (self, isoms, isowd):
        """
        Appends the white dwarf isochrone to the main isochrone. 
        
        :param isoms: Main isochrone.
        :type isoms: dict 
        :param isowd: White-dwarf isochrone.
        :type isowd: dict 
        
        :return: Concatenated isochrone. 
        :rtype: dict 
        """
        new ={}
        keys = list(isoms.keys())

        for i in range(len(keys)):
            new[keys[i]] = np.concatenate((isoms[keys[i]],isowd[keys[i]]))

        return new


    def apply_IMF(self, imf, iso_masses, mass):
        r"""
        Applies the IMF to the isochrone mass grid.

        The isochrone mass column is converted into mass-bin edges by placing
        bin boundaries halfway between neighboring initial masses. The returned
        array gives the expected number surface density in each mass bin,
        normalized by the total stellar mass formed in this age-metallicity bin.

        A vectorized fast path is used when ``imf`` is a bound ``IMF.number_stars``
        method with precomputed ``m_lin`` and ``Nmdm`` arrays. Otherwise the code
        falls back to calling ``imf(m1, m2)`` for every mass bin.

        :param imf: IMF function returning the fraction of stars in a mass interval.
        :type imf: callable
        :param iso_masses: Initial stellar masses from the isochrone table.
        :type iso_masses: array-like
        :param mass: Total stellar mass formed in this age-metallicity bin.
        :type mass: scalar

        :return: Number surface density for each isochrone mass bin.
        :rtype: 1d-array
        """
        lenm = len(iso_masses)

        m_centers = np.zeros((lenm + 1))
        m_centers[0], m_centers[-1] = iso_masses[0], iso_masses[-1]
        m_centers[1:-1] = (np.asarray(iso_masses[:-1]) + np.asarray(iso_masses[1:])) / 2

        fast_weights = _fast_imf_weights(imf, m_centers)
        if fast_weights is not None:
            return fast_weights * mass

        return np.array([
            imf(m_centers[k], m_centers[k + 1]) * mass
            for k in np.arange(lenm)
        ])

    
    

def stellar_assemblies_iso(mode,photometric_system,met,age,mass,imf,**kwargs):
    r"""
    Creates a table with the semi-(metallicity,age,mass) 'stellar assemblies'.
    
    :param mode: Defines which set of isochrones is used, can be ``'Padova'``, ``'MIST'``, 
        or ``'BaSTI'``. This only corresponds to the main isochrone 
        (main sequence and giants). 
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
    :param wd: Optional. Prescribes whether white dwarf population should be created. 
        Can be ``'ms+wd'`` (WD and other populations) or ``'wd'`` (WDs only). 
    :type wd: str 
    
    :return: Isochrone table for the given metallicity and age, with several additional columns.
    :rtype: dict            
    """ 
    
    met_available_table = METALLICITY_GRID
    cols = ColumnsIso()
    
    if 'wd' not in kwargs or ('wd' in kwargs and kwargs['wd']=='ms+wd'):

        all_columns_ms = cols.column_namespace(mode,photometric_system)

        if mode=='BaSTI':
            folder_name = os.path.join('MS+','gaiaedr3')
        else:
            folder_name = 'multiband'
        
        
        #grid_mask = np.loadtxt(os.path.join(localpath,'input','isochrones',mode,folder_name,
        #                                    ''.join(('grid_mask_',mode,'.txt')))).T
        #grid_mask = np.array(grid_mask,dtype=bool)
        
        # File grid_mask is a boolean mask indicating what isochrone ages are available 
        # for a given metallicity. In fact, only needed with BaSTI isochrones, as for 
        # Padova and MIST all ages in the range of 0-13 Gyr are available for the adopted 
        # metallicity grid. 
        
        #met_available = met_available_table[1][grid_mask[0]]
        met_available = met_available_table[1]

        # Main isochrone
        # -----------------------------------------------------------
        age_available = AGE_AVAILABLE
        # Find closest metallicity in the grid of available metallicities
        # for this isochrone grid (e.g. Fe/H values of +0.46 and +0.47 dex  
        # from our standard metallicity grid are not available for BaSTI)
        index_best_met = np.argmin(np.abs(met_available - met))

            
        index_best_met2 = np.argmin(
            np.abs(met_available_table[1] - met_available[index_best_met])
        )
        
        # Get available ages for the adopted metallicity
        #age4met_available = age_available[grid_mask[:,index_best_met2]]
        age4met_available = age_available

        # Find closest available age to the modelled one 
        index_best_age = np.argmin(np.abs(age4met_available - age))
                                                      
        
        name = os.path.join(localpath,'input','isochrones',mode,folder_name,
                            ''.join(('iso_fe',str(round(met_available[index_best_met],2)))),
                            ''.join(('iso_age',str(round(age4met_available[index_best_age],2)),'.txt'))) 
    
        isochrone = _load_isochrone_table(name).copy()
        
        indices = cols.column_positions(mode,all_columns_ms)
        iso = cols.read_columns(mode,isochrone,all_columns_ms,indices)
        
        #iso = cols.sort_mass_column(iso) # not needed any more, new isochrone grid has sorted mass column 
        iso['N'] = cols.apply_IMF(imf,iso['Mini'],mass)
        iso['age'], iso['FeH'] = [age for i in iso['logT']],[met for i in iso['logT']]
        if mode=='BaSTI':
            iso['phase'] = np.repeat(1,len(iso['N'])) #MS to AGB are 1

        if 'wd' in kwargs and kwargs['wd']=='ms+wd':
            iso['age_WD'] = [np.nan for _ in np.arange(len(iso['Mini']))]

    
    if 'wd' in kwargs and (kwargs['wd']=='ms+wd' or kwargs['wd']=='wd'):

        all_columns_wd = cols.column_namespace(mode,photometric_system,wd=True)

        mode_wd = _normalize_wd_mode(kwargs.get('mode_wd', 'BaSTI'))
        wd_root = _wd_grid_root(mode_wd)

        dawd_met, dawd_folder = _nearest_metallicity_folder(
            os.path.join(wd_root, 'H'), met
        )
        dbwd_met, dbwd_folder = _nearest_metallicity_folder(
            os.path.join(wd_root, 'He'), met
        )


        
        # DA white-dwarf isochrone
        # -----------------------------------------------------------
        age_available_dawd = AGE_AVAILABLE
        #age_available_dawd = np.hstack((0.080,np.arange(0.100,2.700+0.050,0.050), 
        #                                np.arange(2.900,12.700,0.050)))

        index_best_age_dawd = np.argmin(np.abs(age_available_dawd - age))

        
        # DB white-dwarf isochrone
        # -----------------------------------------------------------
        #age_available_dbwd = age_available_dawd[:111] # only for age < 5.7 Gyr
        age_available_dbwd = AGE_AVAILABLE
        
        index_best_age_dbwd = np.argmin(np.abs(age_available_dbwd - age))


        # Check for metallicities and ages
        '''
        print('modeled Fe/H: ', met, ' modeled age: ',age)
        print('DA WDs')
        print('chosen Fe/H:', round(met_available_table[1][index_best_met],2),
              ' chosen age:', round(age_available_dawd[index_best_age_dawd],2))
        print('DB WDs')
        print('chosen Fe/H:', round(met_available_table[1][index_best_met],2),
              ' chosen age:', round(age_available_dbwd[index_best_age_dbwd],2))
        '''
    
        name_dawd = os.path.join(
            dawd_folder,
            ''.join(('iso_age', str(round(age_available_dawd[index_best_age_dawd], 2)), '.txt'))
        )

        
        name_dbwd = os.path.join(
            dbwd_folder,
            ''.join(('iso_age', str(round(age_available_dbwd[index_best_age_dbwd], 2)), '.txt'))
        )

    
        isochrone_dawd = _load_isochrone_table(name_dawd).copy()
        isochrone_dbwd = _load_isochrone_table(name_dbwd).copy()
        
        if mode!='BaSTI':
            all_columns_wd.remove('phase')

        #for DAWD
        indices_wd = cols.column_positions('BaSTI',all_columns_wd,wd=True)
        
        iso_dawd = cols.read_columns('BaSTI',isochrone_dawd,all_columns_wd,indices_wd)
        
        iso_dawd = cols.sort_mass_column(iso_dawd)
        iso_dawd['N'] = cols.apply_IMF(imf,iso_dawd['Mini'],mass)*0.8
        #iso_dawd['N'] = iso_dawd['N']*(1 - fdb_parabola(10**iso_dawd['logT']/10**3))*0.8 
        iso_dawd['age'], iso_dawd['FeH'] = [age for i in iso_dawd['logT']],[met for i in iso_dawd['logT']]
        iso_dawd['phase'] = np.repeat(10,len(iso_dawd['N'])) # DA WDs are 10 
        
        #for DB WD
        iso_dbwd = cols.read_columns('BaSTI',isochrone_dbwd,all_columns_wd,indices_wd)
        iso_dbwd = cols.sort_mass_column(iso_dbwd)
        iso_dbwd['N'] = cols.apply_IMF(imf,iso_dbwd['Mini'],mass)*0.2
        #iso_dbwd['N'] = iso_dbwd['N'] * (fdb_parabola(10**iso_dbwd['logT']/10**3))*0.2
        iso_dbwd['age'], iso_dbwd['FeH'] = [age for i in iso_dbwd['logT']],[met for i in iso_dbwd['logT']]
        iso_dbwd['phase'] = np.repeat(11,len(iso_dbwd['N'])) # DB WDs are 11
        
        iso_wd = cols.append_iso(iso_dawd,iso_dbwd)

    if 'wd' in kwargs:
        if kwargs['wd']=='ms+wd':
            iso_tot = cols.append_iso(iso,iso_wd)
        if kwargs['wd']=='wd':
            iso_tot = iso_wd
    else:
        iso_tot = iso 
        
    return iso_tot
    
  
def fdb_parabola(Teff):
    #need Teff in 10^3 format only!!!!
    params = np.array([ 1.40000000e-04, -1.15983436e-02,  3.11929944e-01]) #best fit parameters
    def parabola(x, a, b, c): #function to fit
        return a*x**2 + b*x + c
    return parabola(np.array(Teff), *[1.40000000e-04, -1.15983436e-02,  3.11929944e-01]) #returns fraction of He dom atm, so for Da it will be 1-this frac


def _starmap_with_kwargs_(pool, fn, args_iter, kwargs_iter):
    args_for_starmap = zip(repeat(fn), args_iter, kwargs_iter)
    return pool.starmap(_apply_args_and_kwargs_, args_for_starmap)

def _apply_args_and_kwargs_(fn, args, kwargs):
    return fn(*args, **kwargs)


    
def stellar_assemblies_r(R,p,a,amrd,amrt,sfrd,sfrt,sigmash,imf,mode,photometric_system,**kwargs):
    r"""
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
    :param wd: Optional. Prescribes whether white dwarf population should be created. 
        Can be ``'ms+wd'`` (WD and other populations) or ``'wd'`` (WDs only). 
    :type wd: str 
    
    :return: None. Saves the calculated tables to the output directory defined in the directory tree ``a.T``.         
    """ 
    
    this_function = inspect.stack()[0][3]
    ch = CheckIsoInput()
    ch.check_mode_isochrone(mode,this_function)
    
    print(''.join(('\nStellar population synthesis for R = ', str(R),' kpc:')))
    # By default, the halo metalicity distribution is a Gaussian 
    # with mean at -1.5 and std=0.4 (An, Beers+2013). 
    
    if 'wd' in kwargs and kwargs['wd']=='wd' and mode!='BaSTI':
        print('Warning. Note that currently the only WD isochrone set is BaSTI, \
              and you chose mode =',mode,'. Changed mode to BaSTI.')
        mode = 'BaSTI'
        
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
                               ','+str(round(metmax,2))+'], i.e., the adopted best isochrones'+\
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
    labels = ['d','t','sh']
        
    for i in range(len(pop)):
        
        age, met = np.subtract(tp,pop[i][0]), pop[i][1] 
        indt = np.array(np.array(pop[i][0])//tr,dtype=int)
        
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
        kwargs_list = repeat(kwargs)            
                  
        pool = Pool(processes=p.nprocess)
        #result = pool.starmap(stellar_assemblies_iso,stellar_assemblies_iso)
        result = _starmap_with_kwargs_(pool,stellar_assemblies_iso,argument_list,kwargs_list)
        pool.close()
        pool.join()
        
        # Create output lists
        iso_columns = list(result[0].keys())

        all_columns = list(dict.fromkeys(['N', 'age', 'FeH'] + iso_columns))
        if mode == 'BaSTI' and 'phase' not in all_columns:
            all_columns.append('phase')
        all_columns.append('disk_label')


        ncols = len(all_columns)
        output = [[] for i in range(ncols)]

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
                         
                                   
        

