"""
Created on Fri Jun 26 15:48:28 2020

@author: skevja
"""

import os
import shutil as sh
import numpy as np 
import matplotlib.pyplot as plt 
from astropy.table import Table
from . import localpath
from .tools import resave_parameters


def dir_tree(p,**kwargs):
    """
    Builds an output directory tree (name suffix of the main output directory 
    can be specified in the parameter file).
     
    :param p: Set of model parameters from the parameter file. 
    :type p: namedtuple
    :param make: Optional. If True, the directory tree is created, 
        otherwise only all the folder names are defined. 
    :type make: boolean
    
    :return: Directory tree for the model output.
    :rtype: dict
    """
    
    if p.run_mode==0:
        Rdirmain = os.path.join('output',''.join(('Rsun',str(p.Rsun),'_',str(p.out_dir))))
    else:
        Rname = ''.join((str(p.Rmin),'R',str(p.Rmax)))
        Rdirmain = os.path.join('output',''.join((Rname,'_dR',str(p.dR),'_',str(p.out_dir))))
    
    T = {}
    T['main'] = Rdirmain
    
    T['dens'] = os.path.join(T['main'],'dens')
    T['denstab'],T['densplt'] = os.path.join(T['dens'],'tab'),os.path.join(T['dens'],'plt')
    T['maps'] = os.path.join(T['main'],'maps')
    T['mapstab'],T['mapsplt'] = os.path.join(T['maps'],'tab'),os.path.join(T['maps'],'plt')
    T['hess'] = os.path.join(T['main'],'hess')
    T['hesstab'],T['hessplt'] = os.path.join(T['hess'],'tab'),os.path.join(T['hess'],'plt')
    T['fi'] = os.path.join(T['main'],'fi')
    T['fitab'],T['fiplt'] = os.path.join(T['fi'],'tab'),os.path.join(T['fi'],'plt')
    T['age'] = os.path.join(T['main'],'age')
    T['agetab'],T['ageplt'] = os.path.join(T['age'],'tab'),os.path.join(T['age'],'plt')
    T['inp'] = os.path.join(T['main'],'inp')
    T['inptab'],T['inpplt'] = os.path.join(T['inp'],'tab'),os.path.join(T['inp'],'plt')
    T['height'] = os.path.join(T['main'],'height')
    T['heighttab'],T['heightplt'] = os.path.join(T['height'],'tab'),os.path.join(T['height'],'plt')
    T['kin'] = os.path.join(T['main'],'kinem')
    T['kintab'],T['kinplt'] = os.path.join(T['kin'],'tab'),os.path.join(T['kin'],'plt')
    T['met'] = os.path.join(T['main'],'met')
    T['mettab'],T['metplt'] = os.path.join(T['met'],'tab'),os.path.join(T['met'],'plt')
    T['stat'] = os.path.join(T['main'],'stat')
    T['pop'] = os.path.join(T['main'],'pop')
    T['poptab'] = os.path.join(T['pop'],'tab')
    T['popplt'] = os.path.join(T['pop'],'plt')
    T['poprctab'] = os.path.join(T['poptab'],'rc')
    T['popcephtab'] = os.path.join(T['poptab'],'ceph')
    T['popatab'] = os.path.join(T['poptab'],'a')
    T['popftab'] = os.path.join(T['poptab'],'f')
    T['popgdwtab'] = os.path.join(T['poptab'],'gdw')
    T['popkdwtab'] = os.path.join(T['poptab'],'kdw')
    
    dirs_level1, dirs_level2, dirs_level3 = [], [], [] 
    keys_list = list(T.keys())
    for i in range(len(keys_list)):
        if ('tab' not in T[keys_list[i]]) and ('plt' not in T[keys_list[i]]):
            dirs_level1.append(T[keys_list[i]])
        else:
            if 'pop' not in T[keys_list[i]]:
                dirs_level2.append(T[keys_list[i]])
            else:
                dirs_level3.append(T[keys_list[i]])
                
    if 'make' in kwargs and kwargs['make']==True:
        if p.out_mode==1:
            if os.path.isdir(T['main'])==True:
                print('Results of this run will be saved to already existing folder',T['main'])
            else:
                for i in dirs_level1: 
                    os.makedirs(i)
                for i in dirs_level2: 
                    os.mkdir(i)
                for i in dirs_level3: 
                    os.mkdir(i)
                print('Results of this run will be saved to',T['main'])
        else:
            if os.path.isdir(T['main'])==True:
                sh.rmtree(T['main'])
            print('\nResults of this run will be saved to ',T['main'],
                  '\nWarning: Old results will be overwritten (out_mode=0 in parameter file, '
                    +'change to 1 for a unique name of output directory).\n') 
            for i in dirs_level1: 
                os.makedirs(i)
            for i in dirs_level2: 
                os.mkdir(i)
            for i in dirs_level3: 
                os.mkdir(i)   
        
        if p.pkey==0:
            resave_parameters(os.path.join('.','parameters'),os.path.join(T['stat'],'parameters'),p)
        else:
            resave_parameters([os.path.join('.','parameters'),
                               os.path.join('.','sfrd_peaks_parameters')],
                              [os.path.join(T['stat'],'parameters'),
                               os.path.join(T['stat'],'sfrd_peaks_parameters')],p)
            
        # sh.copyfile(os.path.join('.','parameters'),os.path.join(T['stat'],'parameters'))
        # if p.pkey!=0:
        #     sh.copyfile(os.path.join('.','sfrd_peaks_parameters'),
        #                 os.path.join(T['stat'],'sfrd_peaks_parameters'))
        
        print('\nOutput directory tree created.')
    
    return T


def tab_sorter(name,p,T,**kwargs):
    """
    Sorts all kinds of the calculated quantities into the output subdirectories. 
    
    :param name: Short name of the quantity.
    :type name: str
    :param p: Set of model parameters from the parameter file. 
    :type p: namedtuple
    :param T: Output directory tree (created by :func:`jjmodel.iof.dir_tree`).
    :type T: dict
    :param R: Optional. Galactocentric distance, kpc. Must be specified for the vertical 
        density profiles (**name** = ``'rhoz'``), vertical force (``'Kz'``), 
        contributions of the additional peaks to the thin-disk SFR (``'Fp'``), 
        and peaks' scale heights (``'Hdp'``). 
    :type R: scalar 
    :param print: Optional. If True, the full list of names is printed. 
    :type print: boolean 

    :return: Path to the directory where the quantity given by **name** will be saved
        as txt table. Or, if **print** is True, none. 
    :rtype: str or none
    """
    
    namespace = {'SR':os.path.join(T['inptab'],'Sigma.txt'),
                 'SFRd':os.path.join(T['inptab'],'SFR_d.txt'),
                 'NSFRd':os.path.join(T['inptab'],'NSFR_d.txt'),
                 'SFRt':os.path.join(T['inptab'],'SFR_t.txt'),
                 'NSFRt':os.path.join(T['inptab'],'NSFR_t.txt'),
                 'SFRtot':os.path.join(T['inptab'],'SFR_tot.txt'),
                 'NSFRtot':os.path.join(T['inptab'],'NSFR_tot.txt'),
                 'gd':os.path.join(T['inptab'],'g_d.txt'),
                 'gt':os.path.join(T['inptab'],'g_t.txt'),
                 'AMRt':os.path.join(T['inptab'],'AMR_t.txt'),
                 'AMRd':os.path.join(T['inptab'],'AMR_d.txt'),
                 'AMRtot':os.path.join(T['inptab'],'AMR_tot.txt'),
                 'Hg1':os.path.join(T['inptab'],'H_g1.txt'),
                 'Hg2':os.path.join(T['inptab'],'H_g2.txt'),
                 'H2':os.path.join(localpath,'input','gas','H2.txt'),
                 'HI':os.path.join(localpath,'input','gas','HI.txt'),
                 'SFRd0':os.path.join(T['inptab'],''.join(('SFR_d_R',str(p.Rsun),'.txt'))),
                 'SFRt0':os.path.join(T['inptab'],''.join(('SFR_t_R',str(p.Rsun),'.txt'))),
                 'SFRtot0':os.path.join(T['inptab'],''.join(('SFR_tot_R',str(p.Rsun),'.txt'))),
                 'NSFRd0':os.path.join(T['inptab'],''.join(('NSFR_d_R',str(p.Rsun),'.txt'))),
                 'NSFRtot0':os.path.join(T['inptab'],''.join(('NSFR_tot_R',str(p.Rsun),'.txt'))),
                 'gd0':os.path.join(T['inptab'],''.join(('g_d_R',str(p.Rsun),'.txt'))),
                 'AMRd0':os.path.join(T['inptab'],''.join(('AMR_d_R',str(p.Rsun),'.txt'))),
                 'AVR':os.path.join(T['kintab'],'AVR.txt'),
                 'Sige':os.path.join(T['kintab'],'Sige.txt'),
                 'Sigt':os.path.join(T['kintab'],'Sig_t.txt'),
                 'Sigg1':os.path.join(T['kintab'],'Sig_g1.txt'),
                 'Sigg2':os.path.join(T['kintab'],'Sig_g2.txt'),
                 'Heffd':os.path.join(T['heighttab'],'Heff_d.txt'),
                 'Heffd0':os.path.join(T['heighttab'],''.join(('Heff_d_R',str(p.Rsun),'.txt'))),
                 'Hd':os.path.join(T['heighttab'],'H_d.txt'),
                 'Phi':os.path.join(T['fitab'],'Fiz.txt'),
                 'Hsh':os.path.join(T['heighttab'],'H_sh.txt'),
                 'Hdh':os.path.join(T['heighttab'],'H_dh.txt'),
                 'Ht':os.path.join(T['heighttab'],'H_t.txt'),
                 'Kz0':os.path.join(T['fitab'],''.join(('Kz_R',str(p.Rsun),'.txt'))),
                 'AVR0':os.path.join(T['kintab'],''.join(('AVR_R',str(p.Rsun),'.txt'))),
                 'Hd0':os.path.join(T['heighttab'],''.join(('H_d_R',str(p.Rsun),'.txt'))),
                 'Ht0':os.path.join(T['heighttab'],''.join(('H_t_R',str(p.Rsun),'.txt'))),
                 'Hsh0':os.path.join(T['heighttab'],''.join(('H_sh_R',str(p.Rsun),'.txt'))),
                 'Hdh0':os.path.join(T['heighttab'],''.join(('H_dh_R',str(p.Rsun),'.txt'))),
                 'Phi0':os.path.join(T['fitab'],''.join(('Fiz_R',str(p.Rsun),'.txt'))),
                 'rhoz0':os.path.join(T['denstab'],''.join(('Rho_R',str(p.Rsun),'.txt'))),
                 'Hdp0':os.path.join(T['heighttab'],''.join(('H_dp_R',str(p.Rsun),'.txt'))),
                 'Fp0':os.path.join(T['inptab'],''.join(('F_dp_R',str(p.Rsun),'.txt')))
                 }
    
    if 'print' not in kwargs or kwargs['print']==False: 
        if 'R' in kwargs:
            namespace['rhoz'] = os.path.join(T['denstab'],''.join(('Rhoz_R',str(kwargs['R']),'.txt')))
            namespace['Kz'] = os.path.join(T['fitab'],''.join(('Kz_R',str(kwargs['R']),'.txt')))
            namespace['Fp'] = os.path.join(T['inptab'],''.join(('F_dp_R',str(kwargs['R']),'.txt')))
            namespace['Hdp'] = os.path.join(T['heighttab'],''.join(('H_dp_R',str(kwargs['R']),'.txt')))
            
        return namespace[name]
    
    else: 
        print(namespace.keys())
        
    
 
    
def tab_reader(names,p,T,**kwargs):
    """
    Reads data from the output directory tree.
    
    :param names: Names of the tables to read. Names of the quantites are the same 
        as for :func:`jjmodel.iof.tab_sorter`. Also, if **tab** is True, 
        **names** can refer to stellar populations: 
            
            - ``'ceph'`` - Cepheids Type I (selected by :meth:`jjmodel.analysis.GetPopulations.cepheids_type1`)
            - ``'a'`` - A stars (:meth:`jjmodel.analysis.GetPopulations.a_stars`)
            - ``'f'`` - F stars (:meth:`jjmodel.analysis.GetPopulations.f_stars`)
            - ``'rc+'`` - RC stars, contaminated by HGB (:meth:`jjmodel.analysis.GetPopulations.rc_simple`)
            - ``'rc'`` - RC stars, clean (:meth:`jjmodel.analysis.GetPopulations.rc_clean`)
            - ``'gdw'`` - G dwarfs (:meth:`jjmodel.analysis.GetPopulations.g_dwarfs`)
            - ``'kdw'`` - K dwarfs (:meth:`jjmodel.analysis.GetPopulations.k_dwarfs`)
            - ``'mdw'`` - M dwarfs (:meth:`jjmodel.analysis.GetPopulations.m_dwarfs`)
            - ``'ssp'`` - full stellar assembly table
            
    :type names: list
    :param p: Set of model parameters from the parameter file. 
    :type p: namedtuple
    :param T: Output directory tree (created by :func:`jjmodel.iof.dir_tree`).
    :type T: dict
    :param tab: If True, tables with stellar assemblies will be read (csv tables). 
        Otherwise some model produced txt tables are read. 
    :type tab: boolean 
    :param mode_iso: Optional. Defines which set of isochrones is used, can be ``'Padova'``, ``'MIST'``, or ``'BaSTI'``. 
        If not specified, Padova is the default isochrone set. 
    :type mode_iso: str 
    :param R: Optional. Galactocentric distance, kpc. Must be specified for the vertical 
        density profiles (**name** = ``'rhoz'``), vertical force (``'Kz'``), 
        contributions of the additional peaks to the thin-disk SFR (``'Fp'``), 
        and peaks' scale heights (``'Hdp'``). Also, has to be given if **tab** is True. 
    :type R: scalar 
    
    :return: Data tables. 
    :rtype: list[array-likes]
    """
    if ('tab' in kwargs) and (kwargs['tab']==True):
        Pn = {'a':'A','f':'F','rc':'RC','rc+':'RC+',
                   'gdw':'Gdw','kdw':'Kdw','rrl':'RRL','ceph':'Ceph'}
        pn = {'a':'a','f':'f','rc':'rc','rc+':'rc',
              'gdw':'gdw','kdw':'kdw','rrl':'rrl','ceph':'ceph'}
        
        if names=='ssp':
            savedir = 'poptab'
            tabname = 'SSP'
        else:
            savedir = ''.join(('pop',pn[names],'tab'))
            tabname = Pn[names]
        if 'mode_iso' not in kwargs:
            mode_iso = 'Padova'
        else:
            mode_iso = kwargs['mode_iso']
        readpath = os.path.join(T[savedir],''.join((tabname,'_R',str(kwargs['R']),'_',kwargs['mode'],
                                                    '_',mode_iso,'.csv'))) 
        tables = Table.read(readpath)
    else:
        tables = [np.loadtxt(tab_sorter(i,p,T,**kwargs)).T for i in names]
    return tables


def hdp_reader(p,T,R):
    """
    Returns schale heights for peaks' subpopulations in the form of list of lists, [[]]. 
    """
    
    npeak = len(p.sigp)
    
    only_local = False
    r_scalar = False
    try:
        len(R)
    except:
        r_scalar = True
        if R==p.Rsun:
            only_local = True
    
    if r_scalar:
        if only_local:
            if npeak==1:
                Hdp = tab_reader(['Hdp0'],p,T)[0]
                sigp = [Hdp[0]]
                Hdp = [Hdp[1]]
            else:
                Hdp = tab_reader(['Hdp0'],p,T)[0]
                sigp = Hdp[0]
                Hdp = Hdp[1]
        else:
            if npeak==1:
                Hdp = tab_reader(['Hdp'],p,T,R=R)[0]
                sigp = [Hdp[0]]
                Hdp = [Hdp[1]]
            else:
                Hdp = tab_reader(['Hdp'],p,T,R=R)[0]
                sigp = Hdp[0]
                Hdp = Hdp[1]
    else:
        Hdp = [tab_reader(['Hdp'],p,T,R=radius)[0] for radius in R]
       
        if npeak==1:
            sigp = [[table[0]] for table in Hdp]
            Hdp = [[table[1]] for table in Hdp]
        else:
            sigp = [table[0] for table in Hdp]
            Hdp = [table[1] for table in Hdp]
    
    return (sigp, Hdp)
    


class TabSaver():
    """
    Sorts and saves calculated quantities into the output subfolders. 
    """
    
    def __init__(self,p,a,**kwargs):
        """
        Initialization of the class instance. 
        
        :param p: Set of model parameters from the parameter file. 
        :type p: namedtuple
        :param a: Collection of the fixed model parameters, useful quantities, and arrays.
        :type a: namedtuple
        :param number: Optional. If True, calculated quantity is (or is weighted by) the spatial number 
            density of stars in :math:`\mathrm{number \ pc^{-3}}`, 
            not matter density in :math:`\mathrm{M_\odot \ pc^{-3}}`. 
        :type number: boolean
        :param sigma: Optional. If True, the result is (or is weighted by) surface density 
            in :math:`\mathrm{M_\odot \ pc^{-2}}`, 
            otherwise the midplane mass density in :math:`\mathrm{M_\odot \ pc^{-3}}` is used. 
            In combination with **number** = True, uses the *number* surface density in 
            :math:`\mathrm{number \ pc^{-2}}`.
        :type sigma: boolean
        :param fig: Optional. If True, a plot will be saved. By default, class methods save a table.  
        :type fig: boolean
        :param save_format: Optional. Format of the figure. Can be used only when **fig** is True. 
        :type save_format: str
        :param normalized: Optional. Applicable in case if vertical density profiles will be saved. 
            If True, the profiles are normalized at each height  
            on the total density at this height. Also the file name will start from 
            ``'NRhoz'`` instead of ``'Rhoz'``. 
        :type normalized: boolean
        :param cumulative: Optional. Applicable to density profiles, age and metallicity distributions. 
            If True, the normalized cumulative quantities are saved. In general, ``'C'`` is added in front 
            of a standard file name; in case of density profiles, file will be called ``'NCMassz'``. 
        :type cumulative: boolean
        :param mode_pop: Optional. Name of stellar population. Can be a pre-defined one 
            (``'a'``, ``'f'``, ``'ceph'``, ``'rc'``, ``'rc+'``, ``'gdw'``, ``'kdw'``, ``'mdw'``) or custom 
            (if it was selected and saved as a table in advance). Adds a suffix with the name of 
            population to the standard file name. 
        :type mode_pop: str
        :param number: Optional. If True, this indicates that the quantity to save was calculated with 
            the help of stellar assembly tables (contains stellar number densities or something weighted 
            by them). Usually adds a suffix ``'(nw)'`` (stands for 'number-weighted') to the standard file name. 
            Otherwise, the suffix is ``'(rw)'`` ('rho', i.e., 'density-weighted'). 
        :type number: boolean
        :param between: Optional. If True, the output quantity corresponds to the age intervals, not  
            to individual single mono-age subpopulations. Applicable to methods with parameter **ages**. 
            If False, a suffix ``'sgl'`` will be added to the standard file name 
            (stands for 'single-age populations').  
        :type between: boolean
        :param dz: Optional. Vertical resolution, pc. Applicable to methods which save R-z maps, 
            adds a suffix ``'dz'`` with **z** in kpc. 
        :type dz: scalar
        :param R: Optional. Galactocentric distance, kpc. Suffix ``'_R'`` with the given value of radis 
            is added to the output file name. 
        :type R: scalar
        :param ages: Set of age bins, Gyr. If specified, a suffix ``'age'`` with the range of ages 
            (or min age, max age, and step) is added to the file name. 
        :type ages: array-like
        :param mets: Set of metallicity bins. If specified, a suffix ``'FeH'`` with the range of metallicities 
            (or min metallicity, max metallicity, and step) is added to the file name. 
        :type mets: array-like
        :param vln: Optional. A string with information about the volume. Applicable to methods 
            which save Hess diagrams and stellar number densitites corresponding to some volume. 
        :type vln: str
        """
        
        self.p, self.a = p, a
        self.kwargs = kwargs
        Q0 = 'Rho'
        Q1 = 'rho[Msun/pc^3]'
        w = 'density-weighted'
        w0 = 'rw'
        if 'number' in kwargs and kwargs['number']==True:
            Q0 = 'N'  
            Q1 = 'N[1/pc^3]'
            w = 'number-weighted'
            w0 = 'nw'
        if 'sigma' in kwargs and kwargs['sigma']==True:
            if 'number' in kwargs and kwargs['number']==True:
                Q0 = 'SigmaN'
                Q2 = 'N[1/pc^2]'
            else:    
                Q0 = 'Sigma'
                Q2 = 'Sigma[Msun/pc^2]'
        else:
            if 'number' in kwargs and kwargs['number']==True:
                Q2 = 'N[1/pc^3]'
            else:    
                Q2 = 'rho[Msun/pc^3]'      
        self.Q0 = Q0
        self.Q1 = Q1
        self.Q2 = Q2
        self.w = w 
        self.w0 = w0 
                    #tabname = tab_sorter('rctab',self.p,self.a.T,R=self.R,mode=mode_comp,mode_pop='rc') 
            #rc_sample_tab.write(tabname,overwrite=True)
        self.lb = {'d':'thin disk','t':'thick disk','sh':'stellar halo',
                   'dt':'total disk','tot':'total disk + halo'}
        self.pn = {'a':'a','f':'f','rc':'rc','rc+':'rc','rc_compl':'rc',
                   'gdw':'gdw','kdw':'kdw','rrl':'rrl','ceph':'ceph'}
        self.Pn = {'a':'A','f':'F','rc':'RC','rc+':'RC+','rc_compl':'RC_compl',
                   'gdw':'Gdw','kdw':'Kdw','rrl':'RRL','ceph':'Ceph'}
        
        self.dir, self.fmt = 'tab', '.txt'
        if 'fig' in self.kwargs and self.kwargs['fig']==True:
            self.dir2 = 'plt'
            self.fmt2 = 'png'
            if 'save_format' in kwargs:
                self.fmt2 = '.' + self.kwargs['save_format']
            
    def rhoz_save(self,profiles,R):
        """
        Saves vertical density profiles to the subfolder ``a.T['dens']``. 
        File name base is ``'Rhoz'``, ``'NRohz'``, or ``'NCMassz'`` (depending on the class 
        instance *kwargs*). 
        
        :param profiles: Vertical density profiles, where the first column is ``a.z`` and 
            the rest are density profiles of the thin disk, thick disk, molecular gas, atomic gas, DM halo, 
            and stellar halo. 
        :type profiles: array-like
        :param R: Galactocentric distance, kpc. Suffix ``'_R'`` with the given value of radis 
            is added to the output file name. 
        :type R: scalar
        
        :return: None. 
        """        
        
        filename = 'Rhoz'
        if 'normalized' in self.kwargs and self.kwargs['normalized']==True:
            filename = 'NRhoz'
        if 'cumulative' in self.kwargs and self.kwargs['cumulative']==True:
            filename = 'NCMassz'
        savepath = ''.join((filename,'_R',str(R)))
        if 'fig' in self.kwargs and self.kwargs['fig']==True:
            plt.savefig(os.path.join(self.a.T[''.join(('dens',self.dir2))],''.join((savepath,self.fmt2))))
        else:
            head_rhoz = '|z|[pc], Q_thin.d, Q_mol.gas, Q_atom.gas, Q_thick.d, Q_DM, Q_st.halo with Q = '
            if (('normalized' in self.kwargs and self.kwargs['normalized']==True) or
                ('cumulative' in self.kwargs and self.kwargs['cumulative']==True)):
                if 'normalized' in self.kwargs and self.kwargs['normalized']==True:
                    head_rhoz += 'rho(|z|)_i/rho_tot(|z|)'
                if 'cumulative' in self.kwargs and self.kwargs['cumulative']==True:
                    head_rhoz += 'Mass(|z|)_i/Mass_tot(|z|)'
            else:
                head_rhoz += 'rho(|z|)_i[Msun/pc^3]'
            np.savetxt(os.path.join(self.a.T[''.join(('dens',self.dir))],''.join((savepath,self.fmt))),
                       profiles,header=head_rhoz)
            
            
    def fi_iso_save(self,profiles):
        """
        Saves vertical density profiles to the subfolder ``a.T['dens']``. 
        File name base is ``'Rhoz'``, ``'NRohz'``, or ``'NCMassz'`` (depending on the class 
        instance *kwargs*). 
        
        :param profiles: Vertical density profiles, where the first column is ``a.z`` and 
            the rest are density profiles of the thin disk, thick disk, molecular gas, atomic gas, DM halo, 
            and stellar halo. 
        :type profiles: array-like
        :param R: Galactocentric distance, kpc. Suffix ``'_R'`` with the given value of radis 
            is added to the output file name. 
        :type R: scalar
        
        :return: None. 
        """  
        savepath = 'FiRz_iso'
        if 'fig' in self.kwargs and self.kwargs['fig']==True:
            plt.savefig(os.path.join(self.a.T[''.join(('fi',self.dir2))],''.join((savepath,self.fmt2))))
        np.savetxt(os.path.join(self.a.T[''.join(('fi',self.dir))],''.join((savepath,self.fmt))),
                   profiles,header=''.join(('|z|[pc], ','Phi[m^2/s^2] at R = [',str(self.p.Rmin),',',
                               str(self.p.Rmax),'] kpc with step dR = ',str(self.p.dR),' kpc')))  
                       

    def rot_curve_save(self,profiles):
        """
        Saves rotation curve to the subfolder ``a.T['kintab']``. 
        File name base is ``'Vc_R'``. 
        
        :param profiles: The first column is Galactocentric distance R, kpc. 
            Other columns contain rotation velocity (:math:`\mathrm{km \ s^{-1}}`) 
            in the following order: total, bulge, thin disk, thick disk, molecular gas, 
            atomic gas, DM halo, stellar halo. 
        :type profiles: array-like
        
        :return: None. 
        """  
        
        np.savetxt(os.path.join(self.a.T['kintab'],''.join(('Vc_R',self.fmt))),
                   profiles,header='R[kpc], Q_tot, Q_bulge, Vc_thin.d, Q_thick.d, Q_mol.g, '+\
                                        'Q_at.g, Q_DM, Q_st.halo with Q = Vc[km/s]')  

            
    def poptab_save(self,table,mode,mode_iso,R,mode_pop_name):
        """
        Saves stellar assembly table to the subfolder ``a.T['poptab']`` . 
        
        :param table: Table with stellar assemblies, output of 
            :func:`jjmodel.populations.stellar_assemblies_iso` or methods of 
            :class:`jjmodel.analysis.GetPopulations`.
        :type table: astropy table
        :param mode: Galactic component, can be ``'d'``, ``'t'``, ``'sh'``, ``'dt'``, or ``'tot'`` 
            (thin disk, thick disk, halo, thin+thick disk, or total). 
            Adds suffix ``'_'+mode`` to the file name. 
        :type mode: str
        :param mode_iso: Defines which set of isochrones is used, can be ``'Padova'``, ``'MIST'``, 
            or ``'BaSTI'``. Adds suffix ``'_'+mode_iso`` to the file name. 
        :type mode_iso: str 
        :param R: Galactocentric distance, kpc. Suffix ``'_R'`` with the given value of radis 
            is added to the output file name. 
        :type R: scalar
        :param mode_pop_name: Short name of the population (see :func:`jjmodel.iof.tab_reader`) 
            or any custom name for the table. 
        :type mode_pop_name: str 
                
        :return: None. 
        """  
        
        try:
            savedir = ''.join(('pop',self.pn[mode_pop_name],'tab'))
            tabname = self.Pn[mode_pop_name]
        except: 
            savedir = 'poptab'
            tabname = mode_pop_name
        savepath = os.path.join(self.a.T[savedir],''.join((tabname,'_R',str(R),'_',mode,
                                                           '_',mode_iso,'.csv')))                                                    
        table.write(savepath,overwrite=True)
                                           
    
    def rhoz_monoage_save(self,profiles,mode,R,ages):
        """
        Saves vertical mono-age density profiles to the subfolder ``a.T['dens']``. 
        File name base is ``'Rho'``, ``'N'``,``'Sigma'``, or ``'SigmaN'`` 
        (depending on parameters **sigma** and **number** in the class instance *kwargs*). 
        
        :param profiles: Vertical density profiles, where the first column is ``a.z`` and 
            the rest are density profiles of the mono-age subpopulations (bins) corresponding to 
            an age grid. 
        :type profiles: array-like
        :param mode: Galactic component, can be ``'d'``, ``'t'``, ``'sh'``, ``'dt'``, or ``'tot'`` 
            (thin disk, thick disk, halo, thin+thick disk, or total). 
            Adds suffix ``'_'+mode`` to the file name. 
        :type mode: str
        :param R: Galactocentric distance, kpc. Suffix ``'_R'`` with the given value of radis 
            is added to the output file name. 
        :type R: scalar
        :param ages: Set of age bins, Gyr. Adds suffix ``'_age'`` with the min-max age and age step. 
        :type ages: array-like
        
        :return: None. 
        """    
        
        savepath = ''.join((self.Q0,'z_R',str(R),'_age[',str(ages[0]),':',str(ages[-1]),                                                                           
                                        ',',str(np.mean(np.diff(ages))),']'))
        if ('between' not in self.kwargs) or (self.kwargs['between']==False):
            savepath += 'sgl'
        savepath += '_' + mode
        label = self.lb[mode]
        if ('mode_pop' in self.kwargs):
            savepath += '_' + self.kwargs['mode_pop']
            label += ', ' + self.Pn[self.kwargs['mode_pop']] 
        if ('tab' in self.kwargs):
            savepath += '_tab'
            label += ', custom population'
        if 'fig' in self.kwargs and self.kwargs['fig']==True:
            plt.savefig(os.path.join(self.a.T[''.join(('dens',self.dir2))],''.join((savepath,self.fmt2))))
        else:
            np.savetxt(os.path.join(self.a.T[''.join(('dens',self.dir))],''.join((savepath,self.fmt))),
                       np.hstack((self.a.z.reshape(self.a.n,1),profiles.T)),
                       header=''.join(('|z|[pc], ',self.Q1,' (',label,') for ages = [',str(ages[0]),',',
                               str(ages[-1]),'] Gyr with step = ',str(round(np.mean(np.diff(ages)),3)),
                               ' Gyr at R = ',str(R),' kpc')))                               
        
                                   
    def rhoz_monomet_save(self,profiles,mode,R,mets):
        """
        Saves vertical mono-metallicity density profiles to the subfolder ``a.T['dens']``. 
        File name base is ``'Rho'``, ``'N'``,``'Sigma'``, or ``'SigmaN'`` 
        (depending on parameters **sigma** and **number** in the class instance *kwargs*). 
        
        :param profiles: Vertical density profiles, where the first column is ``a.z`` and 
            the rest are density profiles of the mono-metallicity subpopulations (bins) corresponding to 
            the metallicity grid. 
        :type profiles: array-like
        :param mode: Galactic component, can be ``'d'``, ``'t'``, ``'sh'``, ``'dt'``, or ``'tot'`` 
            (thin disk, thick disk, halo, thin+thick disk, or total). 
            Adds suffix ``'_'+mode`` to the file name. 
        :type mode: str
        :param R: Galactocentric distance, kpc. Suffix ``'_R'`` with the given value of radis 
            is added to the output file name. 
        :type R: scalar
        :param mets: Set of metallicity bins. Adds suffix ``'_FeH'`` with the min-max metallicity and 
            step in metallicity. 
        :type mets: array-like
        
        :return: None. 
        """  
        
        savepath = ''.join((self.Q0,'z_R',str(R),
                                        '_FeH[',str(round(mets[0],3)),':',str(round(mets[-1],3)),
                                         ',',str(round(np.mean(np.diff(mets)),3)),']_',mode))
        label = self.lb[mode]
        if ('mode_pop' in self.kwargs):
            savepath += '_' + self.kwargs['mode_pop']
            label += ', ' + self.Pn[self.kwargs['mode_pop']] 
        if ('tab' in self.kwargs):
            savepath += '_tab'
            label += ', custom population'
        if 'fig' in self.kwargs and self.kwargs['fig']==True:
            plt.savefig(os.path.join(self.a.T[''.join(('dens',self.dir2))],''.join((savepath,self.fmt2))))
        else:
            np.savetxt(os.path.join(self.a.T[''.join(('dens',self.dir))],''.join((savepath,self.fmt))),
                       np.hstack((self.a.z.reshape(self.a.n,1),profiles.T)),                   
                       header=''.join(('|z|[pc], ',self.Q1,' (',label,') for [Fe/H] = [',
                               str(round(mets[0],3)),',',str(round(mets[-1],3)),'] with step = ',
                               str(round(np.mean(np.diff(mets)),3)),' at R = ',str(R),' kpc')))
                
        
    def rhor_monoage_save(self,profiles,mode,zlim,ages):        
        """
        Saves radial mono-age density profiles to the subfolder ``a.T['dens']``. 
        File name base is ``'Rho'``, ``'N'``,``'Sigma'``, or ``'SigmaN'`` 
        (depending on parameters **sigma** and **number** in the class instance *kwargs*). 
        
        :param profiles: Density profiles, where the first column is ``a.R`` and 
            the rest are density profiles of the mono-age subpopulations (bins) corresponding to 
            an age grid. 
        :type profiles: array-like
        :param mode: Galactic component, can be ``'d'``, ``'t'``, ``'sh'``, ``'dt'``, or ``'tot'`` 
            (thin disk, thick disk, halo, thin+thick disk, or total). 
            Adds suffix ``'_'+mode`` to the file name. 
        :type mode: str
        :param zlim: Range of heights, pc. Suffix ``'_z'`` with the given values  
            is added to the output file name. 
        :type R: scalar
        :param ages: Set of age bins, Gyr. Adds suffix ``'_age'`` with the min-max age and age step. 
        :type ages: array-like
        
        :return: None. 
        """  
        
        savepath = ''.join((self.Q0,'R_z[',str(zlim[0]/1e3),',',str(zlim[1]/1e3),
                                         ']_age[',str(ages[0]),':',str(ages[-1]),',',
                                         str(round(np.mean(np.diff(ages)),3)),']'))  
        if ('between' not in self.kwargs) or (self.kwargs['between']==False):
            savepath += 'sgl'   
        savepath += '_' + mode  
        label = self.lb[mode]                                        
        if 'mode_pop' in self.kwargs:
            savepath += '_' + self.kwargs['mode_pop']
            label += ', ' + self.Pn[self.kwargs['mode_pop']] 
        if ('tab' in self.kwargs):
            savepath += '_tab'
            label += ', custom population'
        if 'fig' in self.kwargs and self.kwargs['fig']==True:
            plt.savefig(os.path.join(self.a.T[''.join(('dens',self.dir2))],''.join((savepath,self.fmt2))))
        else:
            np.savetxt(os.path.join(self.a.T[''.join(('dens',self.dir))],''.join((savepath,self.fmt))),
                       np.hstack((self.a.R.reshape(self.a.Rbins,1),profiles.T)),
                       header=''.join(('R[kpc], ',self.Q2,' (',label,') for ages = [',str(ages[0]),',',
                               str(ages[-1]),'] Gyr with step = ',str(round(np.mean(np.diff(ages)),3)),
                               ' Gyr at z = ',str(zlim).replace(' ', ''),' pc')))
                                                                     

    def rhor_monomet_save(self,profiles,mode,zlim,mets):
        """
        Saves radial mono-metallicity density profiles to the subfolder ``a.T['dens']``. 
        File name base is ``'Rho'``, ``'N'``,``'Sigma'``, or ``'SigmaN'`` 
        (depending on parameters **sigma** and **number** in the class instance *kwargs*). 
        
        :param profiles: Density profiles, where the first column is ``a.R`` and 
            the rest are density profiles of the mono-metallicity subpopulations (bins) corresponding to 
            metallicity grid. 
        :type profiles: array-like
        :param mode: Galactic component, can be ``'d'``, ``'t'``, ``'sh'``, ``'dt'``, or ``'tot'`` 
            (thin disk, thick disk, halo, thin+thick disk, or total). 
            Adds suffix ``'_'+mode`` to the file name. 
        :type mode: str
        :param zlim: Range of heights, pc. Suffix ``'_z'`` with the given values  
            is added to the output file name. 
        :type R: scalar
        :param mets: Set of metallicity bins. Adds suffix ``'_FeH'`` with the min-max metallicity 
            and step in metallicity. 
        :type mets: array-like
        
        :return: None. 
        """ 
        
        savepath = ''.join((self.Q0,'R_z[',str(zlim[0]/1e3),',',str(zlim[1]/1e3),
                                         ']_FeH[',str(round(mets[0],3)),':',str(round(mets[-1],3)),',',
                                         str(round(np.mean(np.diff(mets)),3)),']_',mode))   
        label = self.lb[mode]                                                   
        if 'mode_pop' in self.kwargs:
            savepath += '_' + self.kwargs['mode_pop']
            label += ', ' + self.Pn[self.kwargs['mode_pop']] 
        if ('tab' in self.kwargs):
            savepath += '_tab'
            label += ', custom population'
        if 'fig' in self.kwargs and self.kwargs['fig']==True:
            plt.savefig(os.path.join(self.a.T[''.join(('dens',self.dir2))],''.join((savepath,self.fmt2))))
        else:
            np.savetxt(os.path.join(self.a.T[''.join(('dens',self.dir))],''.join((savepath,self.fmt))),
                       np.hstack((self.a.R.reshape(self.a.Rbins,1),profiles.T)),
                       header=''.join(('R[kpc], ',self.Q2,' (',label,') for [Fe/H] = [',
                               str(round(mets[0],3)),',',str(round(mets[-1],3)),'] with step = ',
                               str(round(np.mean(np.diff(mets)),3)),
                               ' at z = ',str(zlim).replace(' ', ''),' pc')))
        
                   
    def agez_save(self,profiles,mode):
        """
        Saves vertical age profile to the subfolder ``a.T['age']``. 
        File name base is ``'Age'``. 
        
        :param profiles: Vertical age profiles. Can include two tables - profiles at the different R 
            and at ``p.Rsun``, or a single table with the local profile. Tables contain only age profiles,
            without the vertical grid column ``a.z``. 
        :type profiles: array-like
        :param mode: Galactic component, can be ``'d'``, ``'t'``, ``'sh'``, ``'dt'``, or ``'tot'`` 
            (thin disk, thick disk, halo, thin+thick disk, or total). 
            Adds suffix ``'_'+mode`` to the file name. 
        :type mode: str
        
        :return: None. 
        """  
        
        only_local = False
        if 'R' in self.kwargs and self.kwargs['R']==self.p.Rsun:
            only_local = True
        
        savepath0 = ''.join(('Age(',self.w0,')z_R',str(self.p.Rsun),'_',mode))
        label = self.lb[mode]
        if not only_local:
            savepath = ''.join(('Age(',self.w0,')z_',mode))
        
        if 'mode_pop' in self.kwargs:
            savepath0 += '_' + self.kwargs['mode_pop']
            label += ', ' + self.Pn[self.kwargs['mode_pop']] 
            if not only_local:
                savepath += '_' + self.kwargs['mode_pop']
        if ('tab' in self.kwargs):
            savepath0 += '_tab'
            label += ', custom population'
            if not only_local:
                savepath += '_tab'
                
        if only_local:
            agezr0 = profiles
        else:
            agezr,agezr0 = profiles
            
        if 'fig' in self.kwargs and self.kwargs['fig']==True:
            if not only_local:
                plt.savefig(os.path.join(self.a.T[''.join(('age',self.dir2))],
                                         ''.join((savepath,self.fmt2))))
            else:
                plt.savefig(os.path.join(self.a.T[''.join(('age',self.dir2))],
                                         ''.join((savepath0,self.fmt2))))
        else:
            if not only_local:
                np.savetxt(os.path.join(self.a.T[''.join(('age',self.dir))],''.join((savepath,self.fmt))),
                           np.hstack((self.a.z.reshape(self.a.n,1),agezr.T)),                   
                           header=''.join(('|z|[pc], <age>[Gyr] (',self.w,', ',label,') at R = [',str(self.p.Rmin),
                                       ',',str(self.p.Rmax),'] kpc with step = ',str(self.p.dR),' kpc')))
            np.savetxt(os.path.join(self.a.T[''.join(('age',self.dir))],''.join((savepath0,self.fmt))),
                       np.stack((self.a.z,agezr0),axis=-1),                   
                       header=''.join(('|z|[pc], <age>[Gyr] (',self.w,', ',label,') at R = ',
                                   str(self.p.Rsun),' kpc')))
                                       
        
    def ager_save(self,profiles,mode,zlim):
        """
        Saves radial age profile to the subfolder ``a.T['age']``. 
        File name base is ``'Age'``. 
        
        :param profiles: Radial age profiles. Tables contain only age profiles,
            without the radial grid column ``a.R``. 
        :type profiles: array-like
        :param mode: Galactic component, can be ``'d'``, ``'t'``, ``'sh'``, ``'dt'``, or ``'tot'`` 
            (thin disk, thick disk, halo, thin+thick disk, or total). 
            Adds suffix ``'_'+mode`` to the file name. 
        :type mode: str
        :param zlim: Range of heights to be considered, pc. Adds to the file name suffix ``'_z'`` 
            with the given heights (converted for shortness into kpc). 
        :type zlim: array-like 
        
        :return: None. 
        """ 
        
        if len(zlim)==2:
            savepath = ''.join(('Age(',self.w0,')R_z[',str(zlim[0]/1e3),',',str(zlim[1]/1e3),']_',mode))
        else:
            dz = np.mean(np.diff(zlim))/1e3
            savepath = ''.join(('Age(',self.w0,')R_z[',str(zlim[0]/1e3),':',str(zlim[-1]/1e3),',',
                                str(dz),']_',mode))
        label = self.lb[mode]                        
        if 'mode_pop' in self.kwargs:
            savepath += '_' + self.kwargs['mode_pop']
            label += ', ' + self.Pn[self.kwargs['mode_pop']] 
        if ('tab' in self.kwargs):
            savepath += '_tab'
            label += ', custom population'
        if 'fig' in self.kwargs and self.kwargs['fig']==True:
            plt.savefig(os.path.join(self.a.T[''.join(('age',self.dir2))],''.join((savepath,self.fmt2))))
        else:
            try:
                out = np.hstack((self.a.R.reshape(self.a.Rbins,1),profiles.T))
            except:
                out = np.stack((self.a.R,profiles),axis=-1)
            np.savetxt(os.path.join(self.a.T[''.join(('age',self.dir))],''.join((savepath,self.fmt))),
                       out,header=''.join(('R[kpc], <age>[Gyr] (',self.w,', ',label,') at |z| = ',
                                   str(zlim).replace(' ', ''),' pc')))               
    
        
    def metz_save(self,profiles,mode):
        """
        Saves vertical netallicity profile to the subfolder ``a.T['met']``. 
        File name base is ``'FeH'``. 
        
        :param profiles: Vertical metallicity profiles. Can include two tables - profiles at the different R 
            and at ``p.Rsun``, or a single table with the local profile. Tables contain only metalllicity 
            profiles, without the vertical grid column ``a.z``. 
        :type profiles: array-like
        :param mode: Galactic component, can be ``'d'``, ``'t'``, ``'sh'``, ``'dt'``, or ``'tot'`` 
            (thin disk, thick disk, halo, thin+thick disk, or total). 
            Adds suffix ``'_'+mode`` to the file name. 
        :type mode: str
        
        :return: None. 
        """  
        
        only_local = False
        if 'R' in self.kwargs and self.kwargs['R']==self.p.Rsun:
            only_local = True
        
        if not only_local:
            savepath = ''.join(('FeH(',self.w0,')z_',mode))
        savepath0 = ''.join(('FeH(',self.w0,')z_R',str(self.p.Rsun),'_',mode))
        label = self.lb[mode]
        
        if 'mode_pop' in self.kwargs:
            if not only_local:
                savepath += '_' + self.kwargs['mode_pop']
            savepath0 += '_' + self.kwargs['mode_pop']
            label += ', ' + self.Pn[self.kwargs['mode_pop']] 
        if ('tab' in self.kwargs):
            if not only_local:
                savepath += '_tab'
            savepath0 += '_tab'
            label += ', custom population'
        if not only_local:
            fehzr,fehzr0 = profiles
        else:
            fehzr0 = profiles
            
        if 'fig' in self.kwargs and self.kwargs['fig']==True:
            if not only_local:
                plt.savefig(os.path.join(self.a.T[''.join(('met',self.dir2))],
                                         ''.join((savepath,self.fmt2))))
            else:
                plt.savefig(os.path.join(self.a.T[''.join(('met',self.dir2))],
                                         ''.join((savepath0,self.fmt2))))
        else:
            if not only_local:
                np.savetxt(os.path.join(self.a.T[''.join(('met',self.dir))],''.join((savepath,self.fmt))),
                           np.hstack((self.a.z.reshape(self.a.n,1),fehzr.T)),                   
                           header=''.join(('|z|[pc], <[Fe/H]> (',self.w,', ',label,') for R = [',str(self.p.Rmin),
                                           ',',str(self.p.Rmax),'] kpc with step = ',str(self.p.dR),' kpc')))
            np.savetxt(os.path.join(self.a.T[''.join(('met',self.dir))],''.join((savepath0,self.fmt))),
                   np.stack((self.a.z,fehzr0),axis=-1),header=''.join(('|z|[pc], <[Fe/H]> (',self.w,        
                   ', ',label,') at R = ',str(self.p.Rsun),' kpc')))                                   
                                   
        
    def metr_save(self,profiles,mode,zlim):
        """
        Saves radial metallicity profile to the subfolder ``a.T['met']``. 
        File name base is ``'FeH'``. 
        
        :param profiles: Radial metallicity profiles. Tables contain only metallicity profiles,
            without the radial grid column ``a.R``. 
        :type profiles: array-like
        :param mode: Galactic component, can be ``'d'``, ``'t'``, ``'sh'``, ``'dt'``, or ``'tot'`` 
            (thin disk, thick disk, halo, thin+thick disk, or total). 
            Adds suffix ``'_'+mode`` to the file name. 
        :type mode: str
        :param zlim: Range of heights to be considered, pc. Adds to the file name suffix ``'_z'`` 
            with the given heights (converted for shortness into kpc). 
        :type zlim: array-like 
        
        :return: None. 
        """ 
        
        if len(zlim)>2:
            savepath = ''.join(('FeH(',self.w0,')R_z[',str(zlim[0]/1e3),':',str(zlim[-1]/1e3),',',
                                str(np.mean(np.diff(zlim))/1e3),']_',mode))
        else:
            savepath = ''.join(('FeH(',self.w0,')R_z[',str(zlim[0]/1e3),',',str(zlim[1]/1e3),']_',mode))
        label = self.lb[mode]                        
        if 'mode_pop' in self.kwargs:
            savepath += '_' + self.kwargs['mode_pop']
            label += ', ' + self.Pn[self.kwargs['mode_pop']] 
        if ('tab' in self.kwargs):
            savepath += '_tab'
            label += ', custom population'
        if 'fig' in self.kwargs and self.kwargs['fig']==True:
            plt.savefig(os.path.join(self.a.T[''.join(('met',self.dir2))],''.join((savepath,self.fmt2))))
        else:
            try:
                out = np.hstack((self.a.R.reshape(self.a.Rbins,1),profiles.T))
            except:
                out = np.stack((self.a.R,profiles),axis=-1)
            np.savetxt(os.path.join(self.a.T[''.join(('met',self.dir))],''.join((savepath,self.fmt))),
                       out,header=''.join(('R[kpc], <[Fe/H]> (',self.w,', ',label,
                                           ') at |z| = ',str(zlim).replace(' ', ''),' pc')))                   
                                       

    def agehist_save(self,ages,mode,zlim):
        """
        Saves age distributions for the different Galactocentric distances 
        to the subfolder ``a.T['age']``. File name base is ``'f(age)'``. 
        
        :param ages: Can contain two tables - age distributions for the grid ``a.R`` and at ``p.Rsun``, 
            or only a single age distribution for the local volume. Tables contain only age distributions,
            without the time(age) grid column ``a.t``. 
        :type ages: array-like
        :param mode: Galactic component, can be ``'d'``, ``'t'``, ``'sh'``, ``'dt'``, or ``'tot'`` 
            (thin disk, thick disk, halo, thin+thick disk, or total). 
            Adds suffix ``'_'+mode`` to the file name. 
        :type mode: str
        :param zlim: Range of heights to be considered, pc. Adds to the file name suffix ``'_z'`` 
            with the given heights (converted for shortness into kpc). 
        :type zlim: array-like 
        
        :return: None. 
        """ 
        
        only_local = False
        if 'R' in self.kwargs and self.kwargs['R']==self.p.Rsun:
            only_local = True
        
        savepath0 = ''.join(('f(age)(',self.w0,')_R',str(self.p.Rsun),'_z[',
                             str(zlim[0]/1e3),',',str(zlim[1]/1e3),']_',mode))
        if not only_local:
            savepath = ''.join(('f(age)(',self.w0,')R_z[',str(zlim[0]/1e3),',',
                                str(zlim[1]/1e3),']_',mode))
        
        label = self.lb[mode]
        if 'cumulative' in self.kwargs and self.kwargs['cumulative']==True:
            savepath0 = 'C' + savepath0
            if not only_local:
                savepath = 'C' + savepath            
        if 'mode_pop' in self.kwargs:
            savepath0 += '_' + self.kwargs['mode_pop']
            label += ', ' + self.Pn[self.kwargs['mode_pop']] 
            if not only_local:  
                savepath += '_' + self.kwargs['mode_pop']       
        if ('tab' in self.kwargs):
            savepath0 += '_tab'
            label += ', custom population'
            if not only_local:  
                savepath += '_tab'            
        if 'sigma_gauss' in self.kwargs:
            savepath0 += '_sig' + str(self.kwargs['sigma_gauss'])
            label += ' x Gauss(' + str(self.kwargs['sigma_gauss']) + '[Gyr])'
            if not only_local:      
                savepath += '_sig' + str(self.kwargs['sigma_gauss'])
                
        if only_local:   
            ages0 = ages
        else:
            ages, ages0 = ages
            
        if 'fig' in self.kwargs and self.kwargs['fig']==True:
            if only_local:
                plt.savefig(os.path.join(self.a.T[''.join(('age',self.dir2))],
                                         ''.join((savepath0,self.fmt2))))
            else:
                plt.savefig(os.path.join(self.a.T[''.join(('age',self.dir2))],
                                         ''.join((savepath,self.fmt2))))
        else:
            if not only_local:
                np.savetxt(os.path.join(self.a.T[''.join(('age',self.dir))],''.join((savepath,self.fmt))),
                           np.hstack((self.a.t.reshape(self.a.jd,1),np.array(ages).T)),                   
                           header=''.join(('t[Gyr], fraction (',self.w,', ',label,') at R = [',str(self.p.Rmin),
                                           ',',str(self.p.Rmax),'] kpc with step = ',str(self.p.dR),' kpc and |z| = ',
                                           str(zlim).replace(' ', ''),' pc')))
            np.savetxt(os.path.join(self.a.T[''.join(('age',self.dir))],''.join((savepath0,self.fmt))),
                       np.stack((self.a.t,ages0),axis=-1),                   
                       header=''.join(('t[Gyr], fraction (',self.w,', ',label,') at R = ',str(self.p.Rsun),
                                       ' kpc and |z| = ',str(zlim).replace(' ', ''),' pc')))                                   
        
            
    def methist_save(self,mets,mode,zlim):
        """
        Saves metallicity distributions for the different Galactocentric distances 
        to the subfolder ``a.T['met']``. File name base is ``'f(FeH)'``. 
        
        :param mets: Can contain two tables - metallicity distributions for the grid ``a.R`` and at ``p.Rsun``, 
            or only a single metallicity distribution for the local volume. Also, 
            metallicity column must be added after the tables. 
        :type mets: array-like
        :param mode: Galactic component, can be ``'d'``, ``'t'``, ``'sh'``, ``'dt'``, or ``'tot'`` 
            (thin disk, thick disk, halo, thin+thick disk, or total). 
            Adds suffix ``'_'+mode`` to the file name. 
        :type mode: str
        :param zlim: Range of heights to be considered, pc. Adds to the file name suffix ``'_z'`` 
            with the given heights (converted for shortness into kpc). 
        :type zlim: array-like 
        
        :return: None. 
        """ 
        only_local = False
        if 'R' in self.kwargs and self.kwargs['R']==self.p.Rsun:
            only_local = True
        
        savepath0 = ''.join(('f(FeH)(',self.w0,')_R',str(self.p.Rsun),'_z[',
                             str(zlim[0]/1e3),',',str(zlim[1]/1e3),']_',mode))
        label = self.lb[mode]
        if not only_local:
            savepath = ''.join(('f(FeH)(',self.w0,')R_z[',str(zlim[0]/1e3),',',
                                str(zlim[1]/1e3),']_',mode))
        
        if 'cumulative' in self.kwargs and self.kwargs['cumulative']==True:
            savepath0 = 'C' + savepath0
            if not only_local:
                savepath = 'C' + savepath
        if 'mode_pop' in self.kwargs:
            savepath0 += '_' + self.kwargs['mode_pop']
            label += ', ' + self.Pn[self.kwargs['mode_pop']] 
            if not only_local:
                savepath += '_' + self.kwargs['mode_pop']
        if ('tab' in self.kwargs):
            savepath0 += '_tab'
            label += ', custom population'
            if not only_local:  
                savepath += '_tab'            
        if 'sigma_gauss' in self.kwargs:
            savepath0 += '_sig' + str(self.kwargs['sigma_gauss'])
            label += ' x Gauss(' + str(self.kwargs['sigma_gauss']) + ')'
            if not only_local:                
                savepath += '_sig' + str(self.kwargs['sigma_gauss'])
        
        if only_local:
            metdist0, metbinsc = mets
        else:
            metdist, metdist0, metbinsc = mets
            
        if 'fig' in self.kwargs and self.kwargs['fig']==True:
            if only_local:
                plt.savefig(os.path.join(self.a.T[''.join(('met',self.dir2))],
                                         ''.join((savepath0,self.fmt2))))
            else:
                plt.savefig(os.path.join(self.a.T[''.join(('met',self.dir2))],
                                         ''.join((savepath,self.fmt2))))
        else:
            if not only_local:
                np.savetxt(os.path.join(self.a.T[''.join(('met',self.dir))],''.join((savepath,self.fmt))),
                           np.hstack((metbinsc.reshape(len(metbinsc),1),np.array(metdist).T)),
                           header=''.join(('[Fe/H], fraction (',self.w,', ',label,') at R = [',str(self.p.Rmin),
                                           ',',str(self.p.Rmax),'] kpc with step = ',str(self.p.dR),' kpc and |z| = ',
                                           str(zlim).replace(' ', ''),' pc')))
            np.savetxt(os.path.join(self.a.T[''.join(('met',self.dir))],''.join((savepath0,self.fmt))),
                       np.stack((metbinsc,metdist0),axis=-1),
                       header=''.join(('[Fe/H], fraction (',self.w,', ',label,') at R = ',str(self.p.Rsun),
                                       ' kpc and |z| = ',str(zlim).replace(' ', ''),' pc')))                                   
                                   
    
    def hr_monoage_save(self,H,mode,ages):
        """
        Saves scale heights calculated for the mono-age subpopulations for the different Galactocentric 
        distances. Output subfolder is ``a.T['height']``. File name base is ``'H'``. 
        
        :param H: Table with scale heights (radial profiles in columns, age changes from column to column). 
            Radial grid ``a.R`` is not included into **H**.
        :type H: array-like
        :param mode: Galactic component, can be ``'d'``, ``'dt'``, or ``'tot'`` 
            (thin disk, thin+thick disk, or total). Adds suffix ``'_'+mode`` to the file name. 
        :type mode: str
        :param ages: Set of age bins, Gyr.
        :type ages: array-like
        
        :return: None. 
        """ 
        
        savepath = ''.join(('H(',self.w0,')R_age[',str(ages[0]),':',str(ages[-1]),',',
                                                  str(round(np.mean(np.diff(ages)),3)),']'))                                                     
        if 'between' not in self.kwargs or self.kwargs['between']==False:
            savepath += 'sgl'
        savepath += '_' + mode
        label = self.lb[mode]
        if 'mode_pop' in self.kwargs:
            savepath += '_' + self.kwargs['mode_pop']
            label += ', ' + self.Pn[self.kwargs['mode_pop']] 
        if 'tab' in self.kwargs:
            savepath += '_tab'
            label += ', custom population'
        if 'fig' in self.kwargs and self.kwargs['fig']==True:
            plt.savefig(os.path.join(self.a.T[''.join(('height',self.dir2))],''.join((savepath,self.fmt2))))
        else:
            np.savetxt(os.path.join(self.a.T[''.join(('height',self.dir))],''.join((savepath,self.fmt))),
                       np.hstack((self.a.R.reshape(self.a.Rbins,1),H)),
                       header=''.join(('R[kpc], h[pc] (',self.w,', ',label,') for ages = [',str(ages[0]),',',
                                       str(ages[-1]),'] Gyr with step = ',
                                       str(round(np.mean(np.diff(ages)),3)),' Gyr')))
        
        
    def hr_monomet_save(self,H,mode,mets):
        """
        Saves scale heights calculated for the mono-metallicity subpopulations (bins) 
        for the different Galactocentric distances. Output subfolder is ``a.T['height']``. 
        File name base is ``'H'``. 
        
        :param H: Table with scale heights (radial profiles in columns, metallicity changes 
            from column to column). Radial grid ``a.R`` is not included into **H**.
        :type H: array-like
        :param mode: Galactic component, can be ``'d'``, ``'dt'``, or ``'tot'`` 
            (thin disk, thin+thick disk, or total). Adds suffix ``'_'+mode`` to the file name. 
        :type mode: str
        :param mets: Set of metallicity bins.
        :type mets: array-like
        
        :return: None. 
        """ 
        
        savepath = ''.join(('H(',self.w0,')R_FeH[',
                                                str(round(mets[0],3)),':',str(round(mets[-1],3)),',',
                                                str(round(np.mean(np.diff(mets)),3)),']_',mode))
        label = self.lb[mode]
        if 'mode_pop' in self.kwargs:
            savepath += '_' + self.kwargs['mode_pop']
            label += ', ' + self.Pn[self.kwargs['mode_pop']] 
        if 'tab' in self.kwargs:
            savepath += '_tab'
            label += ', custom population'
        if 'fig' in self.kwargs and self.kwargs['fig']==True:
            plt.savefig(os.path.join(self.a.T[''.join(('height',self.dir2))],''.join((savepath,self.fmt2))))
        else:
            np.savetxt(os.path.join(self.a.T[''.join(('height',self.dir))],''.join((savepath,self.fmt))),
                       np.hstack((self.a.R.reshape(self.a.Rbins,1),H)),
                   header=''.join(('R[kpc], h[pc] (',self.w,', ',label,') for [Fe/H] = [',str(round(mets[0],3)),
                                   ',',str(round(mets[-1],3)),'] with step = ',
                                   str(round(np.mean(np.diff(mets)),3)))))

    
    def sigwz_save(self,profiles,mode):
        """
        Saves vertical profile of W velocity dispersion to the subfolder ``a.T['kinem']``. 
        File name base is ``'SigW'``. 
        
        :param profiles: Velocity dispersion profiles. Can include two tables - profiles at the different R 
            and at ``p.Rsun``, or a single table with the local profile. Tables contain only 
            velocity dispersion profiles, without the vertical grid column ``a.z``. 
        :type profiles: array-like
        :param mode: Galactic component, can be ``'d'``, ``'dt'``, or ``'tot'`` 
            (thin disk, thin+thick disk, or total). 
        :type mode: str
        
        :return: None. 
        """ 
        
        only_local = False
        if 'R' in self.kwargs and self.kwargs['R']==self.p.Rsun:
            only_local = True
            
        savepath0 = ''.join(('SigW(',self.w0,')z_R',str(self.p.Rsun),'_',mode))
        label = self.lb[mode]          
        if not only_local:
            savepath = ''.join(('SigW(',self.w0,')z_',mode))
                       
        if 'mode_pop' in self.kwargs:
            savepath0 += '_' + self.kwargs['mode_pop']
            label += ', ' + self.Pn[self.kwargs['mode_pop']] 
            if not only_local:
                savepath += '_' + self.kwargs['mode_pop']
        if 'tab' in self.kwargs:
            savepath0 += '_tab'
            label += ', custom population'
            if not only_local:
                savepath += '_tab'
        
        if only_local:
            sigwz0 = profiles
        else:
            sigwz, sigwz0 = profiles
            
        if 'fig' in self.kwargs and self.kwargs['fig']==True:
            if only_local:
                plt.savefig(os.path.join(self.a.T[''.join(('kin',self.dir2))],
                                         ''.join((savepath0,self.fmt2))))
            else:
                plt.savefig(os.path.join(self.a.T[''.join(('kin',self.dir2))],
                                         ''.join((savepath,self.fmt2))))
        else:
            if not only_local:
                np.savetxt(os.path.join(self.a.T[''.join(('kin',self.dir))],''.join((savepath,self.fmt))),
                           np.hstack((self.a.z.reshape(self.a.n,1),sigwz.T)),
                           header=''.join(('|z|[pc], <sigma_W>[km/s] (',self.w,', ',label,') at R = [',
                                           str(self.p.Rmin),',',str(self.p.Rmax),'] kpc with step = ',
                                           str(self.p.dR),' kpc')))
            np.savetxt(os.path.join(self.a.T[''.join(('kin',self.dir))],''.join((savepath0,self.fmt))),
                       np.stack((self.a.z,sigwz0),axis=-1),                   
                       header=''.join(('|z|[pc], <sigma_W>[km/s] (',self.w,', ',label,
                                       ') at R = ',str(self.p.Rsun),' kpc')))
        
    
    def sigwr_save(self,profiles,mode,zlim):
        """
        Saves radial W-velocity dispersion profiles to the subfolder ``a.T['kinem']``. 
        File name base is ``'SigW'``. 
        
        :param profiles: Radial velocity dispersion profiles. Tables contain only kinematics,
            without the radial grid column ``a.R``. 
        :type profiles: array-like
        :param mode: Galactic component, can be ``'d'``, ``'dt'``, or ``'tot'`` 
            (thin disk, thin+thick disk, or total). Adds suffix ``'_'+mode`` to the file name.             
        :type mode: str
        :param zlim: Range of heights to be considered, pc. Adds to the file name suffix ``'_z'`` 
            with the given heights (converted for shortness into kpc). 
        :type zlim: array-like 
        
        :return: None. 
        """ 
        
        if len(zlim)>2:
            savepath = ''.join(('SigW(',self.w0,')R_z[',str(zlim[0]/1e3),':',str(zlim[-1]/1e3),',',
                                str(round(np.mean(np.diff(zlim))/1e3,3)),']_',mode)) 
        else:
            savepath = ''.join(('SigW(',self.w0,')R_z[',str(zlim[0]/1e3),',',str(zlim[1]/1e3),']_',mode)) 
        label = self.lb[mode]
        if 'mode_pop' in self.kwargs:
            savepath += '_' + self.kwargs['mode_pop']
            label += ', ' + self.Pn[self.kwargs['mode_pop']] 
        if ('tab' in self.kwargs):
            savepath += '_tab'
            label += ', custom population'
        if 'fig' in self.kwargs and self.kwargs['fig']==True:
            plt.savefig(os.path.join(self.a.T[''.join(('kin',self.dir2))],''.join((savepath,self.fmt2))))
        else:
            try:
                out = np.hstack((self.a.R.reshape(self.a.Rbins,1),profiles))
            except:
                out = np.stack((self.a.R,profiles),axis=-1)
                
            np.savetxt(os.path.join(self.a.T[''.join(('kin',self.dir))],''.join((savepath,self.fmt))),
                       out,header=''.join(('R[kpc], <sigma_W>[km/s] (',self.w,', ',label,') at |z| = [',
                               str(zlim[0]),',',str(zlim[1]),'] pc')))                      
        
        
    def sigwr_monoage_save(self,profiles,mode,zlim,ages):
        """
        Saves radial W-velocity dispersion profiles calculated for the mono-age subpopulations. 
        Output subfolder is ``a.T['kinem']``. File name base is ``'SigW'``. 
        
        :param profiles: Radial velocity dispersion profiles (ages change from column to column). 
            Tables contain only kinematics, without the radial grid column ``a.R``. 
        :type profiles: array-like
        :param mode: Galactic component, can be ``'d'``, ``'dt'``, or ``'tot'`` 
            (thin disk, thin+thick disk, or total). Adds suffix ``'_'+mode`` to the file name. 
        :type mode: str
        :param zlim: Range of heights to be considered, pc. Adds to the file name suffix ``'_z'`` 
            with the given heights (converted for shortness into kpc). 
        :type zlim: array-like 
        :param ages: Set of age bins, Gyr.
        :type ages: array-like
        
        :return: None. 
        """ 
        
        savepath = ''.join(('SigW(',self.w0,')R_z[',str(zlim[0]/1e3),',',str(zlim[1]/1e3),                              
                            ']_age[',str(ages[0]),':',str(ages[-1]), ',',                                                          
                            str(round(np.mean(np.diff(ages)),3)),']'))
        if ('between' not in self.kwargs) and (self.kwargs['between']==False):
            savepath += 'sgl'  
        savepath += '_' + mode
        label = self.lb[mode]
        if 'mode_pop' in self.kwargs:
            savepath += '_' + self.kwargs['mode_pop']
            label += ', ' + self.Pn[self.kwargs['mode_pop']] 
        if 'tab' in self.kwargs:
            savepath += '_tab'
            label += ', custom population'
        if 'fig' in self.kwargs and self.kwargs['fig']==True:
            plt.savefig(os.path.join(self.a.T[''.join(('kin',self.dir2))],''.join((savepath,self.fmt2))))
        else:
            np.savetxt(os.path.join(self.a.T[''.join(('kin',self.dir))],''.join((savepath,self.fmt))),
                       np.hstack((self.a.R.reshape(self.a.Rbins,1),profiles)),           
                       header=''.join(('R[kpc], <sigma_W>[km/s] (',self.w,', ',label,') for ages = [',
                               str(ages[0]),',',str(ages[-1]),'] Gyr with step = ',
                               str(round(np.mean(np.diff(ages)),3)),' Gyr at |z| = [',
                               str(zlim[0]),',',str(zlim[1]),'] pc')))
                               
        
    def sigwr_monomet_save(self,profiles,mode,zlim,mets):
        """
        Saves radial W-velocity dispersion profiles calculated for the mono-metallicity subpopulations (bins). 
        Output subfolder is ``a.T['kinem']``. File name base is ``'SigW'``. 
        
        :param profiles: Radial velocity dispersion profiles (metallicity changes from column to column). 
            Tables contain only kinematics, without the radial grid column ``a.R``. 
        :type profiles: array-like
        :param mode: Galactic component, can be ``'d'``, ``'dt'``, or ``'tot'`` 
            (thin disk, thin+thick disk, or total). Adds suffix ``'_'+mode`` to the file name. 
        :type mode: str
        :param zlim: Range of heights to be considered, pc. Adds to the file name suffix ``'_z'`` 
            with the given heights (converted for shortness into kpc). 
        :type zlim: array-like 
        :param ages: Set of metallicity bins.
        :type ages: array-like
        
        :return: None. 
        """ 
        
        savepath = ''.join(('SigW(',self.w0,')R_z[',str(zlim[0]/1e3),',',str(zlim[1]/1e3),                                                                    
                            ']_FeH[',str(round(mets[0],3)),':',str(round(mets[-1],3)),',',
                            str(round(np.mean(np.diff(mets)),3)),']_',mode))
        label = self.lb[mode]
        if 'mode_pop' in self.kwargs:
            savepath += '_' + self.kwargs['mode_pop']
            label += ', ' + self.Pn[self.kwargs['mode_pop']] 
        if 'tab' in self.kwargs:
            savepath += '_tab' 
            label += ', custom population'
        if 'fig' in self.kwargs and self.kwargs['fig']==True:
            plt.savefig(os.path.join(self.a.T[''.join(('kin',self.dir2))],''.join((savepath,self.fmt2))))
        else:
            np.savetxt(os.path.join(self.a.T[''.join(('kin',self.dir))],''.join((savepath,self.fmt))),
                       np.hstack((self.a.R.reshape(self.a.Rbins,1),profiles.T)),           
                       header=''.join(('R[kpc], <sigma_W>[km/s] (',self.w,', ',label,') for [Fe/H] = [',
                               str(round(mets[0],3)),',',str(round(mets[-1],3)),'] with step = ',
                               str(round(np.mean(np.diff(mets)),3)),' at |z| = [',
                               str(zlim[0]),',',str(zlim[1]),'] pc')))
        
        
    def mean_quantity_save(self,profiles,mode,R,zlim,quantity):
        """
        Saves vertical profiles of some quantity calculated based on the stellar assemblies. 
        Output subfolder is ``a.T['pop']``. File name base is ``'Mean_'``. 
        
        :param profiles: Vertical profiles of **quantity**. 
            Tables do not contain vertical grid ``a.z``. 
        :type profiles: array-like
        :param mode: Galactic component, can be ``'d'``, ``'t'``, ``'sh'``, ``'dt'``, or ``'tot'`` 
            (thin disk, thick disk, halo, thin+thick disk, or total). 
            Adds suffix ``'_'+mode`` to the file name. 
        :type mode: str
        :param R: Galactocentric distance, kpc. Suffix ``'_R'`` with the given value of radis 
            is added to the output file name. 
        :type R: scalar
        :param zlim: Range of heights to be considered [*zmin,zmax*], pc. 
            Adds to the file name suffix ``'_z'`` 
            with the given heights (converted for shortness into kpc). 
        :type zlim: array-like 
        :param quantity: Name of the column in a stellar assemblies table for which 
            the profile was calculated; use ``'sigw'`` for velocity dispersion, 
            ``'age``` for age, and ``'FeH'`` for metallicity. 
        :type quantity: str
        
        :return: None. 
        """         
        
        if quantity=='age' or quantity=='FeH' or quantity=='sigw':
            if quantity=='age':
                Q3 = '<age>[Gyr]'
            if quantity=='FeH':
                Q3 = '<[Fe/H]>'
            if quantity=='sigw':
                Q3 = '<sigma_W>[km/s]'
        else:
            Q3 = '<' + quantity + '>[same units as in the stellar assemblies tables]'
        savepath = ''.join(('Mean_',quantity,'_R',str(R),'_z',
                                         str(zlim/1e3).replace(' ', ''),'_',mode,'_'))
        label = self.lb[mode]                        
        if 'tab' not in self.kwargs:
            savepath += self.kwargs['mode_pop']
        else:
            savepath += 'tab'
            label += ', custom population'
        if 'fig' in self.kwargs and self.kwargs['fig']==True:
            plt.savefig(os.path.join(self.a.T[''.join(('pop',self.pn[self.kwargs['mode_pop']],self.dir2))],
                                     ''.join((savepath,self.fmt2))))
        else:
            np.savetxt(os.path.join(self.a.T[''.join(('pop',self.pn[self.kwargs['mode_pop']],self.dir))],
                                    ''.join((savepath,self.fmt))),
                       np.hstack((self.a.z.reshape(self.a.n,1),profiles.T)),
                       header=''.join(('|z|[pc], ',Q3,' (',self.w,', ',label,'), R = ',
                                       str(R),' kpc, |z| =',str(zlim).replace(' ', ''),'pc')))
            
        
    def pops_in_volume_save(self,table,mode,R,mode_pop_name):
        """
        Saves table(s) with number densities of stellar assemblies for some volume.
        Output subfolder is ``a.T['pop']``. File name base is ``'SSP_R'``. 
        
        :param table: Saves stellar assemblies table(s) with a column ``'Nz'`` containing the number 
            of stars located in the specified volume (described by parameter ``'vln'`` in *kwargs*  
            of the class instance). Can be a single table or a list of tables, depending on **mode**. 
            Output of :func:`jjmodel.analysis.pops_in_volume`. 
        :type table: astropy table or list[astropy table]
        :param mode: Galactic component, can be ``'d'``, ``'t'``, ``'sh'``, ``'dt'``, or ``'tot'`` 
            (thin disk, thick disk, halo, thin+thick disk, or total). 
        :type mode: str
        :param R: Galactocentric distance, kpc. Suffix ``'_R'`` with the given value of radis 
            is added to the output file name. 
        :type R: scalar
        :param mode_pop_name: Short name of the population (see :func:`jjmodel.iof.tab_reader`) 
            or any custom name for the table. 
        :type mode_pop_name: str 
        
        :return: None. 
        """
        
        mode_iso = 'Padova'
        if 'mode_iso' in self.kwargs:
            mode_iso = self.kwargs['mode_iso']
        savepath = ''.join(('SSP_R',str(R)))
        if 'vln' in self.kwargs:
            savepath += '_' + self.kwargs['vln'] 
        else:
            savepath += '_volume'
        try:
            savedir = ''.join(('pop',self.pn[mode_pop_name],'tab'))
            tabname = self.Pn[mode_pop_name]
        except: 
            savedir = 'poptab'
            tabname = mode_pop_name
        if mode!='dt' and mode!='tot':
            savepath += '_' + mode
            if 'mode_pop' in self.kwargs:
                savepath += '_' + self.kwargs['mode_pop']
            if 'tab' in self.kwargs:
                savepath += '_tab' 
            savepath = os.path.join(self.a.T[savedir],''.join((savepath,'_',mode_iso,'.csv'))) 
            table.write(savepath,overwrite=True)
        else:
            savepath_d = savepath + '_d'
            savepath_t = savepath + '_t'
            if 'mode_pop' in self.kwargs:
                savepath_d += '_' + self.kwargs['mode_pop']
                savepath_t += '_' + self.kwargs['mode_pop']
            if 'tab' in self.kwargs:
                savepath_d += '_tab' 
                savepath_t += '_tab' 
            savepath_d = os.path.join(self.a.T[savedir],''.join((savepath_d,'_',mode_iso,'.csv'))) 
            savepath_t = os.path.join(self.a.T[savedir],''.join((savepath_t,'_',mode_iso,'.csv'))) 
            table[0].write(savepath_d,overwrite=True)
            table[1].write(savepath_t,overwrite=True)
            if mode=='tot':
                savepath_sh = savepath + '_sh'
                if 'mode_pop' in self.kwargs:
                    savepath_sh += '_' + self.kwargs['mode_pop']
                if 'tab' in self.kwargs:
                    savepath_sh += '_tab' 
                savepath_sh = os.path.join(self.a.T[savedir],''.join((savepath_sh,'_',mode_iso,'.csv'))) 
                table[2].write(savepath_sh,overwrite=True)
    
        
    def rz_map_save(self,profiles,mode):
        """
        Saves R-z map of some **quantity**.  
        Output subfolder is ``a.T['maps']``. File name is ``'Rho'``, ``'N'``, ``'Sigma'``, 
        or ``'SigmaN'``, depending on *kwargs* of the class instance. 
        
        :param profiles: Map of stellar mass or number density in R-z Galactic coordinates.
        :type profiles: 2d-array
        :param mode: Galactic component, can be ``'d'``, ``'t'``, ``'sh'``, ``'dt'``, or ``'tot'`` 
            (thin disk, thick disk, halo, thin+thick disk, or total). 
        :type mode: str
        
        :return: None. 
        """
        
        savepath = ''.join((self.Q0,'Rz_dz'))
        label = self.lb[mode]     
        if 'mode_pop' in self.kwargs:
            label += ', ' + self.Pn[self.kwargs['mode_pop']] 
        if 'tab' in self.kwargs:
            label += ', custom population'
        dz = '0.1'                   
        if 'dz' in self.kwargs:
            dz = str(self.kwargs['dz']/1e3) # in kpc
        savepath += dz
        headname = ''.join((self.Q1,'/[',dz,' kpc x ',str(self.p.dR),' kpc] (',label,')'))
        if 'ages' in self.kwargs:
            savepath += '_age' + str(self.kwargs['ages'])
            headname += ' for age = ' + str(self.kwargs['ages']) + ' Gyr'
        if 'mets' in self.kwargs:
            savepath += '_FeH' + str(self.kwargs['mets'])
            headname += ' for [Fe/H] = ' + str(self.kwargs['mets'])
        savepath += '_' + mode
        if 'mode_pop' in self.kwargs:
            savepath += '_' + self.kwargs['mode_pop']
        if 'tab' in self.kwargs:
            savepath += '_tab' 
        if 'fig' in self.kwargs and self.kwargs['fig']==True:
            plt.savefig(os.path.join(self.a.T[''.join(('maps',self.dir2))],''.join((savepath,self.fmt2))))
        else: 
            np.savetxt(os.path.join(self.a.T[''.join(('maps',self.dir))],
                       ''.join((savepath,self.fmt))),profiles.T,header=headname)
                   
    
    def rz_map_quantity_save(self,profiles,mode,quantity):
        """
        Saves R-z map of some **quantity**.  
        Output subfolder is ``a.T['maps']``. File name begins with **quantity**. 
        
        :param profiles: Map of the chosen **quantity** in R-z Galactic coordinates.
        :type profiles: 2d-array
        :param mode: Galactic component, can be ``'d'``, ``'t'``, ``'sh'``, ``'dt'``, or ``'tot'`` 
            (thin disk, thick disk, halo, thin+thick disk, or total). 
        :type mode: str
        :param quantity: Name of the column in a stellar assemblies table to which 
            the function has to be applied; for velocity dispersion use ``'sigw'``, 
            ``'age``` for age, and ``'FeH'`` for metallicity. 
        :type quantity: str
        
        :return: None. 
        """
        
        savepath = ''.join((quantity,'(',self.w0,')Rz_dz'))
        label = self.lb[mode]
        if 'mode_pop' in self.kwargs:
            label += ', ' + self.Pn[self.kwargs['mode_pop']] 
        if 'tab' in self.kwargs:
            label += ', custom population'
        dz = '0.1'
        if 'dz' in self.kwargs:
            dz = str(self.kwargs['dz']/1e3) # in kpc
        savepath += dz
        units = 'units'
        if quantity=='age':
            units = 'Gyr'
        if quantity=='FeH':
            units='dex'
        if quantity=='sigw':
            quantity = 'sigma_W'
            units='km/s'
        headname = ''.join((quantity,'[',units,' / ',dz,' pc x ',str(self.p.dR),' kpc] (',label,')'))
        if 'ages' in self.kwargs:
            savepath += '_age' + str(self.kwargs['ages'])
            headname += ' for age = ' + str(self.kwargs['ages']) + ' Gyr'
        if 'mets' in self.kwargs:
            savepath += '_FeH' + str(self.kwargs['mets'])
            headname += ' for [Fe/H] = ' + str(self.kwargs['mets'])
        savepath += '_' + mode
        if 'mode_pop' in self.kwargs:
            savepath += '_' + self.kwargs['mode_pop']
        if 'tab' in self.kwargs:
            savepath += '_tab' 
        if 'fig' in self.kwargs and self.kwargs['fig']==True:
            plt.savefig(os.path.join(self.a.T[''.join(('maps',self.dir2))],''.join((savepath,self.fmt2))))
        else: 
            np.savetxt(os.path.join(self.a.T[''.join(('maps',self.dir))],''.join((savepath,self.fmt))),
                       profiles.T,header=headname)
        
        
    def fw_save(self,profiles,mode,R,zlim):
        """
        Saves W-velocity dostribution function. 
        Output subfolder is ``a.T['kinem']``. File name base is ``'f(|W|)'``. 
        
        :param profiles: Normalized on area W-velocity distribution (histogram) and W-grid (bin centers), 
            :math:`\mathrm{km \ s^{-1}}`. 
        :type profiles: [1d-array, 1d-array]
        :type profiles: array-like
        :param mode: Galactic component, can be ``'d'``, ``'t'``, ``'sh'``, ``'dt'``, or ``'tot'`` 
            (thin disk, thick disk, halo, thin+thick disk, or total).
        :type mode_comp: str
        :param R: Galactocentric distance, kpc. 
        :type R: scalar 
        :param zlim: Range of heights [*zmin,zmax*], pc. 
        :type zlim: array-like 
        
        :return: None. 
        """
        
        savepath = ''.join(('f(|W|)(',self.w0,')'))
        label = self.lb[mode]
        if type(R)==list:
            savepath += '_R[' + str(R[0]) + ':' + str(R[-1]) + ',' + str(np.mean(np.diff(R))) + ']'
        else:
            savepath += '_R' + str(R)
        if len(zlim)>2:
            savepath += '_z['+str(zlim[0]/1e3)+':'+str(zlim[-1]/1e3)+','+str(np.mean(np.diff(zlim))/1e3)+']'
        else:
            savepath += '_z['+str(zlim[0]/1e3)+','+str(zlim[-1]/1e3)+']'                                
        if 'ages' in self.kwargs:
            if len(self.kwargs['ages'])>2:
                savepath += '_age[' + str(round(self.kwargs['ages'][0],2)) + ':' +\
                            str(round(self.kwargs['ages'][-1],2)) +\
                            ',' + str(round(np.mean(np.diff(self.kwargs['ages'])),2)) + ']'
            else:
                savepath += '_age[' + str(round(self.kwargs['ages'][0],2)) + ',' +\
                            str(round(self.kwargs['ages'][1],2)) + ']'
            label += ' for age = ' + str(self.kwargs['ages']) + ' Gyr'
        if 'mets' in self.kwargs:
            if len(self.kwargs['mets'])>2:
                savepath += '_FeH[' + str(round(self.kwargs['mets'][0],2)) + ':' +\
                            str(round(self.kwargs['mets'][-1],2)) +\
                            ',' + str(round(np.mean(np.diff(self.kwargs['mets'])),2)) + ']'
            else:
                savepath += '_FeH[' + str(round(self.kwargs['mets'][0],2)) + ',' +\
                            str(round(self.kwargs['mets'][1],2)) + ']'
            label += ' for [Fe/H] = ' + str(self.kwargs['mets'])
        savepath +=  '_' + mode
        if 'mode_pop' in self.kwargs:
            if type(self.kwargs['mode_pop'])==list:
                savepath += '_' + str(self.kwargs['mode_pop']).replace(' ','')
            else:
                savepath += '_' + self.kwargs['mode_pop']
        if 'tab' in self.kwargs:
            savepath += '_tab' 
            
        fw, wgrid = profiles
        if 'fig' in self.kwargs and self.kwargs['fig']==True:
            plt.savefig(os.path.join(self.a.T[''.join(('kin',self.dir2))],''.join((savepath,self.fmt2))))
        else:
            headname = ''.join(('|W|[km/s], f(|W|)(',self.w,', ',label,'), R = ',
                                str(R),' kpc, |z| =',str(zlim).replace(' ', ''),' pc'))
            if 'mode_pop' in self.kwargs:
                label += ', ' + self.Pn[self.kwargs['mode_pop']] 
            if 'tab' in self.kwargs:
                label += ', custom population'
            np.savetxt(os.path.join(self.a.T[''.join(('kin',self.dir))],''.join((savepath,self.fmt))),
                       np.stack((wgrid,fw),axis=-1),header=headname)
               
                    
    def hess_save(self,profiles,mode,mode_geom,bands,mag_range,mag_step):
        """
        Hess diagram for the simple volumes. 
        Output subfolder is ``a.T['hess']``. File name base is ``'Hess_'``. 
        
        :param profiles: Hess diagram, output of :func:`jjmodel.analysis.hess_simple`.  
        :type profiles: 2d-array 
        :param mode: Galactic component, can be ``'d'``, ``'t'``, ``'sh'``, ``'dt'``, or ``'tot'`` 
            (thin disk, thick disk, halo, thin+thick disk, or total).
        :type mode: str
        :param mode_geom: Modeled geometry. Can be ``'local_sphere'``, ``'local_cylinder'``, or ``'rphiz_box'``. 
        :type mode_geom: str
        :param bands: List of names of photometric columns. 
            Three column names must be given, e.g. ``['B','V','V']`` for (*B-V*) versus :math:`M_\mathrm{V}`. 
        :type bands: list[str]
        :param mag_range: Minimal and maximal magnitude along the x- and y-axis of the Hess 
            diagram, [[*x_min,x_max*],[*y_min,y_max*]]. 
        :type mag_range: list[list[scalar]]
        :param mag_step: Step along the x- and y-axis of the Hess diagram, [*dx,dy*]. 
        :type mag_step: array-like
        
        :return: None. 
        """
        
        savepath = ''.join(('Hess_(',bands[0],'-',bands[1],',',bands[2],')_',mode,'_',mode_geom,'_in[',
                            str(mag_range[0][0]),':',str(mag_range[0][1]),',',str(mag_step[0]),';',
                            str(mag_range[1][0]),':',str(mag_range[1][1]),',',str(mag_step[1]),']'))                                                 
        label = self.lb[mode]    
        if 'vln' in self.kwargs:
            label += '\n' + self.kwargs['vln']  
        else:
            label += ' in ' + mode_geom
        if 'mode_pop' in self.kwargs:
            savepath += '_' + self.kwargs['mode_pop']
            label += ', ' + self.Pn[self.kwargs['mode_pop']] 
        if 'tab' in self.kwargs:
            savepath += '_tab' 
            label += ', custom population'
        if 'fig' in self.kwargs and self.kwargs['fig']==True:
            plt.savefig(os.path.join(self.a.T[''.join(('hess',self.dir2))],''.join((savepath,self.fmt2))))
        else:
            np.savetxt(os.path.join(self.a.T[''.join(('hess',self.dir))],''.join((savepath,self.fmt))),
                   profiles.T,header=''.join(('N / [',str(mag_step[0]),' mag x ',str(mag_step[1]),
                                              ' mag] in (',bands[0],' - ',bands[1],', ', bands[2],
                                              ') for ',label)))                                             
                                   
                          
    def disk_brightness_save(self,profiles,mode,mode_geom,band):
        """
        Saves surface brightness or colour profile of the MW if it is viewed 
        edge-on or face-on. Output subfolder is ``a.T['pop']``. File name base is ``'MagR_'``. 
        
        :param profile: Disk surface brightness profile (:math:`\mathrm{mag \ arcsec^{-2}}`) or color 
            profile (mag), output of :func:`jjmodel.analysis.disk_brightness`.  
        :type profile: 1d-array
        :param mode_comp: Galactic component, can be ``'d'``, ``'t'``, ``'sh'``, ``'dt'``, or ``'tot'`` 
            (thin disk, thick disk, halo, thin+thick disk, or total). 
        :type mode_comp: str
        :param mode_geom: Modeled geometry. Disk orientation with respect to the observer: 
            ``'face-on'`` or ``'edge-on'``. 
        :type mode_geom: str
        :param bands: If a string, this parameter corresponds to the band for the surface brightness profile. 
            If it is a list, then **bands** gives the names of two bands to be used for the color profile - e.g. 
            ``['U','V']`` for *U-V*.
        :type bands: str or list
        
        :return: None. 
        """
        
        mode_iso = 'Padova'
        if 'mode_iso' in self.kwargs:
            mode_iso = self.kwargs['mode_iso']
        zlim = [0,self.p.zmax]     
        label = self.lb[mode]
        if type(band)!=str:
            bands = str(band[0]) + '-' + band[1] 
            quantity = ''.join((bands,'[mag]'))
        else:
            bands = band
            quantity = ''.join(('mu_',bands,'[mag/arcsec^2]'))
        if mode_geom=='face-on':
            savepath = ''.join(('MagR_',mode_geom,'_',bands,'_',mode_iso,'_',mode))                             
        else:
            if 'zlim' in self.kwargs:
                zlim = self.kwargs['zlim']
            savepath = ''.join(('MagR_',mode_geom,'_z[',str(zlim[0]/1e3),
                             ',',str(zlim[1]/1e3),']_',bands,'_',mode_iso,'_',mode))                                 
        if 'mode_pop' in self.kwargs:
            savepath += '_' + self.kwargs['mode_pop']
            label += ', ' + self.Pn[self.kwargs['mode_pop']] 
        if 'tab' in self.kwargs:
            savepath += '_tab' 
            label += ', custom population'
        if 'fig' in self.kwargs and self.kwargs['fig']==True:
            plt.savefig(os.path.join(self.a.T[''.join(('pop',self.dir2))],''.join((savepath,self.fmt2))))
        else:
            np.savetxt(os.path.join(self.a.T[''.join(('pop',self.dir))],''.join((savepath,self.fmt))),
                       np.stack((self.a.R,profiles),axis=0), 
                       header=''.join(('R[kpc], ',quantity,' ',label,' ',mode_geom,' at z = ',
                                       str(zlim).replace(' ', ''),
                                       ' pc calculated with ',mode_iso,' isochrones')))                                                           
        
    def input_local_save(self,inp):
        """
        Saves a set of the local model input data: SFR, mass loss function, and AMR 
        (for the thin and thick disk). 
        The output directory is ``a.T['inp']``, and file names are prescribed by :func:`jjmodel.iof.tab_sorter`. 
        
        :param inp: Collection of the input functions including SFR, AVR, AMR, and IMF.
        :type inp: dict   
        
        :return: None. 
        """
        
        p, a = self.p, self.a
        # SFR
        np.savetxt(tab_sorter('SFRd0',p,a.T),np.stack((a.t,inp['SFRd0']),axis=-1),
                   header='t[Gyr], SFR_thin.d[Msun/pc^2/Gyr]')
        np.savetxt(tab_sorter('NSFRd0',p,a.T),np.stack((a.t,inp['NSFRd0']),axis=-1),
                   header='t[Gyr], SFR_thin.d/<SFR_thin.d>')
        if p.pkey==1:
            np.savetxt(tab_sorter('Fp0',p,a.T),
                       np.hstack((a.t.reshape(a.jd,1),inp['Fp0'].T)),
                       header='t[Gyr], SFR_thin.d peak(s) fraction')
        np.savetxt(tab_sorter('SFRt0',p,a.T),np.stack((a.t[:a.jt],inp['SFRt0']),axis=-1),
                   header='t[Gyr], SFR_thick.d[Msun/pc^2/Gyr]')
        np.savetxt(tab_sorter('NSFRt',p,a.T),np.stack((a.t[:a.jt],inp['NSFRt0']),axis=-1),
                   header='t[Gyr], SFR_thick.d/<SFR_thick.d>')
        np.savetxt(tab_sorter('SFRtot0',p,a.T),np.stack((a.t,inp['SFRtot0']),axis=-1),
                   header='t[Gyr], SFR_tot[Msun/pc^2/Gyr]')
        np.savetxt(tab_sorter('NSFRtot0',p,a.T),np.stack((a.t,inp['NSFRtot0']),axis=-1),
                   header='t[Gyr], SFR_tot/<SFR_tot>')
        np.savetxt(tab_sorter('gd0',p,a.T),np.stack((a.t,inp['gd0']),axis=-1),
                   header='t[Gyr], fraction_in_stars&remnants_thin.d') 
        np.savetxt(tab_sorter('gt',p,a.T),np.stack((a.t[:a.jt],inp['gt']),axis=-1),
                   header='t[Gyr], fraction_in_stars&remnants_thick.d')
        np.savetxt(tab_sorter('AMRd0',p,a.T),np.stack((a.t,inp['AMRd0']),axis=-1),
                   header='t[Gyr], [Fe/H]_thin.d')
        np.savetxt(tab_sorter('AMRt',p,a.T),np.stack((a.t[:a.jt],inp['AMRt'][:a.jt]),axis=-1),
                   header='t[Gyr], [Fe/H]_thick.d')
        print('\nInput data saved.')
        
             
    def output_local_save(self,out):
        """
        Saves a set of the local model output data: AVR, scale heights for all model components, 
        vertical potential, force, and density profiles.  
        The output directory and file name for each quantity are prescribed by :func:`jjmodel.iof.tab_sorter` 
        and :meth:`jjmodel.iof.TabSaver.rhoz_save` (for density profiles). 
        
        :param out: Output of :func:`jjmodel.mwdisk.local_run`.
        :type out: dict
        
        :return: None. 
        """
        
        p, a = self.p, self.a
        # Vertical density profiles     
        rhoz0_profiles = np.stack((a.z,out['rhodtot'],out['rhog1'],out['rhog2'],
                                  out['rhot'],out['rhodh'],out['rhosh']),axis=-1)
        self.rhoz_save(rhoz0_profiles,p.Rsun)
        
        # Kinematics
        np.savetxt(tab_sorter('AVR0',p,a.T),np.stack((a.t,out['avr']),axis=-1),
                   header='t[Gyr], sigma_W[km/s]')
        
        # Scale heights
        np.savetxt(tab_sorter('Heffd0',p,a.T),
                   np.stack(([p.Rsun],[out['heffd']]),axis=-1), 
                   header='R[kpc], h_eff[pc]_thin.d')
        np.savetxt(tab_sorter('Hd0',p,a.T),np.stack((a.t,out['hd']),axis=-1),
                   header='t[Gyr], h[pc]_thin.d')
        if p.pkey==1:
            np.savetxt(tab_sorter('Hdp0',p,a.T),np.stack((out['sigp'],out['hdp']),axis=-1),
                       header='sigma_W[km/s], h[pc]_thin.dp')
        np.savetxt(tab_sorter('Ht0',p,a.T),
                   np.stack(([p.Rsun],[out['ht']]),axis=-1),header='R[kpc], h[pc]_thick.d')
        np.savetxt(tab_sorter('Hsh0',p,a.T),np.stack(([p.Rsun],[out['hsh']]),axis=-1),
                   header='R[kpc], h[pc]_st.halo')
        np.savetxt(tab_sorter('Hdh0',p,a.T),np.stack(([p.Rsun],[out['hdh']]),axis=-1),
                   header='R[kpc], h[pc]_DM')
        
        # Potential and vertical force
        np.savetxt(tab_sorter('Phi0',p,a.T),np.stack((a.z,out['phi']),axis=-1), header='z[pc], Phi[m^2/s^2]')
        np.savetxt(tab_sorter('Kz0',p,a.T),np.stack((a.z,out['Kzdtot'],out['Kzg1'],out['Kzg2'],
                                                     out['Kzt'],out['Kzdh'],out['Kzsh']),axis=-1),
                   header='z[pc], Q_thin.d, Q_mol.g, Q_atom.gas, Q_thick.d, Q_DM, '+\
                           'Q_st.halo with Q = Kz[km^2/s^2/kpc]')                             
        print('\nOutput data saved.')
        
        
    def output_extended_save(self,res):
        """ 
        Same as :meth:`jjmodel.iof.TabSaver.output_local`, but for the model extending over 
        some range of Galactocentric distances. 
        
        :param res: Output of :func:`jjmodel.mwdisk.extended_run`.
        :type res: dict
        
        :return: None. 
        """
        
        p, a = self.p, self.a
        # Vertical density profiles                    
        for i in range(a.Rbins):
            rhoz_profiles = np.stack((a.z,res[i]['rhodtot'],res[i]['rhog1'],res[i]['rhog2'],
                                 res[i]['rhot'],res[i]['rhodh'],res[i]['rhosh']),axis=-1)
            self.rhoz_save(rhoz_profiles,a.R[i])
            
            np.savetxt(tab_sorter('Kz',p,a.T,R=a.R[i]),
                       np.stack((a.z,res[i]['Kzdtot'],res[i]['Kzg1'],res[i]['Kzg2'],res[i]['Kzt'],                                                         
                                 res[i]['Kzdh'],res[i]['Kzsh']),axis=-1),
                       header='z[pc], Q_thin.d, Q_mol.g, Q_atom.gas, Q_thick.d, '+\
                               'Q_DM, Q_st.halo with Q = Kz[km^2/s^2/kpc]')    
            if p.pkey==1:
                np.savetxt(tab_sorter('Hdp',p,a.T,R=a.R[i]),
                           np.stack((res[i]['sigp'],res[i]['hdp']),axis=-1),                                 
                           header='sigma_W[km/s], h[pc]_thin.dp')
                               
        # Kinematics
        np.savetxt(tab_sorter('AVR',p,a.T),
                   np.hstack((a.t.reshape(a.jd,1),np.array([res[i]['avr'] for i in a.R_array]).T)), 
                   header=''.join(('t[Gyr], sigma_W[km/s]_thin.d at R = [',str(p.Rmin),',',str(p.Rmax),
                                   '] kpc with step dR = ',str(p.dR),' kpc')))
        np.savetxt(tab_sorter('Sigt',p,a.T), 
                   np.stack((a.R,[res[i]['sigt'] for i in a.R_array]),axis=-1), 
                   header='R[kpc], sigma_W[km/s]_thick.d')
        np.savetxt(tab_sorter('Sigg1',p,a.T), 
                   np.stack((a.R,[res[i]['sigg1'] for i in a.R_array]),axis=-1),
                   header='R[kpc], sigma_W[km/s]_mol.gas')
        np.savetxt(tab_sorter('Sigg2',p,a.T), 
                   np.stack((a.R,[res[i]['sigg2'] for i in a.R_array]),axis=-1), 
                   header='R[kpc], sigma_W[km/s]_atom.gas')
        np.savetxt(tab_sorter('Sige',p,a.T), 
                   np.stack((a.R,[res[i]['sige'] for i in a.R_array]),axis=-1), 
                   header='R[kpc], sigma_e[km/s]')
        
        # Scale heights
        np.savetxt(tab_sorter('Heffd',p,a.T),
                   np.stack((a.R,[res[i]['heffd'] for i in a.R_array]),axis=-1), 
                   header='R[kpc], h_eff[pc]_thin.d')
        np.savetxt(tab_sorter('Hsh',p,a.T), 
                   np.stack((a.R,[res[i]['hsh'] for i in a.R_array]),axis=-1), 
                   header='R[kpc], h[pc]_st.halo')
        np.savetxt(tab_sorter('Hdh',p,a.T), 
                   np.stack((a.R,[res[i]['hdh'] for i in a.R_array]),axis=-1), 
                   header='R[kpc], h[pc]_DM')
        np.savetxt(tab_sorter('Hd',p,a.T),
                   np.hstack((a.t.reshape(a.jd,1),np.array([res[i]['hd'] for i in a.R_array]).T)), 
                   header=''.join(('t[Gyr], h[pc]_thin.d at R = ',str([p.Rmin,p.Rmax]),
                                   ' kpc with step dR = ',str(p.dR),' kpc')))
        np.savetxt(tab_sorter('Ht',p,a.T),
                   np.stack((a.R,np.array([res[i]['ht'] for i in a.R_array])),axis=-1),                               
                   header='R[kpc], h[pc]_thick.d')
        # Potential 
        np.savetxt(tab_sorter('Phi',p,a.T),np.hstack((a.z.reshape(a.n,1),
                   np.array([res[i]['phi'] for i in a.R_array]).T)), 
                   header=''.join(('z[pc], Phi[m^2/s^2] at R = ',str([p.Rmin,p.Rmax]),
                                   ' kpc with step dR = ',str(p.dR),' kpc')))
        
        print('\nOutput data saved.')
        
    
    def input_extended_save(self,inp):
        """ 
        Same as :meth:`jjmodel.iof.TabSaver.input_local`, but for the model extending over 
        some range of Galactocentric distances. 
        
        :param inp: Collection of the input functions including SFR, AVR, AMR, and IMF.
        :type inp: dict   
        
        :return: None. 
        """
        
        p, a = self.p, self.a
        # Input data
        np.savetxt(tab_sorter('SFRd',p,a.T),np.hstack((a.t.reshape(a.jd,1),inp['SFRd'].T)),
                   header=''.join(('t[Gyr], SFR_thin.d[Msun/pc^2/Gyr] at R = ',
                                   str([p.Rmin,p.Rmax]),' kpc with step dR = ',str(p.dR),' kpc')))
        np.savetxt(tab_sorter('NSFRd',p,a.T), np.hstack((a.t.reshape(a.jd,1),inp['NSFRd'].T)),
                   header=''.join(('t[Gyr], SFR_thin.d/<SFR_thin.d> at R = ',
                                   str([p.Rmin,p.Rmax]),' kpc with step dR = ',str(p.dR),' kpc')))
        if p.pkey==1:
            for i in range(a.Rbins):
                np.savetxt(tab_sorter('Fp',p,a.T,R=a.R[i]),
                           np.hstack((a.t.reshape(a.jd,1),inp['Fp'][i].T)),
                           header='t[Gyr], SFR_thin.d peak(s) fraction')
        np.savetxt(tab_sorter('SFRt',p,a.T),np.hstack((a.t[:a.jt].reshape(a.jt,1),inp['SFRt'].T)),
                   header=''.join(('t[Gyr], SFR_thick.d[Msun/pc^2/Gyr] at R = ',str([p.Rmin,p.Rmax]),
                                   ' kpc with step dR = ',str(p.dR),' kpc')))
        np.savetxt(tab_sorter('SFRtot',p,a.T),np.hstack((a.t.reshape(a.jd,1),inp['SFRtot'].T)),
                   header=''.join(('t[Gyr], SFR_tot[Msun/pc^2/Gyr] at R = ',str([p.Rmin,p.Rmax]),
                                   ' kpc with step dR = ',str(p.dR),' kpc')))
        np.savetxt(tab_sorter('NSFRtot',p,a.T),np.hstack((a.t.reshape(a.jd,1),inp['NSFRtot'].T)),
                   header=''.join(('t[Gyr], SFR_tot/<SFR_tot> at R = ',str([p.Rmin,p.Rmax]),
                                   ' kpc with step dR = ',str(p.dR),' kpc')))
        np.savetxt(tab_sorter('AMRd',p,a.T),np.hstack((a.t.reshape(a.jd,1),inp['AMRd'].T)),
                   header=''.join(('t[Gyr], [Fe/H]_thin.d at R = ',
                                   str([p.Rmin,p.Rmax]),' kpc with step dR = ',str(p.dR),' kpc')))
        np.savetxt(tab_sorter('SR',p,a.T),np.hstack((a.R.reshape(a.Rbins,1),np.array(inp['SigmaR']).T)),                  
                   header='R[kpc], Q_thin.d, Q_mol.g, Q_atom.g, Q_thick.d, '+\
                           'Q_DM, Q_st.halo with Q = Sigma[Msun/pc^2]')
        np.savetxt(tab_sorter('Hg1',p,a.T),np.stack((a.R,inp['hg1']),axis=-1),
                   header='R[kpc], h[pc]_mol.gas')
        np.savetxt(tab_sorter('Hg2',p,a.T),np.stack((a.R,inp['hg2']),axis=-1),
                   header='R[kpc], h[pc]_atom.gas')
        np.savetxt(tab_sorter('gd',p,a.T),np.hstack((a.t.reshape(a.jd,1),np.array(inp['gd']).T)),
                   header='t[Gyr], fraction_in_stars&remnants_thin.d at R = '+str([p.Rmin,p.Rmax])
                           +' kpc with step dR = '+str(p.dR)+' kpc') 
        
        print('\nInput data saved.')
        
        

    