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
    Builds directory tree for the model realization defined by the
    input parameters in the parameter file. 
    
    Parameters
    ----------
    p : namedtuple
        Set of the model parameters from the parameter file.
    **kwargs : dict, optional keyword arguments
        make: boolean
            If True, the directory tree is actually created, 
            otherwise only all the folder names are defined. 

    Returns
    -------
    T : dict
        Directory tree for the model output.
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
    Sorts all kinds of the output into directories, where it will
    be stored and from where it can be later read out again. Each
    quantity is also assigned with a unique name depending on 
    the parameters it was calculated with. 
    
    Parameters
    ----------
    name : srt
        Short name of the quantity.
    p : namedtuple
        Set of the model parameters from the parameter file.
    T : dict
        Directory tree.
    **kwargs : dict, optional keyword arguments
        R : scalar
        ages : array_like
        mets : array_like
        sigma : boolean
        zlim : tuple
        mode : str
        mode_iso : str
        band : str

    Returns
    -------
    TYPE
        DESCRIPTION.
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
    
    if 'R' in kwargs:
        namespace['rhoz'] = os.path.join(T['denstab'],''.join(('Rhoz_R',str(kwargs['R']),'.txt')))
        namespace['Kz'] = os.path.join(T['fitab'],''.join(('Kz_R',str(kwargs['R']),'.txt')))
        namespace['Fp'] = os.path.join(T['inptab'],''.join(('F_dp_R',str(kwargs['R']),'.txt')))
        namespace['Hdp'] = os.path.join(T['heighttab'],''.join(('H_dp_R',str(kwargs['R']),'.txt')))
        
    return namespace[name]
    
 
    
def tab_reader(names,p,T,**kwargs):
    """
    Reads model output data from files.
    
    Parameters
    ----------
    names : str
        DESCRIPTION.
    p : namedtuple
        Set of the model parameters from the parameter file.
    T : dict
        Directory tree.
    **kwargs : dict, optional keyword arguments
        Same as for tab_sorter. 

    Returns
    -------
    tables : array_like
        DESCRIPTION.
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


class TabSaver():
    """
    """
    def __init__(self,p,a,**kwargs):
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
        savepath = 'FiRz_iso'
        if 'fig' in self.kwargs and self.kwargs['fig']==True:
            plt.savefig(os.path.join(self.a.T[''.join(('fi',self.dir2))],''.join((savepath,self.fmt2))))
        np.savetxt(os.path.join(self.a.T[''.join(('fi',self.dir))],''.join((savepath,self.fmt))),
                   profiles,header=''.join(('|z|[pc], ','Phi[m^2/s^2] at R = [',str(self.p.Rmin),',',
                               str(self.p.Rmax),'] kpc with step dR = ',str(self.p.dR),' kpc')))  
                       

    def rot_curve_save(self,profiles):
        np.savetxt(os.path.join(self.a.T['kintab'],''.join(('Vc_R',self.fmt))),
                   profiles,header='R[kpc], Q_tot, Q_bulge, Vc_thin.d, Q_thick.d, Q_mol.g, '+\
                                        'Q_at.g, Q_DM, Q_st.halo with Q = Vc[km/s]')  

            
    def poptab_save(self,table,mode,mode_iso,R,mode_pop_name):
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
                                         str(zlim).replace(' ', ''),'_',mode,'_'))
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
        
        

    
