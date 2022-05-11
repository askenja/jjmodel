#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 20:54:52 2022

@author: skevja
"""
import os 
import shutil
import urllib
import requests
from .__init__ import localpath
from .tools import Timer



def isochrone_loader(download_mode,**kwargs):
    """
    Automatically downloads isochrone grids adapted for the JJ model. 
    
    :param download_mode: Can be 'Padova', 'MIST', or 'BaSTI'. 
    :type download_mode: str 
    :param photometry: Optional, photometry of the chosen isochrone library. 
    Currently, the only option for Padova and MIST is 'multiband' and 
        for BaSTI - 'gaiaedr3'. More photometric systems will be added in the future. 
        If this parameter is not specified, all available photometric sets for the chosen 
        stellar evolution library will be loaded. 
    :type photometry: str or list[str]
    
    :return: None. 
    """
    
    main_link = 'https://heidata.uni-heidelberg.de/api/access/datafile/'+\
                ':persistentId?persistentId=doi:10.11588/data/ZCXHOE/'
    
    iso_folder_ids = {'Padova':{'multiband':'7XCJQP'},
                      'MIST':{'multiband':'3T8WV4'},
                      'BaSTI':{'gaiaedr3':'RIGNVU'}
                     }
   
    if ('photometry' in kwargs):
        download_photometry = [kwargs['photometry']]
    else:
        download_photometry = list(iso_folder_ids[download_mode].keys())
                
         
    print('Loading isochrones: ',download_mode,'...')
    
    clock = Timer()
    t1 = clock.start()
    
    iso_folder = os.path.join(localpath,'input','isochrones')
    if not os.path.isdir(iso_folder):
        os.mkdir(iso_folder)
    
    for k in range(len(download_photometry)):
        
        print('\tphotometry: ',download_photometry[k])
        
        destination_folder = os.path.join(localpath,'input','isochrones',download_mode)
                                
        zip_file = ''.join((destination_folder,'.zip'))
        
        urllib.request.urlretrieve(main_link+iso_folder_ids[download_mode][download_photometry[k]],zip_file)
        print('\tdata downloaded, please wait...')
        
        shutil.unpack_archive(zip_file,destination_folder)
        print('\tarchive unpacked')
        
        os.remove(zip_file)
        print('\tzip file removed\n')
        
        print(clock.stop(t1))
        t1 = clock.start()


