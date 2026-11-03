"""
Created on Mon Jan  3 20:54:52 2022

@author: skevja
"""
import os 
import gdown
import shutil
from . import localpath


def isochrone_loader(iso_type,model):
    """
    Automatically downloads isochrone grids adapted for the JJ model. 
    
    :param iso_type: Can be 'main' or 'wd'. 
    :type download_mode: str 
    :param model: Can be 'Padova', 'MIST', or 'BaSTI' for 'main' and 
        'LPCODE', 'Montreal', or 'BaSTI' for 'wd'. 
    :type model: str 
    
    :return: None. 
    """

    links = {
        'main':{
            'Padova':'https://drive.google.com/file/d/1mH2mKAdhQBI5JrgsGVko2G7iH5bZUk-I/view?usp=sharing',
            'MIST':'https://drive.google.com/file/d/1Pv5OgAf92eOmqYiK0ONprRfKTUSh3_2I/view?usp=sharing',
            'BaSTI':'https://drive.google.com/file/d/1mnuQrFb1TrYnxGIFl7zj8O_0e4V7H7jv/view?usp=sharing'
            },
        'wd':{
            'LPCODE':'https://drive.google.com/file/d/13rjsl00muTcpex9NENfJuQi-KJDF-mME/view?usp=sharing',
            'Montreal':'https://drive.google.com/file/d/1PzIWb6FT66-Xd95irmpF4ojwZ-TKSqNK/view?usp=sharing',
            'BaSTI':'https://drive.google.com/file/d/1lKywK0O3SN_jNttShiC0lD5F4kPmTo5L/view?usp=sharing'
            }
        }

    download_link = links[iso_type][model]

    iso_folder = os.path.join(localpath,'input','isochrones')
    if not os.path.isdir(iso_folder):
        os.mkdir(iso_folder)

    destination_base = os.path.join(localpath,'input','isochrones',iso_type)
    destination_folder = os.path.join(destination_base,model)
    os.makedirs(destination_folder,exist_ok=True)
                            
    zip_file = ''.join((destination_folder,'.zip'))

    gdown.download(
            id=download_link.split("/d/")[1].split("/")[0],
        output=zip_file,
        quiet=False,
        resume=False,
    )
    
    print('Unpacking archive... ',end='')
    
    shutil.unpack_archive(zip_file,destination_base)
    print('ok')
    
    os.remove(zip_file)
    print('Done')